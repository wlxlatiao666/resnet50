import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
import socket
import threading

# 定义参数服务器角色
PS_RANK = 0  # 第0个进程作为参数服务器

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组，使用gloo后端以便CPU也能工作
    # (PS架构通常用在CPU集群上，但这里我们同时支持CPU和GPU)
    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ParameterServer:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()
        
    def average_gradients(self, worker_count):
        """接收所有worker的梯度，平均后更新模型参数"""
        with self.lock:
            for param in self.model.parameters():
                # 将梯度初始化为0
                param.grad = torch.zeros_like(param.data)
                
                # 从每个worker接收梯度并累加
                for worker_rank in range(1, worker_count + 1):
                    grad_tensor = torch.zeros_like(param.data)
                    dist.recv(grad_tensor, src=worker_rank)
                    param.grad.add_(grad_tensor)
                
                # 计算梯度平均值
                param.grad.div_(worker_count)
            
            # 更新模型参数
            return True

def ps_worker(rank, world_size, args):
    setup(rank, world_size)
    worker_count = world_size - 1  # 减去参数服务器
    
    if rank == PS_RANK:
        # 参数服务器进程逻辑
        print(f"参数服务器启动在rank={rank}")
        
        # 初始化模型
        model = models.resnet50(weights=None)
        if torch.cuda.is_available():
            model = model.cuda()
        
        # 初始化参数服务器
        param_server = ParameterServer(model)
        
        # 定义优化器
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
        
        # 学习率调度器
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

        # TensorBoard日志
        writer = SummaryWriter(log_dir='./logs/parameter_server')

        best_acc = 0
        results = {
            'train_time': [],
            'train_loss': [],
            'train_acc': [],
            'val_time': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # 训练循环
        for epoch in range(1, args.epochs + 1):
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
            start_time = time.time()
            epoch_loss = 0
            
            # 广播当前模型参数给所有worker
            for param in model.parameters():
                for worker_rank in range(1, world_size):
                    dist.send(param.data, dst=worker_rank)
            
            # 等待所有worker完成训练并发送梯度
            steps = 0
            running_loss = 0
            while steps < len(train_dataset) // (args.batch_size * worker_count):
                # 接收并平均梯度
                if param_server.average_gradients(worker_count):
                    # 执行优化器步骤
                    optimizer.step()
                    optimizer.zero_grad()
                    steps += 1
                
                # 接收训练损失
                loss_tensor = torch.zeros(1)
                for worker_rank in range(1, world_size):
                    dist.recv(loss_tensor, src=worker_rank)
                    running_loss += loss_tensor.item() / worker_count
                
                if steps % args.log_interval == 0:
                    print(f'Train Epoch: {epoch} [{steps * args.batch_size * worker_count}/{len(train_dataset)} '
                          f'({100. * steps / (len(train_dataset) // (args.batch_size * worker_count)):.0f}%)]\t'
                          f'Loss: {running_loss / args.log_interval:.4f}')
                    writer.add_scalar('Loss/train_step', running_loss / args.log_interval, 
                                     epoch * (len(train_dataset) // (args.batch_size * worker_count)) + steps)
                    running_loss = 0
            
            train_time = time.time() - start_time
            results['train_time'].append(train_time)
            
            # 执行验证
            model.eval()
            val_loss = 0
            correct = 0
            val_start_time = time.time()
            
            # 广播开始验证信号
            signal = torch.tensor([1])
            for worker_rank in range(1, world_size):
                dist.send(signal, dst=worker_rank)
            
            # 接收验证结果
            val_loss_tensor = torch.zeros(1)
            val_correct_tensor = torch.zeros(1, dtype=torch.long)
            val_total_tensor = torch.zeros(1, dtype=torch.long)
            
            for worker_rank in range(1, world_count):
                dist.recv(val_loss_tensor, src=worker_rank)
                val_loss += val_loss_tensor.item()
                
                dist.recv(val_correct_tensor, src=worker_rank)
                correct += val_correct_tensor.item()
                
                dist.recv(val_total_tensor, src=worker_rank)
                total = val_total_tensor.item()
            
            val_loss /= worker_count
            val_acc = 100. * correct / total
            val_time = time.time() - val_start_time
            
            results['val_time'].append(val_time)
            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)
            
            print(f"验证集 - 平均损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # 更新学习率
            lr_scheduler.step()
            
            # 保存模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"{args.save_path}/resnet50_ps_best.pth")
                print(f"保存最佳模型, 精度: {best_acc:.2f}%")
            
            # 每个epoch保存一次
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_acc': best_acc,
            }, f"{args.save_path}/resnet50_ps_epoch{epoch}.pth")
        
        # 保存最终模型
        torch.save(model.state_dict(), f"{args.save_path}/resnet50_ps_final.pth")
        
        # 打印最终结果
        print(f"\n训练完成!")
        print(f"最佳验证准确率: {best_acc:.2f}%")
        print(f"平均每个epoch训练时间: {sum(results['train_time'])/args.epochs:.2f}秒")
        print(f"平均每个epoch验证时间: {sum(results['val_time'])/args.epochs:.2f}秒")
    
    else:
        # Worker进程逻辑
        worker_id = rank - 1  # 从0开始的worker ID
        print(f"Worker {worker_id} 启动在rank={rank}")
        
        # 固定随机种子以便结果可复现
        torch.manual_seed(42 + worker_id)
        if torch.cuda.is_available():
            torch.cuda.set_device(worker_id % torch.cuda.device_count())
            torch.cuda.manual_seed(42 + worker_id)
        
        # 数据预处理
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 加载数据集
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_path, 'train'),
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            os.path.join(args.data_path, 'val'),
            transform=val_transform
        )

        # 创建分布式采样器 (仅worker之间分配数据)
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=worker_count, rank=worker_id, shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=worker_count, rank=worker_id, shuffle=False
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=4, pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, sampler=val_sampler,
            num_workers=4, pin_memory=True
        )

        # 定义模型
        model = models.resnet50(weights=None)
        device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}" 
                              if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_sampler.set_epoch(epoch)  # 确保在每个epoch中数据被打乱
            
            # 从参数服务器接收最新模型参数
            for param in model.parameters():
                recv_param = torch.zeros_like(param.data)
                dist.recv(recv_param, src=PS_RANK)
                param.data.copy_(recv_param)
            
            # 训练步骤
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                output = model(data)
                loss = criterion(output, target)
                
                # 反向传播计算梯度
                loss.backward()
                
                # 发送梯度到参数服务器
                for param in model.parameters():
                    if param.grad is not None:
                        dist.send(param.grad, dst=PS_RANK)
                
                # 发送损失到参数服务器
                loss_tensor = torch.tensor([loss.item()])
                dist.send(loss_tensor, dst=PS_RANK)
                
                # 清零梯度
                model.zero_grad()
            
            # 等待参数服务器的验证信号
            signal = torch.zeros(1)
            dist.recv(signal, src=PS_RANK)
            
            # 执行验证
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item() * target.size(0)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            # 发送验证结果到参数服务器
            val_loss_tensor = torch.tensor([val_loss])
            dist.send(val_loss_tensor, dst=PS_RANK)
            
            val_correct_tensor = torch.tensor([correct], dtype=torch.long)
            dist.send(val_correct_tensor, dst=PS_RANK)
            
            val_total_tensor = torch.tensor([total], dtype=torch.long)
            dist.send(val_total_tensor, dst=PS_RANK)
    
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/path/to/imagenet', help='ImageNet数据集路径')
    parser.add_argument('--batch-size', type=int, default=128, help='每个worker的批次大小')
    parser.add_argument('--epochs', type=int, default=90, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--log-interval', type=int, default=10, help='日志打印间隔')
    parser.add_argument('--save-path', default='./checkpoints', help='模型保存路径')
    args = parser.parse_args()
    
    # 创建保存路径
    os.makedirs(args.save_path, exist_ok=True)
    
    # 使用torch.multiprocessing启动多个进程
    world_size = torch.cuda.device_count() + 1  # +1是因为有一个参数服务器
    mp.spawn(ps_worker, args=(world_size, args), nprocs=world_size, join=True)