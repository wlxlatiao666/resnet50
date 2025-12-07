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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ParameterServer:
    def __init__(self, model):
        self.model = model
        self.params = [p.data for p in model.parameters()]
        self.grads = [torch.zeros_like(p.data) for p in model.parameters()]
        self.worker_count = 0

    def push_parameters(self):
        # 将参数广播到所有工作节点
        for idx, param in enumerate(self.params):
            dist.broadcast(param, src=0)
    
    def pull_gradients(self, worker_rank):
        # 接收工作节点的梯度
        for idx, grad in enumerate(self.grads):
            dist.recv(grad, src=worker_rank)
            
        self.worker_count += 1
    
    def update(self, optimizer):
        # 将收集到的梯度应用到模型中
        for param, grad in zip(self.model.parameters(), self.grads):
            param.grad = grad.clone() / self.worker_count
        
        # 执行优化步骤
        optimizer.step()
        optimizer.zero_grad()
        
        # 重置梯度和计数器
        for grad in self.grads:
            grad.zero_()
        self.worker_count = 0

class ParameterClient:
    def __init__(self, model, rank):
        self.model = model
        self.rank = rank
    
    def pull_parameters(self):
        # 从参数服务器获取参数
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
    
    def push_gradients(self):
        # 发送梯度到参数服务器
        for param in self.model.parameters():
            if param.grad is not None:
                dist.send(param.grad.data, dst=0)

def train_ps(rank, world_size, args):
    setup(rank, world_size)
    
    if rank == 0:
        print(f"使用Parameter-Server架构, {world_size-1} 个工作节点")
        
    # 固定随机种子以便结果可复现
    torch.manual_seed(42 + rank)
    
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

    # 创建分布式采样器 - 只用于工作节点
    if rank > 0:  # 工作节点
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size-1, rank=rank-1, shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size-1, rank=rank-1, shuffle=False
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
    else:  # 参数服务器
        # 参数服务器不需要数据加载器，但为了验证需要一个
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

    # 定义模型
    model = models.resnet50(weights=None)
    if torch.cuda.is_available() and rank > 0:  # 工作节点使用GPU
        device = torch.device(f'cuda:{rank-1}')
        model = model.to(device)
    else:  # 参数服务器使用CPU
        device = torch.device('cpu')
        model = model.to(device)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建Parameter Server或Client
    if rank == 0:  # 参数服务器
        server = ParameterServer(model)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                             momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
        writer = SummaryWriter(log_dir='./logs/parameter_server')
    else:  # 工作节点
        client = ParameterClient(model, rank)
        
    # 同步障碍
    dist.barrier()
    
    def train_worker(epoch):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 从参数服务器获取最新参数
            client.pull_parameters()
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 将梯度发送到参数服务器
            client.push_gradients()
            
            # 统计
            _, predicted = output.max(1)
            total_samples += target.size(0)
            running_correct += predicted.eq(target).sum().item()
            running_loss += loss.detach().item() * target.size(0)
            
            # 打印信息
            if batch_idx % args.log_interval == 0:
                print(f'Worker {rank}, Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.detach().item():.4f}')
        
        # 计算平均损失和准确率
        avg_loss = running_loss / total_samples
        accuracy = 100. * running_correct / total_samples
        
        return avg_loss, accuracy
    
    def train_server(epoch):
        total_workers = world_size - 1
        
        # 将参数广播到所有工作节点
        server.push_parameters()
        
        # 等待所有工作节点的梯度
        for worker_rank in range(1, world_size):
            server.pull_gradients(worker_rank)
        
        # 更新模型参数
        server.update(optimizer)
        
        return model
    
    def validate(epoch):
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
        
        val_loss /= total
        accuracy = 100. * correct / total
        
        if rank == 0:  # 参数服务器记录日志
            print(f'Validation Epoch: {epoch}, Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', accuracy, epoch)
        
        return val_loss, accuracy
    
    # 打印模型信息
    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        print(f"批次大小: {args.batch_size} (每个工作节点), 全局批次大小: {args.batch_size * (world_size-1)}")
        print(f"数据集大小: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params/1e6:.2f}M")

    best_acc = 0
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        if rank == 0:
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # 同步所有进程开始训练
        dist.barrier()
        
        # 训练
        train_start = time.time()
        
        if rank == 0:  # 参数服务器
            model = train_server(epoch)
            lr_scheduler.step()
        else:  # 工作节点
            train_loss, train_acc = train_worker(epoch)
        
        train_time = time.time() - train_start
        
        # 确保所有节点完成训练后再进行验证
        dist.barrier()
        
        # 验证 - 只在参数服务器执行
        if rank == 0:
            val_start = time.time()
            val_loss, val_acc = validate(epoch)
            val_time = time.time() - val_start
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch} 完成, 总用时: {epoch_time:.2f}s, "
                  f"训练: {train_time:.2f}s, 验证: {val_time:.2f}s")
            
            # 记录到TensorBoard
            writer.add_scalar('Time/epoch', epoch_time, epoch)
            writer.add_scalar('Time/train', train_time, epoch)
            writer.add_scalar('Time/val', val_time, epoch)
            
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
    
    if rank == 0:
        # 保存最终模型
        torch.save(model.state_dict(), f"{args.save_path}/resnet50_ps_final.pth")
        
        # 打印最终结果
        print(f"\n训练完成!")
        print(f"最佳验证准确率: {best_acc:.2f}%")
    
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/path/to/imagenet', help='ImageNet数据集路径')
    parser.add_argument('--batch-size', type=int, default=128, help='每个工作节点的批次大小')
    parser.add_argument('--epochs', type=int, default=90, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--log-interval', type=int, default=10, help='日志打印间隔')
    parser.add_argument('--save-path', default='./checkpoints', help='模型保存路径')
    args = parser.parse_args()
    
    # 使用多进程启动训练，+1是为了参数服务器
    world_size = torch.cuda.device_count() + 1
    mp.spawn(train_ps, args=(world_size, args), nprocs=world_size, join=True)