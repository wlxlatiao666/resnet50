import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_worker(gpu, world_size, args):
    rank = gpu
    setup(rank, world_size)
    
    if rank == 0:
        print(f"使用 {world_size} 个 GPU 进行训练")
        
    # 固定随机种子以便结果可复现
    torch.manual_seed(42)
    torch.cuda.set_device(rank)
    torch.cuda.manual_seed(42)
    
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

    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=8, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=8, pin_memory=True
    )

    # 定义模型
    model = models.resnet50(weights=None)
    model = model.cuda(rank)
    
    # 将模型包装到DDP中
    model = DDP(model, device_ids=[rank])
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)

    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

    # TensorBoard日志(仅在主进程中)
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir='./logs/ddp')

    def train(epoch):
        model.train()
        train_sampler.set_epoch(epoch)  # 确保在每个epoch中数据被打乱
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(rank), target.cuda(rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = output.max(1)
            total_samples += target.size(0)
            running_correct += predicted.eq(target).sum().item()
            running_loss += loss.detach().item() * target.size(0)
            
            if rank == 0 and batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data) * world_size}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.detach().item():.6f}')
        
        # 聚合所有进程的损失和准确率
        running_loss = torch.tensor(running_loss).cuda(rank)
        running_correct = torch.tensor(running_correct).cuda(rank)
        total_samples = torch.tensor(total_samples).cuda(rank)
        
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        
        epoch_loss = running_loss.item() / total_samples.item()
        epoch_acc = 100. * running_correct.item() / total_samples.item()
        epoch_time = time.time() - start_time
        
        if rank == 0:
            print(f'Train Epoch: {epoch}\tLoss: {epoch_loss:.4f}\tAcc: {epoch_acc:.2f}%\tTime: {epoch_time:.2f}s')
            
            if writer:
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                writer.add_scalar('Time/train', epoch_time, epoch)
        
        return epoch_time, epoch_loss, epoch_acc

    def validate(epoch):
        model.eval()
        val_loss = 0
        correct = 0
        total_samples = 0
        start_time = time.time()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(rank), target.cuda(rank)
                output = model(data)
                val_loss += criterion(output, target).item() * target.size(0)
                _, predicted = output.max(1)
                total_samples += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        # 聚合所有进程的损失和准确率
        val_loss = torch.tensor(val_loss).cuda(rank)
        correct = torch.tensor(correct).cuda(rank)
        total_samples = torch.tensor(total_samples).cuda(rank)
        
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        
        val_loss = val_loss.item() / total_samples.item()
        val_acc = 100. * correct.item() / total_samples.item()
        val_time = time.time() - start_time
        
        if rank == 0:
            print(f'Validation Epoch: {epoch}\tLoss: {val_loss:.4f}\tAcc: {val_acc:.2f}%\tTime: {val_time:.2f}s')
            
            if writer:
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        return val_time, val_loss, val_acc

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        print(f"批次大小: {args.batch_size} (每个GPU), 全局批次大小: {args.batch_size * world_size}")
        print(f"数据集大小: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params/1e6:.2f}M")

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
        if rank == 0:
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # 训练
        train_time, train_loss, train_acc = train(epoch)
        if rank == 0:
            results['train_time'].append(train_time)
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
        
        # 验证
        val_time, val_loss, val_acc = validate(epoch)
        if rank == 0:
            results['val_time'].append(val_time)
            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)
        
        # 更新学习率
        lr_scheduler.step()
        
        # 保存模型(仅在主进程中)
        if rank == 0:
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.module.state_dict(), f"{args.save_path}/resnet50_ddp_best.pth")
                print(f"保存最佳模型, 精度: {best_acc:.2f}%")
            
            # 每个epoch保存一次
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_acc': best_acc,
            }, f"{args.save_path}/resnet50_ddp_epoch{epoch}.pth")
    
    if rank == 0:
        # 保存最终模型
        torch.save(model.module.state_dict(), f"{args.save_path}/resnet50_ddp_final.pth")
        
        # 打印最终结果
        print(f"\n训练完成!")
        print(f"最佳验证准确率: {best_acc:.2f}%")
        print(f"平均每个epoch训练时间: {sum(results['train_time'])/args.epochs:.2f}秒")
        print(f"平均每个epoch验证时间: {sum(results['val_time'])/args.epochs:.2f}秒")
    
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/path/to/imagenet', help='ImageNet数据集路径')
    parser.add_argument('--batch-size', type=int, default=128, help='每个GPU的批次大小')
    parser.add_argument('--epochs', type=int, default=90, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--log-interval', type=int, default=10, help='日志打印间隔')
    parser.add_argument('--save-path', default='./checkpoints', help='模型保存路径')
    args = parser.parse_args()
    
    # 使用torch.multiprocessing启动多个进程
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)