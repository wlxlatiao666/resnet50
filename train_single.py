import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', default='/path/to/imagenet', help='ImageNet数据集路径')
parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
parser.add_argument('--epochs', type=int, default=90, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
parser.add_argument('--log-interval', type=int, default=10, help='日志打印间隔')
parser.add_argument('--save-path', default='./checkpoints', help='模型保存路径')
args = parser.parse_args()

# 确保保存目录存在
os.makedirs(args.save_path, exist_ok=True)

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

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=8, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size,
    shuffle=False, num_workers=8, pin_memory=True
)

# 定义模型
model = models.resnet50(weights=None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                     momentum=args.momentum, weight_decay=args.weight_decay)

# 学习率调度器
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

# TensorBoard日志
writer = SummaryWriter(log_dir='./logs/single_gpu')

def train(epoch):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        _, predicted = output.max(1)
        total_samples += target.size(0)
        running_correct += predicted.eq(target).sum().item()
        running_loss += loss.detach().item() * target.size(0)
        
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.detach().item():.6f}')
    
    epoch_loss = running_loss / total_samples
    epoch_acc = 100. * running_correct / total_samples
    epoch_time = time.time() - start_time
    
    print(f'Train Epoch: {epoch}\tLoss: {epoch_loss:.4f}\tAcc: {epoch_acc:.2f}%\tTime: {epoch_time:.2f}s')
    
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
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() * target.size(0)
            _, predicted = output.max(1)
            total_samples += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= total_samples
    val_acc = 100. * correct / total_samples
    val_time = time.time() - start_time
    
    print(f'Validation Epoch: {epoch}\tLoss: {val_loss:.4f}\tAcc: {val_acc:.2f}%\tTime: {val_time:.2f}s')
    
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    return val_time, val_loss, val_acc

if __name__ == '__main__':
    best_acc = 0
    print(f"训练使用设备: {device}")
    print(f"批次大小: {args.batch_size}")
    print(f"数据集大小: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params/1e6:.2f}M")
    
    # 记录训练信息
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
        
        # 训练
        train_time, train_loss, train_acc = train(epoch)
        results['train_time'].append(train_time)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        
        # 验证
        val_time, val_loss, val_acc = validate(epoch)
        results['val_time'].append(val_time)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        # 更新学习率
        lr_scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{args.save_path}/resnet50_single_best.pth")
            print(f"保存最佳模型, 精度: {best_acc:.2f}%")
        
        # 每个epoch保存一次
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_acc': best_acc,
        }, f"{args.save_path}/resnet50_single_epoch{epoch}.pth")
    
    # 保存最终模型
    torch.save(model.state_dict(), f"{args.save_path}/resnet50_single_final.pth")
    
    # 打印最终结果
    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"平均每个epoch训练时间: {sum(results['train_time'])/args.epochs:.2f}秒")
    print(f"平均每个epoch验证时间: {sum(results['val_time'])/args.epochs:.2f}秒")