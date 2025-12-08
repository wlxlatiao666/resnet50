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
import collections
from contextlib import contextmanager

class CustomDistributedDataParallel(nn.Module):
    def __init__(self, module, device_ids, output_device=None, broadcast_buffers=True, 
                 bucket_cap_mb=25, find_unused_parameters=False, gradient_as_bucket_view=False,
                 num_process_groups=1):
        super(CustomDistributedDataParallel, self).__init__()
        self.module = module
        self.device_ids = device_ids if device_ids else list(range(torch.cuda.device_count()))
        self.output_device = output_device if output_device is not None else self.device_ids[0]
        self.broadcast_buffers = broadcast_buffers
        self.bucket_cap_mb = bucket_cap_mb
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Creating round-robin process groups
        self.process_groups = []
        if num_process_groups > 1:
            for i in range(num_process_groups):
                pg = dist.new_group(ranks=list(range(self.world_size)))
                self.process_groups.append(pg)
        else:
            self.process_groups.append(dist.group.WORLD)
        self.pg_count = len(self.process_groups)
        self.pg_idx = 0
        
        # Parameters that require gradients
        self.params = [p for p in self.module.parameters() if p.requires_grad]
        
        # Create buckets for parameters
        self._create_parameter_buckets()
        
        # Register hooks
        self._register_hooks()
        
        # Sync buffers initially
        self._sync_buffers()
        
        # For skipping gradient synchronization
        self.require_backward_grad_sync = True
        
        # For tracking unused parameters
        if self.find_unused_parameters:
            self.unused_parameters = torch.zeros(len(self.params), dtype=torch.bool, device=self.output_device)
            self.has_used_params = False
    
    def _create_parameter_buckets(self):
        # Create buckets in reverse order to better match backward pass execution
        self.buckets = []
        bucket_size_limit = self.bucket_cap_mb * 1024 * 1024  # Convert MB to bytes
        current_bucket = []
        current_bucket_size = 0
        
        # Sort params by size to optimize bucket creation
        params_with_index = [(i, p) for i, p in enumerate(self.params)]
        params_with_index.sort(key=lambda x: x[1].numel(), reverse=True)
        
        # Assign params to buckets
        for idx, param in params_with_index:
            param_size = param.numel() * param.element_size()
            if current_bucket_size + param_size > bucket_size_limit and current_bucket:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0
            
            current_bucket.append((idx, param))
            current_bucket_size += param_size
        
        # Add the last bucket if not empty
        if current_bucket:
            self.buckets.append(current_bucket)
        
        # Reverse the buckets to align with backward pass order
        self.buckets.reverse()
    
    def _register_hooks(self):
        # Per-parameter hooks
        self.grad_accs = []
        
        # Register backward hooks for overlapping computation and communication
        for bucket_idx, bucket in enumerate(self.buckets):
            for param_idx, param in bucket:
                def grad_hook(param, bucket_idx=bucket_idx):
                    return self._grad_hook(param, bucket_idx)
                
                acc = param.register_hook(grad_hook)
                self.grad_accs.append(acc)
    
    def _grad_hook(self, grad, bucket_idx):
        # Skip if we're not syncing gradients in this iteration
        if not self.require_backward_grad_sync:
            return grad
            
        # Mark parameter as used for unused parameter detection
        if self.find_unused_parameters and grad is not None:
            param_index = next((i for i, (idx, p) in enumerate(self.buckets[bucket_idx]) 
                               if p.grad is grad), None)
            if param_index is not None:
                self.unused_parameters[param_index] = False
                self.has_used_params = True
        
        # Check if all grads in the bucket are ready
        bucket = self.buckets[bucket_idx]
        ready = all(p.grad is not None for _, p in bucket)
        
        if ready:
            # Get the process group in round-robin fashion
            process_group = self.process_groups[self.pg_idx]
            self.pg_idx = (self.pg_idx + 1) % self.pg_count
            
            # Perform AllReduce
            grads = [p.grad for _, p in bucket]
            flat_grads = _flatten_dense_tensors(grads)
            
            # Actual communication - AllReduce
            dist.all_reduce(flat_grads, group=process_group)
            flat_grads.div_(self.world_size)
            
            # Copy the reduced gradients back to their original variables
            _unflatten_dense_tensors(flat_grads, grads)
            
        return grad
    
    def _sync_buffers(self):
        # Synchronize buffers across processes
        if self.broadcast_buffers and len(list(self.module.buffers())) > 0:
            for buf in self.module.buffers():
                # Broadcast from rank 0 to all others
                dist.broadcast(buf, 0)
    
    @contextmanager
    def no_sync(self):
        """
        Context manager to disable gradient synchronization
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync

    def forward(self, *inputs, **kwargs):
        # Sync buffers before forward pass
        if self.broadcast_buffers:
            self._sync_buffers()
        
        # If we're tracking unused parameters, reset the tracking tensor
        if self.find_unused_parameters:
            self.unused_parameters.fill_(True)
            self.has_used_params = False
        
        # Perform forward pass
        output = self.module(*inputs, **kwargs)
        
        # After forward pass, handle unused parameters
        if self.find_unused_parameters and self.has_used_params:
            # We need to sync unused parameters info across processes
            # In a real implementation, you'd use AllReduce to do a logical AND operation
            dist.all_reduce(self.unused_parameters.int(), op=dist.ReduceOp.BAND)
        
        return output

def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer"""
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat

def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors"""
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    
    # Copy into original tensors
    for i, tensor in enumerate(tensors):
        tensor.copy_(outputs[i])

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
    
    # 将模型包装到自定义DDP中
    model = CustomDistributedDataParallel(
        model, 
        device_ids=[rank],
        output_device=rank,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        num_process_groups=3  # 使用3个进程组进行轮询
    )
    
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
            
            # 对于累积梯度，我们可以使用no_sync()跳过梯度同步
            # 除最后一次外的梯度累积不需要同步，这里以4次累积为例
            grad_acc_steps = 4
            if (batch_idx % grad_acc_steps) != grad_acc_steps - 1:
                with model.no_sync():
                    output = model(data)
                    loss = criterion(output, target) / grad_acc_steps
                    loss.backward()
            else:
                output = model(data)
                loss = criterion(output, target) / grad_acc_steps
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            _, predicted = output.max(1)
            total_samples += target.size(0)
            running_correct += predicted.eq(target).sum().item()
            running_loss += loss.detach().item() * target.size(0) * grad_acc_steps
            
            if rank == 0 and batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {running_loss/total_samples:.4f}\t'
                      f'Acc: {100. * running_correct / total_samples:.2f}%')
        
        # 计算最终的训练时间、损失和准确率
        train_time = time.time() - start_time
        train_loss = running_loss / total_samples
        train_acc = 100. * running_correct / total_samples
        
        if rank == 0:
            print(f'Train Epoch: {epoch}\tLoss: {train_loss:.4f}\tAcc: {train_acc:.2f}%\tTime: {train_time:.2f}s')
            
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        return train_time, train_loss, train_acc

    def validate(epoch):
        model.eval()
        val_loss = 0
        correct = 0
        start_time = time.time()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(rank), target.cuda(rank)
                output = model(data)
                val_loss += criterion(output, target).item() * target.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        val_time = time.time() - start_time
        val_loss /= len(val_loader.dataset)
        val_acc = 100. * correct / len(val_loader.dataset)
        
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
    parser.add_argument('--data-path', required=True, help='ImageNet数据集路径')
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