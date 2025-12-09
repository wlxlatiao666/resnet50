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
import threading

class CustomDistributedDataParallel(nn.Module):
    def __init__(self, module, device_ids=None, output_device=None, 
                 broadcast_buffers=True, bucket_cap_mb=25, 
                 find_unused_parameters=False, gradient_as_bucket_view=False,
                 num_process_groups=1):
        super().__init__()
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device if output_device is not None else device_ids[0]
        self.broadcast_buffers = broadcast_buffers
        self.bucket_cap_mb = bucket_cap_mb
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.num_process_groups = num_process_groups
        
        # 创建多个进程组用于Round-Robin调度
        self.process_groups = []
        if self.num_process_groups > 1:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            for i in range(self.num_process_groups):
                pg = dist.new_group(list(range(world_size)))
                self.process_groups.append(pg)
        else:
            self.process_groups = [dist.group.WORLD]
        
        # 注册参数
        self._register_parameters()
        
        # 创建参数到bucket的映射
        self._create_buckets()
        
        # 创建通信锁，用于同步参数广播
        self.comm_lock = threading.Lock()
        
        # 用于跟踪未使用参数的位图
        self.unused_parameter_bitmap = None
        if self.find_unused_parameters:
            self.unused_parameter_bitmap = torch.zeros(
                len(list(self.module.parameters())), dtype=torch.bool, 
                device=f'cuda:{self.device_ids[0]}'
            )
        
        # 用于no_sync上下文管理器
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
    
    def _register_parameters(self):
        """
        注册模型的所有参数并为其准备梯度同步所需的钩子
        """
        self.parameter_list = list(self.module.parameters())
        # map parameter id -> index to avoid tensor equality/hash issues
        self.parameter_list_indices = {id(p): i for i, p in enumerate(self.parameter_list)}
        self.all_parameters_set = set(self.parameter_list)
        
        # 为每个参数注册钩子函数，在反向传播时收集梯度
        self.grad_accs = []
        for param in self.parameter_list:
            if param.requires_grad:
                def grad_hook(param):
                    def hook(*_):
                        self._grad_ready_for_sync(param)
                    return hook
                acc = param.register_hook(grad_hook(param))
                self.grad_accs.append(acc)
    
    def _create_buckets(self):
        """
        根据bucket_cap_mb创建梯度bucket，以便批量进行梯度聚合
        """
        self.buckets = {}
        self.bucket_parameters = collections.defaultdict(list)
        # reverse mapping: parameter id -> bucket id (avoid tensor comparisons)
        self.param_to_bucket = {}
        
        # 以MB为单位的bucket容量
        bucket_cap_bytes = int(self.bucket_cap_mb * 1024 * 1024)
        bucket_id = 0
        current_bucket_bytes = 0
        
        # 以逆序遍历参数列表，因为反向传播时参数梯度的计算顺序与前向传播相反
        for param in reversed(self.parameter_list):
            if not param.requires_grad:
                continue
                
            param_bytes = param.numel() * param.element_size()
            
            # 如果当前bucket已满，创建新的bucket
            if current_bucket_bytes + param_bytes > bucket_cap_bytes and current_bucket_bytes > 0:
                bucket_id += 1
                current_bucket_bytes = 0
                
            # 将参数添加到当前bucket
            self.bucket_parameters[bucket_id].append(param)
            # record reverse mapping by id to avoid tensor equality
            self.param_to_bucket[id(param)] = bucket_id
            current_bucket_bytes += param_bytes
        
        # 为每个bucket创建一个事件，用于同步
        self.bucket_events = {
            bucket_id: torch.cuda.Event(enable_timing=False, blocking=False)
            for bucket_id in self.bucket_parameters.keys()
        }
        
        # 跟踪每个bucket中已准备好的梯度数量
        self.bucket_ready_parameters = {
            bucket_id: 0 for bucket_id in self.bucket_parameters.keys()
        }
        
        # 分配进程组给每个bucket(Round-Robin方式)
        self.bucket_process_groups = {}
        for i, bucket_id in enumerate(self.bucket_parameters.keys()):
            pg_idx = i % len(self.process_groups)
            self.bucket_process_groups[bucket_id] = self.process_groups[pg_idx]
    
    def _get_param_bucket(self, param):
        """获取参数所属的bucket ID"""
        return self.param_to_bucket.get(id(param), None)
    
    def _grad_ready_for_sync(self, param):
        """当参数的梯度计算完成时调用此函数"""
        if not self.require_backward_grad_sync:
            return
        
        # 如果启用了未使用参数检测，更新位图
        if self.find_unused_parameters:
            param_idx = self.parameter_list_indices[param]
            self.unused_parameter_bitmap[param_idx] = True
        
        # 找到参数所属的bucket
        bucket_id = self._get_param_bucket(param)
        if bucket_id is None:
            return
            
        # 增加已准备好参数的计数
        self.bucket_ready_parameters[bucket_id] += 1
        
        # 检查是否所有参数都准备好了
        if self.bucket_ready_parameters[bucket_id] == len(self.bucket_parameters[bucket_id]):
            self._launch_bucket_allreduce(bucket_id)
    
    def _launch_bucket_allreduce(self, bucket_id):
        """为指定bucket启动异步AllReduce操作"""
        # 获取bucket中所有参数的梯度
        bucket_params = self.bucket_parameters[bucket_id]
        
        # 创建bucket张量列表
        bucket_tensors = []
        for param in bucket_params:
            if param.grad is not None:
                bucket_tensors.append(param.grad)
            else:
                # 如果梯度为None，创建一个全零梯度
                param.grad = torch.zeros_like(param.data)
                bucket_tensors.append(param.grad)
        
        # 使用异步AllReduce
        with self.comm_lock:
            process_group = self.bucket_process_groups[bucket_id]
            
            # 启动异步AllReduce
            handles = []
            for tensor in bucket_tensors:
                handle = dist.all_reduce(tensor, group=process_group, async_op=True)
                handles.append(handle)
            
            # 等待所有通信操作完成
            for handle in handles:
                handle.wait()
            
            # 关键修复：除以world_size进行平均
            world_size = dist.get_world_size()
            for tensor in bucket_tensors:
                tensor.div_(world_size)
            
            # 标记bucket已完成
            self.bucket_events[bucket_id].record()
        
        # 重置bucket的ready参数计数
        self.bucket_ready_parameters[bucket_id] = 0

    def _sync_unused_parameters(self):
        """同步未使用参数的信息"""
        if not self.find_unused_parameters:
            return
            
        # 使用AllReduce操作收集所有进程中未使用参数的信息
        dist.all_reduce(self.unused_parameter_bitmap, op=dist.ReduceOp.MAX)
        
        # 对于全局未使用的参数，我们需要将其梯度设置为零
        for i, used in enumerate(self.unused_parameter_bitmap):
            if not used:
                param = self.parameter_list[i]
                if param.grad is not None:
                    param.grad.zero_()
        
        # 重置位图
        self.unused_parameter_bitmap.zero_()
    
    def forward(self, *inputs, **kwargs):
        """前向传递，同步参数并调用底层模块的forward"""
        if self.require_forward_param_sync:
            self._sync_parameters()
            
        return self.module(*inputs, **kwargs)
    
    def _sync_parameters(self):
        """在前向传递之前同步模型参数"""
        with torch.no_grad():
            for param in self.module.parameters():
                if param.requires_grad:
                    dist.broadcast(param.data, 0)
    
    def no_sync(self):
        """上下文管理器，用于暂时禁用梯度同步（实现梯度累积）"""
        class _ContextManager:
            def __init__(self, ddp):
                self.ddp = ddp
                self.old_value = ddp.require_backward_grad_sync
            
            def __enter__(self):
                self.ddp.require_backward_grad_sync = False
                
            def __exit__(self, *args):
                self.ddp.require_backward_grad_sync = self.old_value
        
        return _ContextManager(self)
    
    def synchronize(self):
        """确保所有bucket的AllReduce操作已完成"""
        for event in self.bucket_events.values():
            event.synchronize()

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
    # 使用优化技术：
    # 1. bucket_cap_mb - 梯度桶大小设置
    # 2. num_process_groups - Round-Robin Process Groups
    # 3. find_unused_parameters - 处理未使用参数
    # 4. gradient_as_bucket_view - 优化内存使用
    model = CustomDistributedDataParallel(
        model, 
        device_ids=[rank], 
        output_device=rank,
        bucket_cap_mb=25,  # 调整bucket大小以优化通信
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        num_process_groups=2  # 使用多个进程组进行Round-Robin调度
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
        writer = SummaryWriter(log_dir='./logs/custom_ddp')

    def train(epoch):
        model.train()
        train_sampler.set_epoch(epoch)  # 确保在每个epoch中数据被打乱
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        start_time = time.time()
        
        # 修复：增加梯度累积步数，从2改为4
        accumulation_steps = 4
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(rank), target.cuda(rank)
            
            # 计算当前在累积周期中的位置
            current_step = batch_idx % accumulation_steps
            
            # 修复：正确的梯度累积逻辑
            if current_step == 0:
                optimizer.zero_grad()
            
            # 前向传播
            if current_step < accumulation_steps - 1:
                # 非最后一步：使用no_sync跳过梯度同步
                with model.no_sync():
                    output = model(data)
                    loss = criterion(output, target)
                    # 修复：损失要除以累积步数，保持梯度尺度一致
                    loss = loss / accumulation_steps
                    loss.backward()
            else:
                # 最后一步：正常同步梯度
                output = model(data)
                loss = criterion(output, target)
                # 修复：损失要除以累积步数
                loss = loss / accumulation_steps
                loss.backward()
                # 修复：在累积结束时才更新参数
                optimizer.step()
            
            _, predicted = output.max(1)
            total_samples += target.size(0)
            running_correct += predicted.eq(target).sum().item()
            # 修复：使用缩放后的损失进行统计
            running_loss += loss.detach().item() * target.size(0) * accumulation_steps
            
            if rank == 0 and batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.detach().item() * accumulation_steps:.6f}')
                
                # 写入TensorBoard
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train_step', loss.detach().item() * accumulation_steps, step)
        
        train_time = time.time() - start_time
        train_loss = running_loss / total_samples
        train_acc = 100.0 * running_correct / total_samples
        
        if rank == 0:
            print(f'Train Epoch: {epoch}\tLoss: {train_loss:.4f}\tAcc: {train_acc:.2f}%\tTime: {train_time:.2f}s')
            
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Time/train', train_time, epoch)
        
        return train_time, train_loss, train_acc
    
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
        
        # 收集所有进程的结果
        val_loss_tensor = torch.tensor([val_loss], device=f'cuda:{rank}')
        correct_tensor = torch.tensor([correct], device=f'cuda:{rank}')
        total_tensor = torch.tensor([total_samples], device=f'cuda:{rank}')
        
        dist.reduce(val_loss_tensor, 0, op=dist.ReduceOp.SUM)
        dist.reduce(correct_tensor, 0, op=dist.ReduceOp.SUM)
        dist.reduce(total_tensor, 0, op=dist.ReduceOp.SUM)
        
        val_time = time.time() - start_time
        
        if rank == 0:
            val_loss = val_loss_tensor.item() / total_tensor.item()
            val_acc = 100.0 * correct_tensor.item() / total_tensor.item()
            
            print(f'Validation Epoch: {epoch}\tLoss: {val_loss:.4f}\tAcc: {val_acc:.2f}%\tTime: {val_time:.2f}s')
            
            if writer:
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
        else:
            val_loss, val_acc = 0, 0
            
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
                torch.save(model.module.state_dict(), f"{args.save_path}/resnet50_custom_ddp_best.pth")
                print(f"保存最佳模型, 精度: {best_acc:.2f}%")
            
            # 每个epoch保存一次
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_acc': best_acc,
            }, f"{args.save_path}/resnet50_custom_ddp_epoch{epoch}.pth")
    
    if rank == 0:
        # 保存最终模型
        torch.save(model.module.state_dict(), f"{args.save_path}/resnet50_custom_ddp_final.pth")
        
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