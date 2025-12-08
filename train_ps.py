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
# from torch.utils.tensorboard import SummaryWriter

# Define Parameter Server Rank
PS_RANK = 0

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    # Use Gloo backend for point-to-point communication (send/recv) stability
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def flatten_params(model):
    """Flatten model parameters into a single vector."""
    return torch.nn.utils.parameters_to_vector(model.parameters())

def load_params(model, params_vec):
    """Load parameters from a vector into the model."""
    torch.nn.utils.vector_to_parameters(params_vec, model.parameters())

def flatten_grads(model):
    """Flatten model gradients into a single vector."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
        else:
            grads.append(torch.zeros_like(p.view(-1)))
    return torch.cat(grads)

def get_dummy_dataset(transform):
    """Create a dummy dataset if ImageNet is not found."""
    return datasets.FakeData(size=1280, image_size=(3, 224, 224), num_classes=1000, transform=transform)

def run_worker(rank, world_size, args):
    setup(rank, world_size)
    
    # Determine if this process is a Parameter Server or a Worker
    if rank == PS_RANK:
        run_parameter_server(rank, world_size, args)
    else:
        run_training_worker(rank, world_size, args)
    
    cleanup()

def run_parameter_server(rank, world_size, args):
    print(f"Parameter Server (Rank {rank}) starting...", flush=True)
    
    # Initialize model
    model = models.resnet50(weights=None)
    # PS can stay on CPU to save GPU memory for workers, or use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    
    # writer = SummaryWriter(log_dir='./logs/ps_server')
    
    # Calculate total number of parameters for buffer allocation
    param_vec = flatten_params(model)
    param_numel = param_vec.numel()
    print(f"Model parameters: {param_numel}")

    worker_ranks = [i for i in range(world_size) if i != PS_RANK]
    num_workers = len(worker_ranks)
    
    # Assume fixed number of steps per epoch for simplicity in this demo
    # In a real scenario, we might need a dynamic handshake
    # We'll estimate steps based on dataset size provided by a worker or config
    # For now, let's assume we run for a fixed number of steps or wait for signals
    # But to match train_ddp, we need to know the dataset size.
    # We will wait for a handshake from Worker 1 to know the number of batches.
    
    print("Waiting for batch count from Worker 1...")
    num_batches_tensor = torch.zeros(1, dtype=torch.long)
    dist.recv(num_batches_tensor, src=1)
    num_batches = num_batches_tensor.item()
    print(f"Total batches per epoch: {num_batches}")

    total_time = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        model.train()
        epoch_start = time.time()
        
        for step in range(num_batches):
            step_start = time.time()
            
            # 1. Broadcast current parameters to all workers
            # We use send/recv to simulate PS architecture strictly
            # Optimization: Send flattened parameters
            param_vec = flatten_params(model).cpu() # Send from CPU to avoid some issues with Gloo/GPU direct? 
            # Actually Gloo handles CPU tensors best.
            
            broadcast_start = time.time()
            for worker in worker_ranks:
                dist.send(param_vec, dst=worker)
            broadcast_time = time.time() - broadcast_start
            
            # 2. Receive gradients from all workers
            grad_acc = torch.zeros_like(param_vec)
            recv_buffer = torch.zeros_like(param_vec)
            
            recv_start = time.time()
            for worker in worker_ranks:
                dist.recv(recv_buffer, src=worker)
                grad_acc += recv_buffer
            recv_time = time.time() - recv_start
            
            # 3. Average gradients and update
            grad_acc /= num_workers
            
            # Manually set gradients into model
            # This is a bit tricky with optimizer.step(), so we can just do:
            # param -= lr * grad (simple SGD)
            # Or reconstruct grads into model.parameters() to use the optimizer
            
            # Load grads back into model.grad
            pointer = 0
            for param in model.parameters():
                num_param = param.numel()
                param.grad = grad_acc[pointer:pointer+num_param].view_as(param).to(device)
                pointer += num_param
            
            optimizer.step()
            optimizer.zero_grad()
            
            step_time = time.time() - step_start
            
            if step % args.log_interval == 0:
                print(f"Step [{step}/{num_batches}] Time: {step_time:.3f}s (Bcast: {broadcast_time:.3f}s, Recv: {recv_time:.3f}s)")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} finished in {epoch_time:.2f}s")
        
        # Validation Signal?
        # For simplicity, we skip synchronized validation in this PS demo 
        # or implement a simple one.
        # Let's just save the model.
        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"{args.save_path}/resnet50_ps_epoch{epoch}.pth")

def run_training_worker(rank, world_size, args):
    worker_id = rank - 1
    print(f"Worker {worker_id} (Rank {rank}) starting...", flush=True)
    
    # Setup device
    # Map rank to available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        device_id = worker_id % num_gpus
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # Data Setup
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if os.path.exists(os.path.join(args.data_path, 'train')):
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_path, 'train'),
            transform=train_transform
        )
    else:
        print(f"Warning: Dataset not found at {args.data_path}. Using FakeData.")
        train_dataset = get_dummy_dataset(train_transform)

    # DistributedSampler to partition data among workers
    # Note: world_size for sampler should be number of workers, not total ranks
    num_workers = world_size - 1
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=num_workers, rank=worker_id, shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    # Send batch count to PS (Only Rank 1 needs to do this)
    if rank == 1:
        num_batches = len(train_loader)
        dist.send(torch.tensor(num_batches, dtype=torch.long), dst=PS_RANK)
    
    # Model Setup
    model = models.resnet50(weights=None)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Pre-allocate buffer for parameters
    param_vec = flatten_params(model).cpu()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 1. Receive parameters from PS
            dist.recv(param_vec, src=PS_RANK)
            load_params(model, param_vec.to(device))
            
            # 2. Compute Gradients
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 3. Send Gradients to PS
            grad_vec = flatten_grads(model).cpu()
            dist.send(grad_vec, dst=PS_RANK)
            
            model.zero_grad()
            
            if batch_idx % args.log_interval == 0:
                print(f"Rank {rank} Epoch {epoch} Batch {batch_idx} Loss {loss.detach().item():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/path/to/imagenet', help='ImageNet dataset path')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size per worker')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save-path', default='./checkpoints', help='Save path')
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # Total world size = 1 PS + N Workers (N = GPU count)
    # If no GPU, use 1 PS + 2 Workers for demo
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        world_size = num_gpus + 1
    else:
        world_size = 3
        
    print(f"Starting training with 1 PS and {world_size - 1} Workers.")
    mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)
