import argparse
import os
import random
import shutil
import time
import warnings
import builtins

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from dataset import get_imagenet_loaders

# Fake dataset for testing without ImageNet
class FakeData(torch.utils.data.Dataset):
    def __init__(self, size=1000, transform=None):
        self.size = size
        self.transform = transform
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        # Return random image and label
        return torch.randn(3, 224, 224), random.randint(0, 999)

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='?', default='data/imagenet',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--dummy', action='store_true', help="Use fake data for testing")

    parser.add_argument('--dummy-size', default=1281167, type=int, help="Size of fake data")
    parser.add_argument('--device', default=None, type=str, help="Force device (cpu, cuda, mps)")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
        # Suppress printing on non-master processes
        if args.rank != 0:
            def print_pass(*args):
                pass
            builtins.print = print_pass

    # Determine which GPU to use
    # When using torchrun, LOCAL_RANK is set automatically
    if args.distributed and args.gpu is None:
        # Get local rank from environment (set by torchrun)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.gpu = local_rank
        print(f"Using GPU {args.gpu} (from LOCAL_RANK)")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device('cuda', args.gpu)
            torch.cuda.set_device(args.gpu)
        else:
            device = torch.device('cuda')
    elif torch.backends.mps.is_available():
         device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Create model
    print("=> creating model 'resnet50'")
    model = models.resnet50()

    if device.type == 'cpu' or device.type == 'mps':
        print(f'using {device.type}, this will be slow')
        model = model.to(device)
        if args.distributed:
             model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            # This branch should not be reached now, but keep as fallback
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if device.type == 'cuda':
        cudnn.benchmark = True

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = FakeData(args.dummy_size) # Size of ImageNet train
        val_dataset = FakeData(args.dummy_size // 10 if args.dummy_size > 100 else 10) # Size of ImageNet val
        
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
            
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
            
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        train_loader, val_loader, train_sampler = get_imagenet_loaders(
            args.data, args.batch_size, args.workers, args.distributed)

    if args.evaluate:
        validate(val_loader, model, criterion, args, device)
        return

    # Track overall training statistics
    training_start_time = time.time()
    epoch_stats = []  # Store stats for each epoch
    best_acc1 = 0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        epoch_start = time.time()
        
        # Train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, epoch, args, device)

        # Evaluate on validation set
        val_stats = validate(val_loader, model, criterion, args, device)
        acc1 = val_stats['acc1']
        
        epoch_time = time.time() - epoch_start
        
        # Track best accuracy
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_epoch = epoch
        
        # Store epoch statistics
        epoch_stats.append({
            'epoch': epoch,
            'epoch_time': epoch_time,
            'train_loss': train_stats['loss'],
            'train_acc1': train_stats['acc1'],
            'train_acc5': train_stats['acc5'],
            'train_throughput': train_stats['throughput'],
            'val_loss': val_stats['loss'],
            'val_acc1': val_stats['acc1'],
            'val_acc5': val_stats['acc5'],
            'val_throughput': val_stats['throughput'],
        })
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet50',
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
    
    # Print overall training summary
    total_training_time = time.time() - training_start_time
    print_training_summary(epoch_stats, total_training_time, best_acc1, best_epoch, args)


def train(train_loader, model, criterion, optimizer, epoch, args, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # Switch to train mode
    model.train()

    end = time.time()
    epoch_start = time.time()
    for i, (images, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Compute output
        output = model(images)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.detach().item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            # Calculate throughput
            throughput = images.size(0) / batch_time.val
            avg_throughput = images.size(0) / batch_time.avg
            progress.display(i, throughput, avg_throughput)
    
    # Epoch summary
    epoch_time = time.time() - epoch_start
    total_samples = len(train_loader.dataset)
    avg_throughput = total_samples / epoch_time
    print(f'\n==> Epoch {epoch} Training Summary:')
    print(f'    Total Time: {epoch_time:.2f}s')
    print(f'    Avg Throughput: {avg_throughput:.2f} images/sec')
    print(f'    Avg Batch Time: {batch_time.avg:.3f}s')
    print(f'    Avg Data Load Time: {data_time.avg:.3f}s ({100*data_time.avg/batch_time.avg:.1f}% of batch time)')
    print(f'    Train Loss: {losses.avg:.4f}')
    print(f'    Train Acc@1: {top1.avg:.2f}%')
    print(f'    Train Acc@5: {top5.avg:.2f}%\n')
    
    # Return statistics
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg,
        'throughput': avg_throughput,
        'batch_time': batch_time.avg,
        'data_time': data_time.avg,
    }


def validate(val_loader, model, criterion, args, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # Switch to evaluate mode
    model.eval()

    val_start = time.time()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Compute output
            output = model(images)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.detach().item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                throughput = images.size(0) / batch_time.val
                avg_throughput = images.size(0) / batch_time.avg
                progress.display(i, throughput, avg_throughput)

        val_time = time.time() - val_start
        total_samples = len(val_loader.dataset)
        avg_throughput = total_samples / val_time
        
        print(f'\n==> Validation Summary:')
        print(f'    Total Time: {val_time:.2f}s')
        print(f'    Avg Throughput: {avg_throughput:.2f} images/sec')
        print(f'    Val Loss: {losses.avg:.4f}')
        print(f'    Val Acc@1: {top1.avg:.2f}%')
        print(f'    Val Acc@5: {top5.avg:.2f}%\n')

    # Return statistics
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg,
        'throughput': avg_throughput,
        'batch_time': batch_time.avg,
    }



def print_training_summary(epoch_stats, total_time, best_acc1, best_epoch, args):
    """Print overall training summary after all epochs complete"""
    if not epoch_stats:
        return
    
    print('\n' + '='*100)
    print(' '*35 + 'OVERALL TRAINING SUMMARY')
    print('='*100)
    
    # Calculate overall statistics
    num_epochs = len(epoch_stats)
    avg_epoch_time = sum(s['epoch_time'] for s in epoch_stats) / num_epochs
    avg_train_throughput = sum(s['train_throughput'] for s in epoch_stats) / num_epochs
    avg_val_throughput = sum(s['val_throughput'] for s in epoch_stats) / num_epochs
    
    final_train_loss = epoch_stats[-1]['train_loss']
    final_train_acc1 = epoch_stats[-1]['train_acc1']
    final_val_loss = epoch_stats[-1]['val_loss']
    final_val_acc1 = epoch_stats[-1]['val_acc1']
    
    # Print overall statistics
    print(f'\nTraining Configuration:')
    print(f'  Total Epochs: {num_epochs}')
    print(f'  Batch Size: {args.batch_size}')
    print(f'  Workers: {args.workers}')
    print(f'  Initial LR: {args.lr}')
    
    print(f'\nTime Statistics:')
    print(f'  Total Training Time: {total_time:.2f}s ({total_time/3600:.2f} hours)')
    print(f'  Avg Time per Epoch: {avg_epoch_time:.2f}s ({avg_epoch_time/60:.2f} minutes)')
    print(f'  Fastest Epoch: {min(s["epoch_time"] for s in epoch_stats):.2f}s (Epoch {min(range(len(epoch_stats)), key=lambda i: epoch_stats[i]["epoch_time"])})')
    print(f'  Slowest Epoch: {max(s["epoch_time"] for s in epoch_stats):.2f}s (Epoch {max(range(len(epoch_stats)), key=lambda i: epoch_stats[i]["epoch_time"])})')
    
    print(f'\nThroughput Statistics:')
    print(f'  Avg Training Throughput: {avg_train_throughput:.2f} images/sec')
    print(f'  Avg Validation Throughput: {avg_val_throughput:.2f} images/sec')
    
    print(f'\nAccuracy Statistics:')
    print(f'  Best Val Acc@1: {best_acc1:.2f}% (Epoch {best_epoch})')
    print(f'  Final Train Acc@1: {final_train_acc1:.2f}%')
    print(f'  Final Val Acc@1: {final_val_acc1:.2f}%')
    print(f'  Train/Val Gap: {abs(final_train_acc1 - final_val_acc1):.2f}%')
    
    print(f'\nLoss Statistics:')
    print(f'  Final Train Loss: {final_train_loss:.4f}')
    print(f'  Final Val Loss: {final_val_loss:.4f}')
    print(f'  Best Train Loss: {min(s["train_loss"] for s in epoch_stats):.4f} (Epoch {min(range(len(epoch_stats)), key=lambda i: epoch_stats[i]["train_loss"])})')
    print(f'  Best Val Loss: {min(s["val_loss"] for s in epoch_stats):.4f} (Epoch {min(range(len(epoch_stats)), key=lambda i: epoch_stats[i]["val_loss"])})')
    
    # Print epoch-by-epoch table (show first 5, last 5, and best epoch if not in those ranges)
    print(f'\nEpoch-by-Epoch Progress:')
    print(f'{"Epoch":<8} {"Time(s)":<10} {"Train Loss":<12} {"Train Acc@1":<12} {"Val Loss":<12} {"Val Acc@1":<12} {"Throughput":<15}')
    print('-'*100)
    
    epochs_to_show = set()
    # First 5 epochs
    epochs_to_show.update(range(min(5, num_epochs)))
    # Last 5 epochs
    epochs_to_show.update(range(max(0, num_epochs - 5), num_epochs))
    # Best epoch
    epochs_to_show.add(best_epoch)
    
    last_shown = -1
    for i in sorted(epochs_to_show):
        if i - last_shown > 1:
            print('  ...')
        s = epoch_stats[i]
        marker = ' *' if i == best_epoch else ''
        print(f'{s["epoch"]:<8} {s["epoch_time"]:<10.2f} {s["train_loss"]:<12.4f} '
              f'{s["train_acc1"]:<12.2f} {s["val_loss"]:<12.4f} {s["val_acc1"]:<12.2f} '
              f'{s["train_throughput"]:<15.1f}{marker}')
        last_shown = i
    
    print('\n* = Best validation accuracy')
    print('='*100)
    print(f'Training completed! Best model saved with Val Acc@1: {best_acc1:.2f}% at Epoch {best_epoch}')
    print('='*100 + '\n')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, throughput=None, avg_throughput=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if throughput is not None and avg_throughput is not None:
            entries.append(f'Throughput {throughput:.1f} ({avg_throughput:.1f}) img/s')
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
