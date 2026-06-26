import torch
import torch.distributed as dist
import os


def init_distributed_mode(args):
    """初始化分布式环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)

    # 初始化进程组
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()  # 等待所有进程就绪


def cleanup():
    dist.destroy_process_group()


def is_main_process():
    """判断当前是否为主进程 (Rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_tensor(tensor, n):
    """
    将多个 GPU 上的 tensor (如 Loss) 求平均
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def master_print(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)
