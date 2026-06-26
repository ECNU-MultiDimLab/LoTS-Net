import time

from tqdm import tqdm

from lotsnet.utils.dist import is_main_process


def wrap_dataloader(loader, *, progress="tqdm", desc=None, leave=False, total=None):
    """
    按 progress 模式包装 DataLoader 迭代器。
    tqdm 模式仅主进程显示进度条；log / none 模式各 rank 使用原始 loader。
    """
    if progress == "tqdm" and is_main_process():
        kwargs = {"desc": desc, "leave": leave}
        if total is not None:
            kwargs["total"] = total
        return tqdm(loader, **kwargs)
    return loader


def train_log(msg: str):
    """仅主进程打印，附带时间戳。"""
    if is_main_process():
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def maybe_log_step(*, progress, step, total, log_interval, msg):
    """log 模式下每隔 log_interval 步打印一行。"""
    if progress == "log" and is_main_process() and (step + 1) % log_interval == 0:
        train_log(f"  step {step + 1}/{total}  {msg}")


def maybe_set_postfix(pbar, *, progress, **kwargs):
    """tqdm 模式下更新 postfix；其他模式无操作。"""
    if progress == "tqdm" and is_main_process() and hasattr(pbar, "set_postfix"):
        pbar.set_postfix(**kwargs)
