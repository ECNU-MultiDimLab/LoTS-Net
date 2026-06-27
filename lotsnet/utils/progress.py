import time

from tqdm import tqdm

from lotsnet.utils.dist import is_main_process


def wrap_dataloader(loader, *, progress="tqdm", desc=None, leave=False, total=None):
    if progress == "tqdm" and is_main_process():
        kwargs = {"desc": desc, "leave": leave}
        if total is not None:
            kwargs["total"] = total
        return tqdm(loader, **kwargs)
    return loader


def train_log(msg: str):
    if is_main_process():
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def maybe_log_step(*, progress, step, total, log_interval, msg):
    if progress == "log" and is_main_process() and (step + 1) % log_interval == 0:
        train_log(f"  step {step + 1}/{total}  {msg}")


def maybe_set_postfix(pbar, *, progress, **kwargs):
    if progress == "tqdm" and is_main_process() and hasattr(pbar, "set_postfix"):
        pbar.set_postfix(**kwargs)
