import random

import torch
import transformers


def set_universal_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False


def get_cuda_device() -> str:
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = _get_cuda_device_with_most_free_memory()
        else:
            device = "cuda"
    else:
        device = "cpu"

    print(torch.device(device))

    if "cuda" in device:
        print(torch.cuda.get_device_name())
        print(
            f"{int(torch.cuda.get_device_properties(0).total_memory / 1000000000)} GB VRAM"
        )

    return device


def _get_cuda_device_with_most_free_memory() -> str:
    device = ""
    device_memory = 0

    for i in range(torch.cuda.device_count()):
        current_device = f"cuda:{i}"
        current_device_memory = torch.cuda.mem_get_info(current_device)[0]

        if current_device_memory >= device_memory:
            device = current_device
            device_memory = current_device_memory

    return device
