import torch

print(f"当前GPU是否可用：{torch.cuda.is_available() or torch.backends.mps.is_available()}")
print(f"设备名:{torch.cuda.get_device_name() if torch.cuda.is_available() else torch.backends.mps.get_name() if torch.backends.mps.is_available() else 'CPU'}")
