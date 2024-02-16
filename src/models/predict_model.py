import torch

from src.models.sincnet import SincNet

if __name__ == "__main__":
    # Test SincNet
    model = SincNet()
    waveforms = torch.randn(1, 1, 16000)
    outputs = model(waveforms)
    print(outputs.shape) # torch.Size([1, 60, 166])
