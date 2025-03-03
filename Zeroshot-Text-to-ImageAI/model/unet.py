import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define your layers (example for downsampling, upsampling)
        self.down = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.up = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.down(x)
        x = self.up(x)
        return x

if __name__ == "__main__":
    model = UNet()
    print(model)
