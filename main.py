import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3))
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3))

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--dynamic-size', '-ds', action='store_true')
    
    args = parser.parse_args()
    
    device = args.device
    use_dynamic_size = args.dynamic_size
    
    
    model = TestNet().to(device).eval()

    with torch.no_grad():
        for i in range(100):
            print('processing...', i)
            if use_dynamic_size:
                model(torch.rand(1, 1, 2000 + 10 * i , 200))
            else:
                model(torch.rand(1, 1, 2000, 200))
