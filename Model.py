import torch
import torch.nn as nn
from torch.nn.functional import relu, sigmoid

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 64x64x2
        self.e11 = nn.Conv2d(2, 32, kernel_size=3, padding=1) # output: 64x64x32
        self.e12 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # output: 64x64x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x32

        # input: 32x32x32
        self.e21 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # output: 32x32x64
        self.e22 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 32x32x64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 16x16x64

        # input: 16x16x64
        self.e31 = nn.Conv2d(64,128, kernel_size=3, padding=1) # output: 16x16x128
        self.e32 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 16x16x128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 8x8x128

        # input: 8x8x128
        self.e41 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 8x8x256
        self.e42 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 8x8x256

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    
    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        
        # Decoder
        xu1 = self.upconv1(xe42)
        xu11 = torch.cat([xu1, xe32], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe22], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe12], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        out = sigmoid(self.outconv(xd32))

        return out

