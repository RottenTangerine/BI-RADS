"""
Based on DC-GAN
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channel, out_channel=1):
        super(Generator, self).__init__()

        ngf = 64

        self.conv = nn.Sequential(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(in_channel, ngf * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # out: 1024 x 4 x 4

            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # out: 512 x 8 x 8

            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # out: 512 x 16 x 16

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # out: 256 x 32 x 32

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # out: 128 x 64 x 64

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # out: 64 x 128 x 128

            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 256 x 256
        )

    def forward(self, x):
        x = x.view(*x.shape, 1, 1)
        return self.conv(x)
