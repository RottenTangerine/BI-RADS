"""
Based on DC-GAN
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
from model.basic_block import ResidualBlock
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ConvTranspose2d(input_nc, 128, 4, 1),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 128
        # out_features = in_features*2
        # for _ in range(2):
        #     model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
        #                 nn.InstanceNorm2d(out_features),
        #                 nn.ReLU(inplace=True) ]
        #     in_features = out_features
        #     out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(6):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    G = Generator(13, 3)
    tensor = torch.rand([6, 13, 1, 1])
    out = G(tensor)
    print(out.shape)
