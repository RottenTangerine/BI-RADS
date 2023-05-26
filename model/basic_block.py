import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, bn=True, activate=True):
        super(BasicConv, self).__init__()
        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_channel, output_channel, kernel_size, stride)
        ]
        if bn:
            layers.append(nn.InstanceNorm2d(output_channel))
        if activate:
            layers.append(nn.LeakyReLU(0.2, True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class BasicUpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1, bn=True, activation=True):
        super(BasicUpsampleBlock, self).__init__()

        layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                                     bias=False)]

        if bn:
            layers.append(nn.BatchNorm2d(out_channel))

        if activation:
            layers.append(nn.LeakyReLU(0.2, True))

        self.upsample = nn.Sequential(*layers)

    def forward(self, x):
        x = self.upsample(x)
        return x