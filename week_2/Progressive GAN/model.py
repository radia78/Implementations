import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
from dataclasses import dataclass

'''
Implementation is from Aladdin Persson's ProGan implementation. 
I've made some minor adjustments to make it slightly easier for deployment
'''

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        """
        This is the scaled convolution used to equalize the learning rate 
        by multiplying the weights of the convolution by the He initializer.

        Args:
            in_channels: the number of input channels
            out_channels: the number of out channels
            kernel_size: the size of the kernel window
            stride: stride of the convolution
            padding: how much padding added to the input
            gain: the numerator in the squared term of He's initializer
        """
        super(WSConv2d, self).__init__()
        # initiate the conv layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # compute the He's constant
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        # remove bias
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize the weights with standard Gaussian
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
        This is the pixel normalization that squares the pixel value and sum across channels and meaned
        while adding a little bias epsilon before being square-rooted
        
        Args:
            epsilon: bias value
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(negative_slope)
        self.pn = PixelNorm()
        self.use_pn = use_pixelnorm

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        # Initial convolution block
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(config.z_dim, config.in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(config.negative_slope),
            WSConv2d(config.in_channels, config.in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(config.negative_slope),
            PixelNorm()
        )
        # to rgb layer
        self.initial_rgb = WSConv2d(config.in_channels, config.img_channels, kernel_size=1, stride=1, padding=0)
        
        # module list
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([self.initial_rgb])

        # add the progression and rgb blocks
        for i in range(len(factors) - 1):
            conv_in_c = int(config.in_channels * factors[i])
            conv_out_c = int(config.in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, config.img_channels, kernel_size=1, stride=1, padding=0))
    
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)
    
    def forward(self, x, alpha, steps):
        out = self.initial(x)

        if steps == 0:
            return self.initial_rgb(out)
        
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='nearest')
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)
    
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(config.in_channels * factors[i])
            conv_out = int(config.in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(config.img_channels, conv_in, kernel_size=1, stride=1, padding=0))

        self.initial_rgb = WSConv2d(config.img_channels, config.in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            WSConv2d(config.in_channels + 1, config.in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(config.in_channels, config.in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(config.in_channels, 1, kernel_size=1, padding=0, stride=1)
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)
    
# define the configuration for both discriminator and generator
@dataclass
class ProGANConfig:
    z_dim: int=100
    in_channels: int=256
    negative_slope: float=0.2
    img_channels: int=3
    
if __name__ == "__main__":
    config = ProGANConfig()

    gen = Generator(config)
    critic = Discriminator(config)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((1, config.z_dim, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}")
