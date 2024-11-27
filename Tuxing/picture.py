import torch
from torch import nn
from torchviz import make_dot

# 定义生成器模型（U-Net）
class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = self.conv_block(4, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        bottleneck = self.bottleneck(e3)
        d3 = self.decoder3(bottleneck)
        d3 = self._resize(d3, e3) + e3
        d2 = self.decoder2(d3)
        d2 = self._resize(d2, e2) + e2
        d1 = self.decoder1(d2)
        d1 = self._resize(d1, e1) + e1
        return self.final(d1)

    def _resize(self, x, target):
        return nn.functional.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)

# 创建生成器实例
model = GeneratorUNet()

# 生成一个假输入
x = torch.randn(1, 4, 128, 128)  # 假设输入是一个 128x128 图像，有 4 个通道

# 通过生成器模型进行前向传播
y = model(x)

# 可视化生成器模型，并保存为矢量图格式（PDF 或 SVG）
make_dot(y, params=dict(model.named_parameters())).render("generator_unet", format="svg")  # 或者 "svg"

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 创建判别器实例
discriminator = Discriminator()

# 生成一个假输入
x_disc = torch.randn(1, 3, 128, 128)  # 假设输入是一个 128x128 图像，3 个通道

# 通过判别器模型进行前向传播
y_disc = discriminator(x_disc)

# 可视化判别器模型，并保存为矢量图格式（PDF 或 SVG）
make_dot(y_disc, params=dict(discriminator.named_parameters())).render("discriminator", format="svg")  # 或者 "svg"

# 定义注意力模块
class ImprovedAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map

# 创建注意力模块实例
attention_block = ImprovedAttentionBlock(512)

# 生成一个假输入
x_attn = torch.randn(1, 512, 16, 16)  # 假设输入为 16x16 的特征图，有 512 个通道

# 通过注意力模块进行前向传播
y_attn = attention_block(x_attn)

# 可视化注意力模块，并保存为矢量图格式（PDF 或 SVG）
make_dot(y_attn, params=dict(attention_block.named_parameters())).render("attention_block", format="svg")  # 或者 "svg"
