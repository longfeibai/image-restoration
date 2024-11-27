import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# Dataset class
class CelebADataset(Dataset):
    def __init__(self, img_dir, csv_path, mask_func, transform=None):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_path)
        self.mask_func = mask_func
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{idx + 1:06d}.jpg")
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        mask = self.mask_func(image.shape[-2:])
        corrupted_image = image * (1 - mask)
        return corrupted_image, image, mask


# Mask generation function
def random_mask(size, mask_ratio=0.25):
    h, w = size
    mask = torch.zeros((h, w))
    mask_h, mask_w = int(h * mask_ratio), int(w * mask_ratio)
    top, left = np.random.randint(0, h - mask_h), np.random.randint(0, w - mask_w)
    mask[top:top + mask_h, left:left + mask_w] = 1
    return mask.unsqueeze(0)  # Add channel dimension


# Attention block
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


# Generator with U-Net
class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder1 = self.conv_block(4, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)

        # Output layer
        self.final = nn.Conv2d(64, 3, kernel_size=1)

        # Attention block
        self.attention = ImprovedAttentionBlock(512)  # Ensure attention block matches bottleneck size

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

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        bottleneck = self.bottleneck(e3)
        bottleneck = self.attention(bottleneck)  # Attention applied here

        # Decoder
        d3 = self.decoder3(bottleneck)
        d3 = self._resize(d3, e3)  # Resize d3 to match e3
        d3 = d3 + e3

        d2 = self.decoder2(d3)
        d2 = self._resize(d2, e2)  # Resize d2 to match e2
        d2 = d2 + e2

        d1 = self.decoder1(d2)
        d1 = self._resize(d1, e1)  # Resize d1 to match e1
        d1 = d1 + e1

        # Final resize to match input size (128x128)
        return nn.functional.interpolate(self.final(d1), size=(128, 128), mode='bilinear', align_corners=False)

    def _resize(self, x, target):
        """Resize tensor x to match the spatial dimensions of the target tensor."""
        return nn.functional.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)


# Discriminator
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


# Loss functions
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the pretrained VGG16 model from torchvision
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, real, fake):
        real_features = self.vgg(real)
        fake_features = self.vgg(fake)
        return nn.functional.mse_loss(fake_features, real_features)


def generator_loss(pred_fake, real, fake, mask, perceptual_loss_fn=None):
    # Adjust the size of the generated and real images to match each other
    fake_resized = nn.functional.interpolate(fake, size=real.shape[2:], mode='bilinear', align_corners=False)
    real_resized = nn.functional.interpolate(real, size=real.shape[2:], mode='bilinear', align_corners=False)

    # Compute L1 loss
    l1_loss = nn.L1Loss()(fake_resized * mask, real_resized * mask)

    # Compute perceptual loss (if provided)
    perceptual_loss = 0
    if perceptual_loss_fn:
        perceptual_loss = perceptual_loss_fn(fake_resized, real_resized)

    # Compute adversarial loss
    adv_loss = nn.BCELoss()(pred_fake, torch.ones_like(pred_fake))

    # Return total loss
    return l1_loss + 0.001 * adv_loss + perceptual_loss


def discriminator_loss(pred_real, pred_fake):
    real_loss = nn.BCELoss()(pred_real, torch.ones_like(pred_real))
    fake_loss = nn.BCELoss()(pred_fake, torch.zeros_like(pred_fake))
    return (real_loss + fake_loss) / 2


# Main function
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = CelebADataset(
        img_dir='D:/download/archive/img_align_celeba/img_align_celeba',
        csv_path='D:/download/archive/list_eval_partition.csv',
        mask_func=random_mask,
        transform=transform
    )

    train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)
    perceptual_loss_fn = PerceptualLoss().to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    epochs = 150
    output_dir = 'output_images'  # Directory to save visualizations
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (corrupted, real, mask) in progress_bar:
            corrupted, real, mask = corrupted.to(device), real.to(device), mask.to(device)

            # Train generator
            fake = generator(corrupted, mask)
            pred_fake = discriminator(fake)
            g_loss = generator_loss(pred_fake, real, fake, mask, perceptual_loss_fn)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Train discriminator
            pred_real = discriminator(real)
            pred_fake = discriminator(fake.detach())
            d_loss = discriminator_loss(pred_real, pred_fake)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            progress_bar.set_postfix(G_Loss=g_loss.item(), D_Loss=d_loss.item())

        # Visualization
        if epoch == 0 or epoch % 5 == 0 or epoch == epochs - 1:
            generator.eval()
            with torch.no_grad():
                example_batch = next(iter(train_loader))
                corrupted_example, real_example, mask_example = (
                    example_batch[0].to(device),
                    example_batch[1].to(device),
                    example_batch[2].to(device)
                )
                fake_example = generator(corrupted_example, mask_example)
            generator.train()

            # Save images
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            for i in range(5):
                axes[0, i].imshow(real_example[i].permute(1, 2, 0).cpu().numpy())
                axes[0, i].set_title("Original")
                axes[1, i].imshow(corrupted_example[i].permute(1, 2, 0).cpu().numpy())
                axes[1, i].set_title("Corrupted")
                axes[2, i].imshow(fake_example[i].permute(1, 2, 0).cpu().numpy())
                axes[2, i].set_title("Generated")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"epoch_{epoch+1}.png"))
            plt.close()
