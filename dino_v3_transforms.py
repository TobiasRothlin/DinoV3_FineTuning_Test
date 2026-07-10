import random
from PIL import ImageFilter
import torchvision.transforms as T


class CustomDomainMultiCropTransform:
    def __init__(
            self,
            global_size=(448, 256),  # Bumped up from 256 for finer detail
            local_size=(224, 128),  # Bumped up from 112
            global_crops_scale=(0.4, 1.0),
            local_crops_scale=(0.15, 0.4),  # Shifted up from 0.05 to avoid blank patches
            num_local_crops=6  # Reduced from 8 to save VRAM on the DGX
    ):
        self.num_local_crops = num_local_crops

        # Standard ImageNet stats (Calculate your domain's exact mean/std for better results)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Strong color augmentations are safe and encouraged for true RGB
        color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)

        # 1. Global Crop 1: Standard DINO augmentations
        self.global_transform_1 = T.Compose([
            T.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=1.0),
            T.ToTensor(),
            normalize,
        ])

        # 2. Global Crop 2: Less blur, adds Solarization[cite: 1]
        self.global_transform_2 = T.Compose([
            T.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.1),
            T.RandomSolarize(threshold=128, p=0.2),
            T.ToTensor(),
            normalize,
        ])

        # 3. Local Crops: Highly distorted, small patches
        self.local_transform = T.Compose([
            T.RandomResizedCrop(local_size, scale=local_crops_scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # Append the two asymmetrical global crops
        crops.append(self.global_transform_1(image))
        crops.append(self.global_transform_2(image))

        # Append the local crops
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))

        return crops