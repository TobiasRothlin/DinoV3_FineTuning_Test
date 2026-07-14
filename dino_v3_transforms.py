import torchvision.transforms as T
import torch

from Config import Config


class CustomDomainMultiCropTransform:
    def __init__(
            self,
            global_size=Config.global_size,
            local_size=Config.local_size,
            global_crops_scale=Config.global_crops_scale,
            local_crops_scale=Config.local_crops_scale,
            num_local_crops=Config.num_local_crops,
            for_training=False,
            high_res=False,
    ):
        self.num_local_crops = num_local_crops

        global_ratio_target = global_size[1] / global_size[0]  # 160 / 448 = ~0.357
        global_ratio = (global_ratio_target * Config.ratio_margin_low,
                        global_ratio_target * Config.ratio_margin_high)

        local_ratio_target = local_size[1] / local_size[0]  # 128 / 224 = ~0.571
        local_ratio = (local_ratio_target * Config.ratio_margin_low,
                       local_ratio_target * Config.ratio_margin_high)

        if high_res:
            global_size = (global_size[0] * 2, global_size[1] * 2)

        color_jitter = T.ColorJitter(
            brightness=Config.jitter_brightness,
            contrast=Config.jitter_contrast,
            saturation=Config.jitter_saturation,
            hue=Config.jitter_hue
        )

        global_aug_1 = [
            T.RandomResizedCrop(global_size, scale=global_crops_scale, ratio=global_ratio,
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=Config.horizontal_flip_p),
            T.RandomApply([color_jitter], p=Config.color_jitter_p),
            T.RandomGrayscale(p=Config.grayscale_p),
            T.RandomApply([T.GaussianBlur(kernel_size=Config.blur_kernel_size, sigma=Config.blur_sigma)],
                          p=Config.global_blur_1_p),
        ]

        global_aug_2 = [
            T.RandomResizedCrop(global_size, scale=global_crops_scale, ratio=global_ratio,
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=Config.horizontal_flip_p),
            T.RandomApply([color_jitter], p=Config.color_jitter_p),
            T.RandomGrayscale(p=Config.grayscale_p),
            T.RandomApply([T.GaussianBlur(kernel_size=Config.blur_kernel_size, sigma=Config.blur_sigma)],
                          p=Config.global_blur_2_p),
            T.RandomSolarize(threshold=Config.solarize_threshold, p=Config.solarize_p),
        ]

        local_aug = [
            T.RandomResizedCrop(local_size, scale=local_crops_scale, ratio=local_ratio,
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=Config.horizontal_flip_p),
            T.RandomApply([color_jitter], p=Config.color_jitter_p),
            T.RandomGrayscale(p=Config.grayscale_p),
            T.RandomApply([T.GaussianBlur(kernel_size=Config.blur_kernel_size, sigma=Config.blur_sigma)],
                          p=Config.local_blur_p),
        ]

        # 2. TENSOR CONVERSION FOR TRAINING:
        if for_training:
            normalize = T.Normalize(mean=Config.mean, std=Config.std)
            tensor_transforms = [T.ToTensor(), normalize]

            global_aug_1.extend(tensor_transforms)
            global_aug_2.extend(tensor_transforms)
            local_aug.extend(tensor_transforms)

        self.global_transform_1 = T.Compose(global_aug_1)
        self.global_transform_2 = T.Compose(global_aug_2)
        self.local_transform = T.Compose(local_aug)

    def __call__(self, image):
        crops = []
        # Append the two asymmetrical global crops
        crops.append(self.global_transform_1(image))
        crops.append(self.global_transform_2(image))

        # Append the local crops
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))

        return crops

    @staticmethod
    def to_image(tensor):
        """Convert a tensor back to a PIL image."""
        unnormalize = T.Normalize(
            mean=[-m / s for m, s in zip(Config.mean, Config.std)],
            std=[1 / s for s in Config.std]
        )
        tensor = unnormalize(tensor)
        tensor = torch.clamp(tensor, 0, 1)  # Ensure values are in [0, 1]
        return T.ToPILImage()(tensor)