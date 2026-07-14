import random

from dataset_data import FolderImages
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

import matplotlib.pyplot as plt

from tqdm import tqdm


from dino_v3_transforms import CustomDomainMultiCropTransform

class DinoDataset(Dataset):
    def __init__(self, folder_path, transform=None, high_res=False):
        self.folder_images = FolderImages(folder_path)
        self.high_res = high_res
        if transform is None:
            self.transform = CustomDomainMultiCropTransform(for_training=True, high_res=high_res,)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.folder_images)

    def __getitem__(self, index):
        image = self.folder_images[index]
        if self.transform:
            image = self.transform(image)
        return image

    def get_random_sample(self):
        index = random.randint(0, len(self.folder_images) - 1)
        return self.__getitem__(index)

    def get_random_samples_idx(self, sample_count=1):
        indexes = list(range(len(self.folder_images)))
        selected = random.sample(indexes, sample_count)
        return selected

    def estimate_dataset_stats(self, sample_count=1000):
        indexes = self.get_random_samples_idx(sample_count)

        pixel_sum = torch.zeros(3)
        pixel_sq_sum = torch.zeros(3)
        total_pixels = 0

        for idx in tqdm(indexes, desc="Estimating dataset stats"):
            image = self.folder_images[idx]
            image_tensor = T.ToTensor()(image)  # Shape: [3, H, W]

            # Count how many pixels are in this specific image
            num_pixels = image_tensor.shape[1] * image_tensor.shape[2]
            total_pixels += num_pixels

            # Sum the pixel values per channel
            pixel_sum += image_tensor.sum(dim=(1, 2))

            # Sum the squared pixel values per channel
            pixel_sq_sum += (image_tensor ** 2).sum(dim=(1, 2))

        # Calculate global mean: E[X]
        mean = pixel_sum / total_pixels

        # Calculate global variance: E[X^2] - (E[X])^2
        variance = (pixel_sq_sum / total_pixels) - (mean ** 2)

        # Standard deviation is the square root of variance
        std = torch.sqrt(variance)

        mean_list = mean.tolist()
        std_list = std.tolist()
        print(f"Estimated Mean: [{', '.join(f'{m:.4f}' for m in mean_list)}]")
        print(f"Estimated Std: [{', '.join(f'{s:.4f}' for s in std_list)}]")

        return mean.tolist(), std.tolist()



    def show(self,sample_count=1):

        samples_per_row = len(self.__getitem__(0))

        fig, axes = plt.subplots(sample_count, samples_per_row, figsize=(16, 8))
        fig.tight_layout()

        for sample_idx in range(sample_count):
            sample = self.get_random_sample()


            for idx in range(samples_per_row):
                axes[sample_idx][idx].imshow(CustomDomainMultiCropTransform.to_image(sample[idx]))
                axes[sample_idx][idx].axis('off')
                axes[0][idx].set_title(f"{sample[0][sample_idx].shape}", fontsize=10)

        axes[0][0].set_title(f"High Res" if self.high_res else "Low Res", fontsize=10)
        plt.show()



if __name__ == '__main__':
    dataset = DinoDataset(folder_path=r"E:\Alpha1", high_res=False)
    dataset.show(sample_count=3)
    sample = dataset[0]
    for idx, img in enumerate(sample):
        print(f"Sample {idx} shape: {img.shape}")

    dataset.estimate_dataset_stats(sample_count=5000)