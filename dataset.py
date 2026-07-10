from dataset_data import FolderImages
from torch.utils.data import Dataset



class DinoDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_images = FolderImages(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.folder_images)

    def __getitem__(self, index):
        image = self.folder_images[index]
        if self.transform:
            image = self.transform(image)
        return image