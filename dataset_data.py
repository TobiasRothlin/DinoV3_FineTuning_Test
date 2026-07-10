import os
from PIL import Image

class FolderImages:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images_paths = self._locate_images()

    def _locate_images(self):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
        print(f"Found {len(image_files)} image files in '{self.folder_path}'")
        return image_files


    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.images_paths):
            raise IndexError("Index out of range")
        image_path = self.images_paths[index]
        image = Image.open(image_path).convert('RGB')
        return image
