from PIL import Image
from torch.utils.data import Dataset
import os

class ImageDataset(Dataset):
    def __init__(self, data_dir, train_flag=False, transforms=None):
        data_dir = os.path.normpath(data_dir)

       
        
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}. Please check the directory path.")

        self.transforms = transforms


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image as PIL Image (not numpy array!)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transforms:
            image = self.transforms(image)

        return image, img_path
