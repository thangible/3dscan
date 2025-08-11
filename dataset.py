from PIL import Image
from torch.utils.data import Dataset
import glob

class ImageDataset(Dataset):
    def __init__(self, img_dir_pattern, transform=None):
        self.image_paths = glob.glob(img_dir_pattern)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image as PIL Image (not numpy array!)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, img_path
