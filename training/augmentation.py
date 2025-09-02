import albumentations as A
import numpy as np

def get_augmentations(input_dim = 512, image_size = 512):
    
    Sharpen = A.Sharpen(p=0.3, alpha= (0.6,0.8), lightness = (0.6,1.0))
    Normalize = A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    )
    
    VerticalFlip = A.VerticalFlip(p=0.3)
    RandomRotate90 = A.RandomRotate90(p=0.3)
    HorizontalFlip = A.HorizontalFlip(p=0.3)

    Geos = A.Compose([HorizontalFlip, VerticalFlip, RandomRotate90])

    RandomCrop = A.RandomCrop(height=input_dim, width=input_dim, p=0.3)
    CenterCrop = A.CenterCrop(height=input_dim, width=input_dim, p=0.3)

    Crops = A.Compose([RandomCrop, CenterCrop])
    Resize = A.Resize(height=input_dim, width=input_dim, p=1.0)

    Augmentations = A.Compose(Sharpen, 
                            Geos,
                            Crops,
                            Normalize,
                            Resize,
                            A.ToTensorV2())
    
    return Augmentations
