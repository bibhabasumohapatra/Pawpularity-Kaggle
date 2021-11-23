'''
author: Bibhabasu Mohapatra
kaggle id: https://www.kaggle.com/bibhabasumohapatra
team_name: Normal People
competition: PetFinder.my - Pawpularity Contest
'''

import torch
import torch.nn as nn
from skimage import io, transform
import numpy as np
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,image_path,features,targets,augmentations=None):
        self.image_path = image_path
        self.features = features
        self.targets = targets
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self,item):
        image = io.imread(self.image_path[item])
        features = self.features[item,:]
        targets = self.targets[item]
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }
