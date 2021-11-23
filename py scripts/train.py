'''
author: Bibhabasu Mohapatra
kaggle id: https://www.kaggle.com/bibhabasumohapatra
team_name: Normal People
competition: PetFinder.my - Pawpularity Contest
'''


import os
import timm
from sklearn import metrics
import pandas as pd
import numpy as np
import torch
import albumentations
import create_model
import dataset
import engine
df = pd.read_csv('train_5fold.csv')
device = 'cuda'
epochs = 1
data_path = ''
train_aug = albumentations.Compose(                  ##  AUGMENTATIONs TAKEN FROM ABHISHEK THAKUR's tez Pawpular training
    [
        albumentations.Resize(224,224, p=1),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
#         albumentations.RandomBrightnessContrast(
#             brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
#         ),
#         albumentations.HorizontalFlip(p=0.4),         ##  THis part is from  Manav  check out his NB
#          albumentations.VerticalFlip(p=0.3),
#         albumentations.ShiftScaleRotate(
#                 shift_limit = 0.1, scale_limit=0.1, rotate_limit=45, p=0.5
#             ),
    ],
    p=1.0,
)

valid_aug = albumentations.Compose(
    [
        albumentations.Resize(224, 224, p=1),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

feats = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]
from itertools import chain
scores = []
for fold in range(5):
        model = create_model.get_model()
        model.to(device)
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        df_train = df_train.drop(columns = 'kfold')
        df_valid = df_valid.drop(columns = 'kfold')

        train_images = df_train.Id.values.tolist()
        train_images = [os.path.join(data_path,'train',i + '.jpg') for i in train_images]
        valid_images = df_valid.Id.values.tolist()
        valid_images = [os.path.join(data_path,'train',i + '.jpg') for i in valid_images]

        train_targets = df_train.Pawpularity.values/100
        valid_targets = df_valid.Pawpularity.values/100

        train_dataset = dataset.CustomDataset(image_path = train_images,features=df_train[feats].values,targets = train_targets,augmentations=train_aug)
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8,shuffle=True,pin_memory=True) 
        valid_dataset = dataset.CustomDataset(image_path = valid_images,features=df_valid[feats].values,targets =valid_targets,augmentations=valid_aug)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=8,shuffle=False,pin_memory=True) 

        optimizer = torch.optim.Adam(model.parameters(),lr=5e-5)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,verbose=True)
        print(f'============================== FOLD -- {fold} ==============================')
        for epoch in range(epochs):
            print(f'==================== Epoch -- {epoch} ====================')
            engine.train(model=model,train_loader=train_loader,device=device,optimizer=optimizer)
            
            final_outputs,final_targets = engine.eval(model=model,valid_loader=valid_loader,device=device,optimizer=optimizer)
    
            RMSE = np.sqrt(metrics.mean_squared_error(final_targets,final_outputs))
#             scheduler.step(RMSE)
            
            print(f'valid RMSE={RMSE}')
        torch.save(model.state_dict(),'model-epoch'+str(fold)+'.pth')
        scores.append(RMSE)