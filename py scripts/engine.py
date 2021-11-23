
'''
author: Bibhabasu Mohapatra
kaggle id: https://www.kaggle.com/bibhabasumohapatra
team_name: Normal People
competition: PetFinder.my - Pawpularity Contest
'''

import torch
import torch.nn as nn
import numpy as np

def train(model,train_loader,device,optimizer):
    model.train()
    running_train_loss = 0.0
    for data in train_loader:
        inputs = data['image']
        features = data['features']
        targets = data['targets']

        inputs = inputs.to(device, dtype=torch.float)
        features = features.to(device,dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(inputs,features)
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_train_loss +=loss.item()
        
    train_loss_value = running_train_loss/len(train_loader)
    print(f'train BCE loss is {train_loss_value}')
    
def eval(model,valid_loader,device,optimizer):
    model.eval()
    final_targets = []
    final_outputs = []
    running_val_loss = 0.0
    with torch.no_grad():
        for data in valid_loader:
            inputs = data['image']
            features = data['features']
            targets = data['targets']
            inputs = inputs.to(device, dtype=torch.float)
            features = features.to(device,dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)

            output = model(inputs,features)
            running_val_loss += nn.BCEWithLogitsLoss()(output, targets.view(-1, 1))
            targets = (targets.detach().cpu().numpy()*100).tolist()
            output = (torch.sigmoid(output).detach().cpu().numpy()*100).tolist()
            final_outputs.extend(output)
            final_targets.extend(targets)
        val_loss = running_val_loss/len(valid_loader)    
        print(f'valid BCE loss is {val_loss}')
    return final_outputs,final_targets