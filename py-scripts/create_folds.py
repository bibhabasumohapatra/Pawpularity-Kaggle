import pandas as pd
import numpy as np
from sklearn import model_selection
df = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
df["kfold"] = -1

df = df.sample(frac=1).reset_index(drop=True)

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

for fold, (train_idx, val_idx) in enumerate(kf.split(X=df,y=df.Pawpularity.values)):
    print(len(train_idx), len(val_idx))
    df.loc[val_idx, 'kfold'] = fold
