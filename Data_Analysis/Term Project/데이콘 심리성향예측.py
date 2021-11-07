import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from sklearn.metrics import confusion_matrix, accuracy_score
from google.colab import drive
import os
import random
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

drive.mount('/gdrive', force_remount=True)

workspace_path = '/gdrive/My Drive/한밭대 20187100/4-1/데이터 분석/텀프로젝트/1. 심리성향예측/'
train = pd.read_csv(os.path.join(workspace_path, 'train.csv'), index_col=0)
test = pd.read_csv(os.path.join(workspace_path, 'test_x.csv'), index_col=0)
submission = pd.read_csv(os.path.join(workspace_path, 'submission.csv'), index_col=0)  

train.head()

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_data = train.drop(train[train.familysize > 30].index)
test_data = test
wr_s = ['wr_01', 'wr_02', 'wr_03', 'wr_04', 'wr_05', 'wr_06', 'wr_07', 'wr_08', 'wr_09', 'wr_10', 'wr_11','wr_12', 'wr_13']
train['wr_sum'] = train[wr_s].sum(axis=1)
test['wr_sum'] = test[wr_s].sum(axis=1)
wf_s = ['wf_01', 'wf_02', 'wf_03']
train['wf_sum'] = train[wf_s].sum(axis=1) 
test['wf_sum'] = test[wr_s].sum(axis=1)

drop_list = ['QaE', 'QbE', 'QcE', 'QdE', 'QeE',
             'QfE', 'QgE', 'QhE', 'QiE', 'QjE',
             'QkE', 'QlE', 'QmE', 'QnE', 'QoE',
             'QpE', 'QqE', 'QrE', 'QsE', 'QtE','hand','wr_01', 'wr_02', 'wr_03', 'wr_04', 'wr_05', 'wr_06', 'wr_07', 'wr_08', 'wr_09', 'wr_10', 'wr_11','wr_12', 'wr_13','wf_01', 'wf_02', 'wf_03']

replace_dict = {'education': str, 'engnat': str, 'married': str, 'urban': str}
train_y = train_data['voted']
train_x = train_data.drop(drop_list + ['voted'], axis=1)
test_x = test_data.drop(drop_list, axis=1)
train_x = train_x.astype(replace_dict)
test_x = test_x.astype(replace_dict)
train_x = pd.get_dummies(train_x)
test_x = pd.get_dummies(test_x)

display(train_x.head())
display(train_x.describe())

import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots(figsize=(30, 20), ncols=3, nrows=5)

y_features = ['QaA', 'gender_Male', 'wr_sum', 'wf_sum', 'education_4', 'age_group_50s', 'married_3', 'race_Black', 'urban_3', 'race_Indigenous Australian', 'race_Other', 'religion_Muslim', 'urban_2', 'tp10', 'familysize']


for i, feature in enumerate(y_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x='familysize', y=feature, data=train_x, ax=axs[row][col])

display(test_x.head())
display(test_x.describe())

# Commented out IPython magic to ensure Python compatibility.
#Q()A 데이터 시각화

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
# %matplotlib inline
fig,ax = plt.subplots(5,4, figsize=(10,15))

for i in range(20):
    x_num = i//4
    y_num = i%4

    plot_dict = train_x.iloc[:,i].value_counts().to_dict()
    x_values = sorted(plot_dict.keys())
    y_values = [plot_dict[x] for x in x_values]
    
    ax[x_num, y_num].bar(x_values, y_values)
    ax[x_num, y_num].set_title(train_x.columns[i])
    ax[x_num, y_num].set_ylim(0,40000)
plt.tight_layout()
plt.show()

#Familysize 시각화

train.familysize = train.familysize.apply(lambda x : 11 if x >= 11 else x)
plot_dict = train.loc[:,'familysize'].value_counts().to_dict()
x_values = sorted(plot_dict.keys())
y_values = [plot_dict[x] for x in x_values]
plt.bar(x_values,y_values)
plt.tight_layout()
plt.show()
print(y_values)

#hand 시각화

plot_dict = train.loc[:,'hand'].value_counts().to_dict()
x_values = sorted(plot_dict.keys())
y_values = [plot_dict[x] for x in x_values]
x_names = ['Nan','Right','Left','Both hands']
plt.bar(x_names,y_values)
plt.tight_layout()
plt.show()
print(y_values)

train_x.head()
model = lgbm.LGBMClassifier(n_estimators=500)
model.fit(train_x, train_y)

pred_y = model.predict(test_x)
pred_y
submission['voted']=pred_y
submission.to_csv(os.path.join(workspace_path, f"submission_{datetime.now().strftime('%m%d_%H%M')}.csv"), index=False)
submission.head(20)

