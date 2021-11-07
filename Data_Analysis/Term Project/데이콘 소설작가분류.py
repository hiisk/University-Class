from google.colab import drive
import pandas as pd
import numpy as np
import re
import os
drive.mount('/gdrive', force_remount=True)
workspace_path = '/gdrive/My Drive/한밭대 20187100/4-1/데이터 분석/텀프로젝트/2. 소설작가분류/'
train = pd.read_csv(os.path.join(workspace_path, 'train.csv'), index_col=0)
test = pd.read_csv(os.path.join(workspace_path, 'test_x.csv'), index_col=0)
submission = pd.read_csv(os.path.join(workspace_path, 'submission.csv'), index_col=0)  
train

test

def alpha_num(text):
    return re.sub(r'[^A-Za-z0-9 ]', '', text)


def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stopwords:
            final_text.append(i.strip())
    return " ".join(final_text)


stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", 
             "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", 
             "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", 
             "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", 
             "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", 
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", 
             "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", 
             "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", 
             "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", 
             "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", 
             "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

train['text'] = train['text'].str.lower().apply(alpha_num).apply(remove_stopwords)
test['text'] = test['text'].str.lower().apply(alpha_num).apply(remove_stopwords)

train

test

import matplotlib
import matplotlib.pyplot as plt

train['length'] = train['text'].map(len)
train['length'].hist()

train.groupby('author')['length'].mean() 
#위에 코드로 1번 작가가 글을 길게쓰는것을 알수있다.

group_df = train.groupby('author')
plt_group = train['author'].value_counts()
plt.figure(figsize=(8,4))
sns.barplot(plt_group.index, plt_group.values, alpha=0.8)
plt.ylabel('Number', fontsize=13)
plt.xlabel('Author', fontsize=13)
plt.show()

text_train = train.text.values
author_train = train.author.values
test = test.text.values

!pip install autokeras
!pip install keras-tuner
#AutoKeras의 TextClassifier 사용
import tensorflow as tf
import autokeras as ak

keras = ak.TextClassifier(
    multi_label=True,
    overwrite=True,
    max_trials=1)

keras.fit(text_train, author_train, epochs=5)
model = keras.export_model()

pred = model.predict(test)
submission[['0','1','2','3','4']] = pred
submission.to_csv(os.path.join(workspace_path, f"submission.csv"), index=False)
a = submission
a

import seaborn as sns
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
fig, axs = plt.subplots(figsize=(40, 20), ncols=5, nrows=2)
a['author'] = train['author']
y_features = ['0', '1', '2', '3', '4']

for i, feature in enumerate(y_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x='0', y=feature, data=a, ax=axs[row][col])
      
for i, feature in enumerate(y_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x='author', y=feature, data=a, ax=axs[row][col])

