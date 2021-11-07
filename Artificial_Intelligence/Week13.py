id = '20187100'
name = '강현석'
print(id, name)

"""# 1. 비지도 학습

## 차원 축소
"""


# Commented out IPython magic to ensure Python compatibility.
import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# mnist 데이터셋 불러오기
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape, y_train.shape)


# mnist 데이터셋을 pandas DataFrame 으로 변환
feat_cols = [ 'pixel'+str(i) for i in range(x_train.shape[1]) ]
df = pd.DataFrame(x_train, columns=feat_cols)
df['y'] = y_train
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))


# 실험결과 재현을 위한 설정
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])
df.head()


# 랜덤하게 선별한 30장 이미지 plot
plt.gray()
fig = plt.figure( figsize=(16,7) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
plt.show()


"""### PCA 예제

784차원의 MNIST 데이터를 PCA 이용하여 3차원으로 축소

PCA 계산 복잡도: O(N * D * min(N, D) + D^3)

N: 데이터 개수, D: 차원 수
"""

N = 5000
df_subset = df.loc[rndperm[:N],:].copy()  # 데이터셋 일부만 사용
pca = PCA(n_components=3)  # PCA 성분 수 3개로 설정
pca_result = pca.fit_transform(df_subset[feat_cols].values)  # PCA 수행
df_subset['pca-one'] = pca_result[:,0]  # 첫번째 주성분
df_subset['pca-two'] = pca_result[:,1]  # 두번째 주성분
df_subset['pca-three'] = pca_result[:,2]  # 세번째 주성분
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# 2D scatter 그래프
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset.loc[rndperm[:N],:],
    legend="full",
    alpha=0.3
)


# 3D scatter 그래프
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df_subset.loc[rndperm[:N],:]["pca-one"], 
    ys=df_subset.loc[rndperm[:N],:]["pca-two"], 
    zs=df_subset.loc[rndperm[:N],:]["pca-three"], 
    c=df_subset.loc[rndperm[:N],:]["y"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()


"""### PCA를 이용하여 차원 수 적절히 줄이기

PCA로 데이터셋의 분산 95% 이상 보존하는 범위 내에서 차원 축소
"""


# 현재 데이터셋의 분산 95% 이상 보존하도록 PCA 수행 (데이터셋의 특성을 상당수 보존)
pca_reduced = PCA(n_components=0.95)  # 분산 95% 이상 보존하도록 PCA 수행
pca_result_reduced = pca_reduced.fit_transform(df_subset[feat_cols].values)
print('Cumulative explained variation for principal components: {}'.format(np.sum(pca_reduced.explained_variance_ratio_)))


"""### MNIST 압축"""

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

from sklearn.model_selection import train_test_split

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)  # mnist 데이터셋 불러오기

def plot_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    print("그림 출력", fig_id)
    if tight_layout:
        plt.tight_layout()

pca = PCA(n_components=0.95)  # 분산 95% 이상 보존하도록 PCA 수행
X_reduced = pca.fit_transform(X_train)
pca.n_components_  # 분산 95% 이상 보존하는 차원 수
np.sum(pca.explained_variance_ratio_)  # 현재 pca에서 포함하는 분산
pca = PCA(n_components = 0.95)
X_reduced = pca.fit_transform(X_train)  # PCA 수행하여 차원 축소된 데이터셋
X_recovered = pca.inverse_transform(X_reduced)  # 차원 축소된 데이터셋으로 복원한 데이터셋


# mnist 데이터 시각화를 위한 함수 정의
def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.title("Original", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::2100])
plt.title("Compressed", fontsize=16)

plot_fig("mnist_compression_plot")


"""# 오토인코더"""

import os

import random
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from google.colab import drive
drive.mount('/gdrive')

workspace_path = '/gdrive/My Drive/Colab Notebooks/AI_과제13'  # 과제 파일을 업로드한 경로 반영할 것


# 784차원 vector를 원본 데이터 분포로 복원하고 28x28 이미지로 변형
# [-1, 1] -> [0, 1] 분포로 복원
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


# 결과 재현을 위한 설정
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# 주요 하이퍼파라미터 설정
num_epochs = 20
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [0, 1] -> [-1, 1]로 입력값 분포 변환
])

dataset = MNIST(os.path.join(workspace_path, 'mnist_data'), transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 28 * 28, out_channels = 128, kernel_size=3, stride=2,
                      padding=1),  
            nn.BatchNorm2d(128),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size=3, stride=2,
                      padding=1),  
            nn.BatchNorm2d(64),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size=3, stride=2,
                      padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size=3, stride=2,
                      padding=1),  
            nn.BatchNorm2d(16),  
            nn.AdaptiveAvgPool2d(1)  
        )
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(16, 10)
       
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32 , kernel_size=3, stride=2,
                      padding=1), 
            nn.BatchNorm2d(32),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(64),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(128),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 784, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(784), 
            nn.AdaptiveAvgPool2d(1))  


    def forward(self, x):
        x = x.float()
        x.view(-1, 1, 28, 28)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder()  # 오토인코더 모델 생성
criterion = nn.MSELoss()  # Mean Square Error를 손실함수로 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Adam 최적화 알고리즘 사용

if not os.path.exists(os.path.join(workspace_path, 'mnist_mlp_img')):
    os.mkdir(os.path.join(workspace_path, 'mnist_mlp_img'))  # 이미지파일 저장 폴더 생성

for epoch in range(num_epochs+1):
    for i, data in enumerate(dataloader):
        img, _ = data
        img = img.view(img.size(0), -1, 1, 1)
        # ===================forward=====================
        output = model(img)  # 입력값을 모델에 넣어 출력값 획득
        loss = criterion(output, img)  # 손실값 계산
        # ===================backward====================
        optimizer.zero_grad()  # gradient 초기화
        loss.backward()  # 손실값 이용하여 gradient 계산
        optimizer.step()  # gradient 이용하여 파라미터 업데이트
        # ===================log========================
        if i % 50 == 0:
            print('epoch [{}/{}], iter [{}/{}], loss:{:.4f}'
                .format(epoch, num_epochs, i, len(dataloader), loss.item()))
    if epoch % 5 == 0:
        pic = to_img(output.data)  # [-1, 1] -> [0, 1] 분포로 변환 (이미지 파일 출력을 위함)
        save_image(pic, os.path.join(workspace_path, 'mnist_mlp_img/image_{}.png'.format(epoch))) 
torch.save(model.state_dict(), os.path.join(workspace_path, 'mnist_mlp_img/simple_autoencoder.pt'))