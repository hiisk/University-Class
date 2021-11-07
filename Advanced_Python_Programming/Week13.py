id = '20200000'
name = '홍길동'
print(id, name)

코랩 메뉴 -> 수정 -> 노트 설정 -> 하드웨어 가속기 GPU 권장(행렬 연산이 많기 때문)

구글 드라이브 연동


from google.colab import drive
drive.mount('/gdrive')

"""폴더 경로 설정 (과제 파일 업로드한 폴더로 경로 설정)"""

workspace_path = '/gdrive/My Drive/Colab Notebooks/dacon/컴퓨터비전_학습_경진대회'

"""필요 패키지 로드"""

!pip install albumentations  # data augmentation 패키지 설치

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensor
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

"""결과 재현을 위한 설정"""

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""합성곱 신경망(CNN) 정의"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,
                      stride=1, padding=1, bias=False),  # 1x28x28 -> 32x28x28
            nn.BatchNorm2d(32),  # 배치 정규화
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=1, padding=1, bias=False),  # 32x28x28 -> 64x28x28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 64x28x28 -> 64x14x14 (특징 압축)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                      stride=1, padding=1, bias=False),  # 64x14x14 -> 128x14x14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 128x14x14 -> 128x7x7 (특징 압축)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                      stride=1, padding=1, bias=False),  # 128x7x7 -> 256x7x7
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d(1)  # 256x7x7 -> 256x1x1 (채널 별 평균값 계산)
        )
        self.dropout = nn.Dropout(p=0.5)  # drop 시킬 노드 비율
        self.fc = nn.Linear(256, 10)  # 출력값의 차원은 판별할 클래스 수인 10으로 설정 (MNIST는 손글씨 숫자 10종 판별 문제)

    def forward(self, x):
        x = x.float()
        x = x.view(-1, 1, 28, 28)  # view 함수로 tensor 형태 변경: [batch크기, 1, 28, 28]
        x = self.main(x)  # CNN 모델 feed-forward
        x = x.view(-1, 256)  # view 함수로 형태 변경: [batch크기, 256]
        x = self.dropout(x)  # dropout 수행 (train mode에서만 동작)
        x = self.fc(x)  # 마지막 레이어에는 활성화 함수 사용하지 않음
        return x

print("init model done")

"""모델 학습을 위한 하이퍼파라미터 셋팅"""

batch_size = 64  # 학습 배치 크기
test_batch_size = 1000  # 테스트 배치 크기 (학습 과정을 제외하므로 더 큰 배치 사용 가능)
max_epochs = 50  # 학습 데이터셋 총 훈련 횟수
lr = 0.001  # 학습률
weight_decay = 1e-2  # L2 규제
log_interval = 20  # interval 때마다 로그 남김

use_cuda = torch.cuda.is_available()  # GPU cuda 사용 여부 확인

device = torch.device("cuda" if use_cuda else "cpu")  # GPU cuda 사용하거나 없다면 CPU 사용

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}  # num_workers: data loading할 프로세스 수, pin_memory: 고정된 메모리 영역 사용

print("set vars and device done")

"""데이터 전처리 정의 (데이터 증대 포함)"""

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.RandomCrop(height=8, width=8, p=0.5),
    A.Normalize(
        mean=[2],
        std=[1.414],
    ),
    ToTensor()
])

val_transform = A.Compose([
    A.Normalize(
        mean=[2],
        std=[1.414],
    ),
    ToTensor()
])

"""데이터 로더 정의 (학습용, 테스트용 따로 정의)"""

class train_dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.n_idx = len(df)
        self.data = self.df.loc[:,'0':].astype(float).values.reshape(-1,28,28)
        self.label = self.df['digit'].astype(int).values
    def __len__(self):
        return self.n_idx
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = torch.from_numpy(np.array(self.label[idx]))
        if self.transform:
            augmented = self.transform(image=data)
            image = augmented['image']
        return data, label

class val_dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.n_idx = len(df)
        self.data = self.df.loc[:,'0':].astype(float).values.reshape(-1,28,28)
        self.label = self.df['digit'].astype(int).values
    def __len__(self):
        return self.n_idx
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = torch.from_numpy(np.array(self.label[idx]))
        if self.transform:
            augmented = self.transform(image=data)
            image = augmented['image']
        return data, label
    
class test_dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.n_idx = len(df)
        self.data = self.df.loc[:,'0':].astype(float).values.reshape(-1,28,28)
    def __len__(self):
        return self.n_idx
    
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            augmented = self.transform(image=data)
            image = augmented['image']
        return data

# train 데이터셋 중의 임의의 1/5 를 validation 셋으로 설정
train_data = pd.read_csv(os.path.join(workspace_path, 'train.csv'))
random_index = random.sample(range(len(train_data.index)), len(train_data.index))
val_data = train_data.iloc[random_index[:len(train_data.index) // 5], :]
train_data = train_data.iloc[random_index[len(train_data.index) // 5:], :]
train_loader = DataLoader(train_dataset(train_data, train_transform),
                          batch_size=batch_size, shuffle=True,
                          **kwargs)
val_loader = DataLoader(val_dataset(val_data, val_transform),
                        batch_size=batch_size, shuffle=False,
                        **kwargs)
test_data = pd.read_csv(os.path.join(workspace_path, 'test.csv'))
test_loader = DataLoader(test_dataset(test_data, val_transform),
                         batch_size=test_batch_size, shuffle=False,
                         **kwargs)
train_len, val_len, test_len = len(train_data.index), len(val_data.index), len(test_data.index)
print(train_len, val_len, test_len)
train_data.head()

"""모델, 최적화 알고리즘, 손실 함수 정의"""

model = Net().to(device)  # 모델 정의
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # 최적화 알고리즘 정의 (Adam 사용)
criterion = nn.CrossEntropyLoss()  # 손실 함수 정의 (CrossEntropy 사용)

"""AverageMeter 정의"""

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

"""학습, 테스트용 함수 정의"""

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()  # 모델 학습 모드 설정
    summary_loss = AverageMeter()  # 학습 손실값 기록 초기화
    summary_acc = AverageMeter() # 학습 정확도 기록 초기화
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 현재 미니 배치의 데이터, 정답 불러옴
        optimizer.zero_grad()  # gradient 0으로 초기화
        output = model(data)  # 모델에 입력값 feed-forward
        loss = criterion(output, target)  # 예측값(클래스 별 score)과 정답간의 손실값 계산
        loss.backward()  # 손실값 역전파 (각 계층에서 gradient 계산, pytorch는 autograd로 gradient 자동 계산)
        optimizer.step()  # 모델의 파라미터 업데이트 (gradient 이용하여 파라미터 업데이트)
        summary_loss.update(loss.detach().item())  # 손실값 기록
        pred = output.argmax(dim=1, keepdim=True)  # 예측값 중에서 최고 score를 달성한 클래스 선발
        correct = pred.eq(target.view_as(pred)).sum().item()  # 정답과 예측 클래스가 일치한 개수
        summary_acc.update(correct / data.size(0))  # 정확도 기록
        if idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.4f}, Accuracy: {:.4f}'.format(
                epoch, idx * len(data), len(train_loader.dataset),
                100. * idx / len(train_loader), summary_loss.avg, summary_acc.avg))
            
    return summary_loss.avg, summary_acc.avg

def validate(log_interval, model, device, val_loader):
    model.eval()  # 모델 검증 모드 설정 (inference mode)
    summary_loss = AverageMeter()  # 검증 손실값 기록 초기화
    summary_acc = AverageMeter() # 검증 정확도 기록 초기화
    with torch.no_grad():  # 검증 모드이므로 gradient 계산안함
        for idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)  # 현재 미니 배치의 데이터, 정답 불러옴
            output = model(data)  # 모델에 입력값 feed-forward
            loss = criterion(output, target)  # 예측값(클래스 별 score)과 정답간의 손실값 계산
            summary_loss.update(loss.detach().item())  # 손실값 기록
            pred = output.argmax(dim=1, keepdim=True)  # 예측값 중에서 최고 score를 달성한 클래스 선발
            correct = pred.eq(target.view_as(pred)).sum().item()  # 정답과 예측 클래스가 일치한 개수
            summary_acc.update(correct / data.size(0))  # 정확도 기록

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format
          (summary_loss.avg, summary_acc.avg))  # 정답을 맞춘 개수 / 검증셋 샘플 수 -> Accuracy

    return summary_loss.avg, summary_acc.avg

def test(log_interval, model, device, test_loader):
    model.eval()  # 모델 검증 모드 설정 (inference mode)
    pred_test = np.zeros((test_len), dtype=np.int)  # 테스트 예측값 배열 생성
    with torch.no_grad():  # 테스트 모드이므로 gradient 계산안함
        for idx, (data) in enumerate(test_loader):
            data = data.to(device)  # 현재 미니 배치의 데이터 불러옴
            output = model(data)  # 모델에 입력값 feed-forward
            pred = output.argmax(dim=1, keepdim=True)  # 예측값 중에서 최고 score를 달성한 클래스 선발
            pred = pred.view(-1).detach().cpu().numpy()
            pred_test[test_batch_size * idx:test_batch_size * idx + min(test_batch_size, len(pred))] = pred

    return pred_test

"""학습, 검증, 모델 저장 수행"""

best_acc = 0
best_epoch = 0
best_model = ""
for epoch in range(1, max_epochs+1):
    train_loss, train_acc = train(log_interval, model, device, train_loader, optimizer, epoch)
    val_loss, val_acc = validate(log_interval, model, device, val_loader)

    # 검증에서 best accuracy 달성하면 모델 저장
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch
        best_model = model
        torch.save(model, os.path.join(workspace_path, f'dacon_cnn_model_best_acc.pt'))
        print(f'# save model: dacon_cnn_model_best_acc.pt\n')

print(f'\n\n# Best accuracy model({best_acc * 100:.2f}%, {best_epoch}-epoch): dacon_cnn_model_best_acc.pt\n')

"""테스트 진행"""

pred_test = test(log_interval, model, device, test_loader)
submission = pd.read_csv(os.path.join(workspace_path, 'submission.csv'))  # submission form 읽어오기
submission['digit'] = pred_test  # test prediction 값 반영
from datetime import datetime
submission.to_csv(os.path.join(workspace_path, f"submission_{datetime.now().strftime('%m%d_%H%M')}.csv"), index=False)
submission.head()