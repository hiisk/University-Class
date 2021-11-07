id = '20187100'
name = '강현석'
print(id, name)

"""코랩 메뉴 -> 수정 -> 노트 설정 -> 하드웨어 가속기 GPU 권장(행렬 연산이 많기 때문)

구글 드라이브 연동
"""

from google.colab import drive
drive.mount('/gdrive')

"""폴더 경로 설정"""

workspace_path = "/gdrive/My Drive/한밭대 20187100/4-1/컴퓨터비전/CV_과제7"

"""파이썬 패키지 로드"""

import os
import sys
sys.path.append(os.path.join(workspace_path, 'PyTorch_CIFAR10-master'))  # PyTorch_CIFAR10 반영

"""필요 패키지 로드"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import random
import numpy as np
import matplotlib.pyplot as plt

"""결과 재현을 위한 설정"""

seed = 719
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

"""합성곱 신경망(CNN) 정의"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3,
                      stride=1, padding=1, bias=False),  # 3x32x32 -> 10x32x32
            nn.BatchNorm2d(10),  # 배치 정규화
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3,
                      stride=1, padding=1, bias=False),  # 10x32x32 -> 20x32x32
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 20x32x32 -> 20x16x16 (특징 압축)
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3,
                      stride=1, padding=1, bias=False),  # 20x16x16 -> 40x16x16
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=3,
                      stride=1, padding=1, bias=False),  # 40x8x8 -> 80x8x8
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=80, out_channels=160, kernel_size=3,
                      stride=1, padding=1, bias=False),  
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=160, out_channels=320, kernel_size=3,
                      stride=1, padding=1, bias=False),  
            nn.BatchNorm2d(320),
            nn.AdaptiveAvgPool2d(1)  
        )
        self.fc = nn.Linear(320, 10)  # 출력값의 차원은 판별할 클래스 수인 10으로 설정 (CIFAR-10 10종 판별 문제)

    def forward(self, x):
        x = x.float()
        x = x.view(-1, 3, 32, 32)  # view 함수로 tensor 형태 변경: [batch크기, 3, 32, 32]
        x = self.main(x)  # CNN 모델 feed-forward
        x = x.view(-1, 320)  # view 함수로 형태 변경: [batch크기, 1600]
        x = self.fc(x)  # 마지막 레이어에는 활성화 함수 사용하지 않음
        return x

print("init model done")

"""모델 학습을 위한 하이퍼파라미터 셋팅"""

batch_size = 32  # 학습 배치 크기
test_batch_size = 3000  # 테스트 배치 크기 (학습 과정을 제외하므로 더 큰 배치 사용 가능)
max_epochs = 20  # 학습 데이터셋 총 훈련 횟수
lr = 0.1  # 학습률
momentum = 0.8  # SGD에 사용할 모멘텀 설정 (파라미터 업데이트 시 관성 효과 사용)
log_interval = 200  # interval 때마다 로그 남김

use_cuda = torch.cuda.is_available()  # GPU cuda 사용 여부 확인

device = torch.device("cuda" if use_cuda else "cpu")  # GPU cuda 사용하거나 없다면 CPU 사용

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}  # num_workers: data loading할 프로세스 수, pin_memory: 고정된 메모리 영역 사용

print("set vars and device done")

"""데이터 로더 정의 (학습용, 테스트용 따로 정의)"""

train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                    transforms.RandomErasing()])

val_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                    transforms.RandomErasing()])

# CIFAR-10 link: https://www.cs.toronto.edu/~kriz/cifar.html
# 학습용 데이터 로더 (CIFAR-10 학습 데이터셋 사용)
train_loader = torch.utils.data.DataLoader(
  datasets.CIFAR10(os.path.join(workspace_path, 'data'), train=True, download=True, 
                   transform=train_transform), 
    batch_size = batch_size, shuffle=True, drop_last=True, **kwargs)  # drop_last: 마지막 미니배치 크기가 batch_size 이하면 drop 

# 테스트용 데이터 로더 (CIFAR-10 테스트 데이터셋 사용)
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(os.path.join(workspace_path, 'data'), train=False, download=True,
                         transform=val_transform), 
    batch_size=test_batch_size, shuffle=False, **kwargs)

"""모델, 최적화 알고리즘, 손실 함수 정의"""

# model = Net().to(device)  # 모델 정의
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # 최적화 알고리즘 정의 (SGD 사용)
criterion = nn.CrossEntropyLoss()  # 손실 함수 정의 (CrossEntropy 사용)

# 전이학습 예제 코드
from cifar10_models.vgg import vgg11_bn
model = vgg11_bn(pretrained=True).to(device)  # VGG11 with BatchNorm pretrained model 로드

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
    for batch_idx, (data, target) in enumerate(train_loader):
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
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}, Accuracy: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), summary_loss.avg, summary_acc.avg))
            
    return summary_loss.avg, summary_acc.avg

def test(log_interval, model, device, test_loader):
    model.eval()  # 모델 검증 모드 설정 (inference mode)
    summary_loss = AverageMeter()  # 테스트 손실값 기록 초기화
    summary_acc = AverageMeter() # 테스트 정확도 기록 초기화
    with torch.no_grad():  # 검증 모드이므로 gradient 계산안함
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 현재 미니 배치의 데이터, 정답 불러옴
            output = model(data)  # 모델에 입력값 feed-forward
            loss = criterion(output, target)  # 예측값(클래스 별 score)과 정답간의 손실값 계산
            summary_loss.update(loss.detach().item())  # 손실값 기록
            pred = output.argmax(dim=1, keepdim=True)  # 예측값 중에서 최고 score를 달성한 클래스 선발
            correct = pred.eq(target.view_as(pred)).sum().item()  # 정답과 예측 클래스가 일치한 개수
            summary_acc.update(correct / data.size(0))  # 정확도 기록

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.6f}\n'.format
          (summary_loss.avg, summary_acc.avg))  # 정답을 맞춘 개수 / 테스트셋 샘플 수 -> Accuracy

    return summary_loss.avg, summary_acc.avg

"""학습, 테스트, 모델 저장 수행"""

best_acc = 0
best_epoch = 0
for epoch in range(1, max_epochs+1):
    train_loss, train_acc = train(log_interval, model, device, train_loader, optimizer, epoch)
    test_loss, test_acc = test(log_interval, model, device, test_loader)

    # 테스트에서 best accuracy 달성하면 모델 저장
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch
        torch.save(model, os.path.join(workspace_path, f'cifar10_cnn_model_best_acc_{best_epoch}-epoch.pt'))
        print(f'# save model: cifar10_cnn_model_best_acc_{best_epoch}-epoch.pt\n')

print(f'\n\n# Best accuracy model({best_acc * 100:.2f}%): cifar10_cnn_model_best_acc_{best_epoch}-epoch.pt\n')

"""# 실습과제

## baseline 모델 성능: 63.03%

## baseline보다 성능 높이기

### 1) baseline + 데이터증대 + 하이퍼파라미터 수정하여 성능 높이기 (다른 augmentation 패키지도 사용 가능)
torchvision augmentation 링크: https://pytorch.org/vision/stable/transforms.html

개선 모델 성능: 64.66%

개선 아이디어 설명: 
batch_size 64에서 32로 변경

lr 0.01에서 0.1로 변경


train_transform = transforms.Compose([
                 transforms.ToTensor(), 
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    transforms.RandomErasing()])

위에 식처럼 train_transform, val_transform 둘다 RandomHorizontalFlip 사용하도록 변경

### 2) 데이터증대 + 전이학습으로 성능 높이기 (다른 pretrained model도 사용 가능)
pretrained model 링크: https://github.com/huyvnphan/PyTorch_CIFAR10
"""

# 전이학습 예제 코드
from cifar10_models.vgg import vgg11_bn
model = vgg11_bn(pretrained=True).to(device)  # VGG11 with BatchNorm pretrained model 로드

"""개선 모델 성능: 81.51%

개선 아이디어 설명: 
학습 데이터셋의 평균, 표준편차의 값을 0.5에서 (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)으로 변경

### 3) 데이터증대 + 전이학습 + 추가 아이디어로 성능 높이기 (학습 과정 출력 포함)

개선 모델 성능: 82.60%

개선 아이디어 설명: 

네트워크에서 채널을 80에서 320으로 변경

test_batch_size 1000에서 3000으로 변경

momentum 0.5에서 0.8로 변경

max_epochs 10에서 20으로 변경
"""