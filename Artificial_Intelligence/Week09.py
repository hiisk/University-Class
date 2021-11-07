# 수강생분의 이름, 학번을 반영해주세요.
id = '20187100'
name = '강현석'
print(id, name)

"""코랩 메뉴 -> 수정 -> 노트 설정 -> GPU 설정

구글 드라이브 연동
"""

from google.colab import drive
drive.mount('/gdrive')


"""폴더 경로 설정"""

workspace_path = '/gdrive/My Drive/Colab Notebooks/AI_과제9'


"""필요 패키지 로드"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt


"""인공신경망(ANN) 정의"""

class Net(nn.Module):
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.fc1 = nn.Linear(32*32*3, 256)  # 3072차원(32x32x3) -> 256차원으로 축소
    #     self.fc2 = nn.Linear(256, 32)
    #     self.fc3 = nn.Linear(32, 10)  # 출력값의 차원은 판별할 클래스 수인 10으로 설정 (MNIST는 손글씨 숫자 10종 판별 문제)

    # def forward(self, x):
    #     x = x.float()
    #     h1 = F.relu(self.fc1(x.view(-1, 32*32*3)))  # view 함수로 tensor 형태 변경 (배치수, axis_0, axis_1, ...)
    #     h2 = F.relu(self.fc2(h1))
    #     h3 = self.fc3(h2)
    #     return h3 

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
print("init model done")


"""모델 학습을 위한 하이퍼파라미터 셋팅"""

batch_size = 64  # 학습 배치 크기
test_batch_size = 1000  # 테스트 배치 크기 (학습 과정을 제외하므로 더 큰 배치 사용 가능)
max_epochs = 10  # 학습 데이터셋 총 훈련 횟수
lr = 0.01  # 학습률
momentum = 0.9  # SGD에 사용할 모멘텀 설정 (추후 딥러닝 최적화 과정에서 설명)
seed = 1  # 결과 재현을 위한 seed 설정
log_interval = 200  # interval 때마다 로그 남김
use_cuda = torch.cuda.is_available()  # GPU cuda 사용 여부 확인
torch.manual_seed(seed)  # 결과 재현을 위한 seed 설정
device = torch.device("cuda" if use_cuda else "cpu")  # GPU cuda 사용하거나 없다면 CPU 사용
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}  # num_workers: data loading할 프로세스 수, pin_memory: 고정된 메모리 영역 사용
print("set vars and device done")


"""데이터 로더 정의 (학습용, 테스트용 따로 정의)"""

transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 임의의 값으로 초기화 (일반적으로는 학습 데이터셋의 평균, 표준편차 사용)

# CIFAR-10 link: https://www.cs.toronto.edu/~kriz/cifar.html
# 학습용 데이터 로더 (CIFAR-10 학습 데이터셋 사용)
train_loader = torch.utils.data.DataLoader(
  datasets.CIFAR10(os.path.join(workspace_path, 'data'), train=True, download=True, 
                   transform=transform), 
    batch_size = batch_size, shuffle=True, drop_last=True, **kwargs)  # drop_last: 마지막 미니배치 크기가 batch_size 이하면 drop 

# 테스트용 데이터 로더 (CIFAR-10 테스트 데이터셋 사용)
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(os.path.join(workspace_path, 'data'), train=False, download=True,
                         transform=transform), 
    batch_size=test_batch_size, shuffle=True, **kwargs)


"""모델, 최적화 알고리즘, 손실 함수 정의"""

model = Net().to(device)  # 모델 정의
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # 최적화 알고리즘 정의 (SGD 사용)
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

for epoch in range(1, max_epochs+1):
    best_acc = 0
    train_loss, train_acc = train(log_interval, model, device, train_loader, optimizer, epoch)
    test_loss, test_acc = test(log_interval, model, device, test_loader)

    # 테스트에서 best accuracy 달성하면 모델 저장
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model, os.path.join(workspace_path, f'cifar10_ann_model_best_acc_{epoch}-epoch.pt'))
        print(f'# save model: cifar10_ann_model_best_acc_{epoch}-epoch.pt\n')

"""# 실습과제

1) 네트워크 구조 개선하여 baseline보다 성능 높이기

2) 네트워크 구조 외의 요소 개선하여 baseline보다 성능 높이기

3) 개선 아이디어 설명서 작성: 다양한 네트워크 구조 개선 아이디어, 네트워크 외의 요소 개선 아이디어

4) 설명 동영상 제작: 솔루션 노트북 설명, 시도해본 아이디어 설명, 설명서(pdf) 내용 설명 (음성 녹음 필수, 자막/기타 효과는 선택)

**과제 제출물: 솔루션 노트북, 최고 성능 모델, 솔루션 설명서, 설명 녹화 동영상**

baseline 성능: 53.74%
"""
#1.인공신경망을 합성곱 신경망으로 바꾸었다
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
print("init model done")

#2. momentum 0.5 0.9 을 에서 로 변환

#어려웠던 점은 지금까지 배운 것 중에서 접근성이 가장 높은 내용이었고 구글링을 하였을 때 한국말로 설명되어 있는 게 많지 않아서 참조하기가 힘들었습니다.