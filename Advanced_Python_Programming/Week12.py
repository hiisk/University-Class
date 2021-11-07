id = '20200000'
name = '홍길동'
print(id, name)

"""제공한 베이스라인 코드 정도의 연산에서는 가속기 사용 안하는 것 권장함 (GPU 사용량이 많아지면 세션 유지시간도 짧아지고 시스템 성능도 낮아지기 때문)

하드웨어 가속기 GPU 필요한 경우,
코랩 메뉴 -> 수정 -> 노트 설정 -> 하드웨어 가속기 GPU 설정
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from sklearn.metrics import confusion_matrix, accuracy_score
from google.colab import drive
import os
drive.mount('/gdrive', force_remount=True)

# 아래 경로를 과제파일 저장한 경로로 수정하기 바랍니다.
workspace_path = '/gdrive/My Drive/Colab Notebooks/dacon/심리성향예측_AI경진대회'

# 경로 설정을 꼭 신경써주세요.
train = pd.read_csv(os.path.join(workspace_path, 'train.csv'), index_col=0)
test = pd.read_csv(os.path.join(workspace_path, 'test_x.csv'), index_col=0)
submission = pd.read_csv(os.path.join(workspace_path, 'submission_form.csv'), index_col=0)  

train.head()

import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns

# 결과 재현을 위한 설정
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sns.set() # setting seaborn default for plots

train_data = train.drop(train[train.familysize > 30].index)  # familysize outlier 제거
test_data = test
drop_list = ['QaE', 'QbE', 'QcE', 'QdE', 'QeE',
             'QfE', 'QgE', 'QhE', 'QiE', 'QjE',
             'QkE', 'QlE', 'QmE', 'QnE', 'QoE',
             'QpE', 'QqE', 'QrE', 'QsE', 'QtE']  # 질문 응답시간 특성 제거

replace_dict = {'education': str, 'engnat': str, 'married': str, 'urban': str, 'hand': str}  # one-hot vector로 바꿀 카테고리 정보는 문자열처리
train_y = train_data['voted']  # 정답 'voted' 추출
train_x = train_data.drop(drop_list + ['voted'], axis=1)  # drop_list와 'voted' 제외한 데이터로 train_x 구성
test_x = test_data.drop(drop_list, axis=1)  # drop_list 제외한 데이터로 test_x 구성
train_x = train_x.astype(replace_dict)  # 수치형 데이터를 문자열 데이터로 타입 변경 (one-hot vector 생성을 위함)
test_x = test_x.astype(replace_dict)
train_x = pd.get_dummies(train_x)  # 카테고리 정보 벡터화: LabelEncoding -> OneHotEncoder 자동 처리
test_x = pd.get_dummies(test_x)

train_x.describe()  # train 데이터 주요 통계치 확인

test_x.describe()  # test 데이터 주요 통계치 확인

# corr between QA and voted
pd_train = pd.concat([train_x.iloc[:, :20], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# corr between family_size and voted
pd_train = pd.concat([train_x.iloc[:, 20], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# corr between tp and voted
pd_train = pd.concat([train_x.iloc[:, 21:31], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# corr between (wf, wr) and voted
pd_train = pd.concat([train_x.iloc[:, 31:47], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# correlation between age_groups and voted
pd_train = pd.concat([train_x.iloc[:, 47:54], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# correlation between education and voted
pd_train = pd.concat([train_x.iloc[:, 54:59], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# correlation between engnat(native speaker) and voted
pd_train = pd.concat([train_x.iloc[:, 59:62], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# correlation between sex and voted
pd_train = pd.concat([train_x.iloc[:, 62:64], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# correlation between hand and voted
pd_train = pd.concat([train_x.iloc[:, 64:68], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# correlation between married and voted
pd_train = pd.concat([train_x.iloc[:, 68:72], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# correlation between race and voted (good)
pd_train = pd.concat([train_x.iloc[:, 72:79], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# correlation between religion and voted
pd_train = pd.concat([train_x.iloc[:, 79:91], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

# correlation between urban and voted
pd_train = pd.concat([train_x.iloc[:, 91:], train_y], axis=1)
correlations = pd_train.corr(method = 'spearman')
sns.heatmap(correlations, cmap="coolwarm", square=True, center=0)

train_x.head()

test_x.head()

train_y_np = 2 - train_y.to_numpy()  # [1, 2] -> [1, 0] 으로 레이블 정보 변경 (binary classification을 위함)
train_x_np = train_x.to_numpy()  # numpy array 로 변경
test_x_np = test_x.to_numpy()

train_x_np[:, :20] = (train_x_np[:, :20] - 3.) / 2.  # QA 값의 범위를 [-1, 1]로 변경
test_x_np[:, :20] = (test_x_np[:, :20] - 3.) / 2.
train_x_np[:, 21:31] = (train_x_np[:, 21:31] - 3.5) / 3.5  # TP 값의 범위를 [-1, 1]로 변경
test_x_np[:, 21:31] = (test_x_np[:, 21:31] - 3.5) / 3.5
train_y_t = torch.tensor(train_y_np, dtype=torch.float32)  # 딥러닝 모델에서 사용할 수 있는 tensor 형태로 변경
train_x_t = torch.tensor(train_x_np, dtype=torch.float32)
test_x_t = torch.tensor(test_x_np, dtype=torch.float32)
train_len, test_len = len(train_x_t), len(test_x_t)

# train 데이터셋 중의 임의의 1/5 를 validation 셋으로 설정
random_index = random.sample(range(len(train_x_t)), len(train_x_t))
val_x_t = train_x_t[random_index[:len(train_y_t) // 5]]
val_y_t = train_y_t[random_index[:len(train_y_t) // 5]]
train_x_t = train_x_t[random_index[len(train_y_t) // 5:]]
train_y_t = train_y_t[random_index[len(train_y_t) // 5:]]
train_len, val_len, test_len = len(train_x_t), len(val_x_t), len(test_x_t)
print(train_len, val_len, test_len)

N_EPOCH = 20  # 모델 학습 횟수
BATCH_SIZE = 512  # mini-batch 크기
N_MODEL = 3  # 학습할 모델 개수 (추후 결과 앙상블)
LOADER_PARAM = {
    'batch_size': BATCH_SIZE,
    'num_workers': 1,
    'pin_memory': True
}

pos_weight = sum(train_y_t == 0) / sum(train_y_t == 1)  # 정답 비율이 다른 경우 보정하기 위한 pos_weight 설정
print(train_x_t.size())
print(test_x_t.size())
print(torch.mean(train_y_t))
print(train_y_t)
print(val_y_t)
print(f'# of label 0: {sum(train_y_t == 0)}, # of label 1: {sum(train_y_t == 1)}')
print(f'# pos_weight: {pos_weight}')

# Multi-Layer Perceptron 구조 설계 (baseline)
class MLP_Net(nn.Module):
    def __init__(self):
        super(MLP_Net, self).__init__()
        # Linear -> BN -> ReLU -> Dropout -> Linear 순
        self.main = nn.Sequential(
            nn.Linear(train_x_t.size(1), 20, bias=False),
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # drop할 노드의 비율 설정 (train mode 에서만 동작)
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.main(x)

pred_test = np.zeros((test_len), dtype=np.float32)  # 테스트 예측값 배열 생성

# 같은 구조의 여러 개의 모델을 생성하여 예측값 앙상블
for no in range(N_MODEL):
    model = MLP_Net().to(DEVICE)  # 모델 생성 (생성 시점마다 모델 랜덤 초기화)
    best_model = ""
    best_auc = 0
    best_epoch = 0
    
    # 데이터로더 설정: train, val, test
    train_loader = DataLoader(TensorDataset(train_x_t, train_y_t),
                              shuffle=True, drop_last=True, **LOADER_PARAM)
    val_loader = DataLoader(TensorDataset(val_x_t, val_y_t),
                              shuffle=False, drop_last=True, **LOADER_PARAM)
    test_loader = DataLoader(TensorDataset(test_x_t, torch.zeros((test_len,), dtype=torch.float32)),
                             shuffle=False, drop_last=False, **LOADER_PARAM)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))  # 이진분류 손실 함수 설정
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # 학습기(optimizer) 설정

    for epoch in tqdm(range(N_EPOCH), desc=f'[{(no + 1):02d}/{N_MODEL:02d}]'):
        pred_train = np.zeros((train_len // BATCH_SIZE * BATCH_SIZE), dtype=np.float32)
        label_train = np.zeros((train_len // BATCH_SIZE * BATCH_SIZE), dtype=np.float32)
        pred_val = np.zeros((val_len // BATCH_SIZE * BATCH_SIZE), dtype=np.float32)
        label_val = np.zeros((val_len // BATCH_SIZE * BATCH_SIZE), dtype=np.float32)
        loss_train = 0
        loss_val = 0

        # training
        model.train()  # 모델 학습 모드 설정
        for idx, (xx, yy) in enumerate(train_loader):
            optimizer.zero_grad()  # gradient 초기화
            xx, yy = xx.to(DEVICE), yy.to(DEVICE)  # 현재 사용 중인 device 반영
            pred = model(xx).squeeze()  # feed-forward 과정: 모델에 입력값 넣어 출력값 획득
            loss = criterion(pred, yy)  # 예측값과 정답 이용하여 손실값 계산
            loss.backward()  # 손실값 이용하여 gradient 계산, back-propagation
            optimizer.step()  # 학습 파라미터 업데이트

            # loss, pred, label 로그 남기기
            loss_train += loss.item()
            pred = torch.sigmoid(pred.detach().to('cpu')).numpy()
            pred_train[BATCH_SIZE * idx:min(BATCH_SIZE * (idx + 1), len(pred_train))] = pred
            label = yy.detach().to('cpu').numpy()
            label_train[BATCH_SIZE * idx:min(BATCH_SIZE * (idx + 1), len(pred_train))] = label
        
        # validation
        model.eval()  # 모델 평가 모드 설정
        with torch.no_grad():
            for val_idx, (xx, yy) in enumerate(val_loader):
                optimizer.zero_grad()
                xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                pred = model(xx).squeeze()
                loss = criterion(pred, yy)

                # log loss, pred, label
                loss_val += loss.item()
                pred = torch.sigmoid(pred.detach().to('cpu')).numpy()
                pred_val[BATCH_SIZE * val_idx:min(BATCH_SIZE * (val_idx + 1), len(pred_val))] = pred
                label = yy.detach().to('cpu').numpy()
                label_val[BATCH_SIZE * val_idx:min(BATCH_SIZE * (val_idx + 1), len(pred_val))] = label

        # 평가척도 ROC_AUC 계산: train, val 데이터셋
        auc_train = roc_auc_score(label_train, pred_train)
        auc_val = roc_auc_score(label_val, pred_val)
        print(f'  loss_train: {loss_train / (idx + 1) :.4f}, auc_train: {auc_train:.4f}, loss_val: {loss_val / (val_idx + 1) :.4f}, auc_val: {auc_val:.4f}')

        # best_auc 이면 best_model에 반영하기, 모델 저장
        if auc_val > best_auc:
            best_auc = auc_val
            best_model = model
            best_epoch = epoch

    print(f'\n# Test prediction using the best_model(epoch: {best_epoch}, best_auc: {best_auc:.4f})')
    best_model.eval()  # best_model로 테스트 결과 생성
    with torch.no_grad():
        for idx, (xx, _) in enumerate(test_loader):
            xx = xx.to(DEVICE)
            pred = model(xx).squeeze()
            pred = (2. - torch.sigmoid(pred.detach().to('cpu'))).numpy()  # [1, 0] -> [1, 2] voted 정보로 복원
            pred_test[BATCH_SIZE * idx:min(BATCH_SIZE * (idx + 1), len(pred_test))] += pred / N_MODEL  # 모델 개수만큼 등분하여 예측값 반영

submission = pd.read_csv(os.path.join(workspace_path, 'submission_form.csv'))  # submission form 읽어오기
submission['voted'] = pred_test  # test prediction 값 반영
submission.to_csv(os.path.join(workspace_path, f"submission_{datetime.now().strftime('%m%d_%H%M')}.csv"), index=False)
submission.head()