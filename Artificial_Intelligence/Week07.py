# 파이썬 ≥3.5 필수
import sys
assert sys.version_info >= (3, 5)


# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.20"


# 공통 모듈 임포트
import numpy as np
import os


# 노트북 실행 결과를 동일하게 유지하기 위해
np.random.seed(42)


# 깔끔한 그래프 출력을 위해
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X.shape  # 28x28 이미지를 1D 벡터로 변환한 입력데이터 X
y.shape  # 손글씨 숫자 이미지에 대한 레이블 Y
28 * 28


# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)  # 벡터를 28x28 이미지로 변환
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
save_fig("some_digit_plot")
plt.show()


y[0]


y = y.astype(np.uint8)
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")


# 숫자 그림을 위한 추가 함수
def plot_digits(instances, images_per_row=10, **options):
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
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot")
plt.show()


y[0]


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]  # 학습, 테스트용 데이터셋 분리시킴


"""# 이진 분류기
### 코드 내의 파라미터 값 변경해가면서 실습해보세요.
"""


y_train_5 = (y_train == 5)  # 숫자 5에 대한 학습 레이블 선별
y_test_5 = (y_test == 5)  # 숫자 5에 대한 학습 레이블 선별

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)  # SGD 분류기 생성. 현 시점에서는 블랙박스로 보기 -> 추후 구체적으로 설명
sgd_clf.fit(X_train, y_train_5)  # SGD 분류기 학습
sgd_clf.predict([some_digit])  # SGD 분류기 예측(inference)


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")  # 교차 검증(cross validation) 진행. 3-fold 사용. 각 fold 스코어 출력

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)  # 교차 검증 진행. 3-fold 사용. 각 fold 검증 결과값 종합하여 출력


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)  # 오차 행렬 생성: 정답(row)에 대한 예측(col) 행렬


from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)  # 정밀도 계산
#3530 / (3530 + 687)


recall_score(y_train_5, y_train_pred)  # 재현율 계산
#3530 / (3530 + 1891)

from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)  # F1 점수 계산

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
y_scores.shape
# input -> model -> output(score == decision_function) -> sigmoid(output) -> predicted_class(0 or 1)
# input -> model -> output(score == decision_function) -> softmax(output) -> argmax -> predicted_class

"""# ROC 곡선"""

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)  # ROC 곡선 그리기에 필요한 정보 추출


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:")
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")
plt.plot([4.837e-3], [0.4368], "ro")
save_fig("roc_curve_plot")
plt.show()


from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)  # ROC AUC 계산

"""# 다중 분류
### 코드 내의 파라미터 값 변경해가면서 실습해보세요.
"""


from sklearn.svm import SVC

svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000]) # 무작위로 섞인 MNIST 데이터셋 중 0~999 샘플로 학습
svm_clf.predict([some_digit])  # MNIST 데이터셋의 0번째 샘플 (X[0])


some_digit_scores = svm_clf.decision_function([some_digit])  # 예측값 생성 전의 클래스(0~9) 별 score 값 추출 
some_digit_scores


np.argmax(some_digit_scores)


svm_clf.classes_  # 정답 클래스 출력


svm_clf.classes_[5]  # 정답 클래스 중 가장 score가 높은 클래스 return