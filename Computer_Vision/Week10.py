"""
# 코드자료 링크: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/

### This notebook is optionally accelerated with a GPU runtime.
### If you would like to use this acceleration, please select the menu option "Runtime" -> "Change runtime type", select "Hardware Accelerator" -> "GPU" and click "SAVE"

----------------------------------------------------------------------

# FCN-ResNet101

*Author: Pytorch Team*

**Fully-Convolutional Network model with a ResNet-101 backbone**

_ | _
- | -
![alt](https://pytorch.org/assets/images/deeplab1.png) | ![alt](https://pytorch.org/assets/images/fcn2.png)
"""

import torch
model = torch.hub.load('pytorch/vision:v0.9.0', 'fcn_resnet101', pretrained=True)
model.eval()

"""All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(N, 3, H, W)`, where `N` is the number of images, `H` and `W` are expected to be at least `224` pixels.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

The model returns an `OrderedDict` with two Tensors that are of the same height and width as the input Tensor, but with 21 classes.
`output['out']` contains the semantic masks, and `output['aux']` contains the auxillary loss values per-pixel. In inference mode, `output['aux']` is not useful.
So, `output['out']` is of shape `(N, 21, H, W)`. More documentation can be found [here](https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection).
"""

# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms, datasets
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

"""The output here is of shape `(21, H, W)`, and at each location, there are unnormalized probabilities corresponding to the prediction of each class.
To get the maximum prediction of each class, and then use it for a downstream task, you can do `output_predictions = output.argmax(0)`.

Here's a small snippet that plots the predictions, with each color being assigned to each class (see the visualized image on the left).
"""

import matplotlib.pyplot as plt
plt.imshow(input_image)
plt.show()

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

plt.imshow(r)
# plt.show()

"""### Model Description

FCN-ResNet101 is constructed by a Fully-Convolutional Network model with a ResNet-101 backbone.
The pre-trained models have been trained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.

Their accuracies of the pre-trained models evaluated on COCO val2017 dataset are listed below.

| Model structure |   Mean IOU  | Global Pixelwise Accuracy |
| --------------- | ----------- | --------------------------|
|  fcn_resnet101  |   63.7      |   91.9                    |

### Resources

 - [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211)

# 수강생분의 이름, 학번을 반영해주세요.
"""

id = '20187100'
name = '강현석'
print(id, name)

"""구글 드라이브 연동"""

from google.colab import drive
drive.mount('/gdrive')

"""실습, 샘플이미지 폴더 경로 설정"""

import os
import glob

workspace_path = "/gdrive/My Drive/한밭대 20187100/4-1/컴퓨터비전/CV_과제10"  # 실습파일 저장 경로 설정
img_path = os.path.join(workspace_path, 'testing')  # 테스트 샘플이미지 폴더 경로 설정
img_list = glob.glob(os.path.join(img_path, '*.jpg'))  # 샘플이미지 목록 추출

img_list  # 샘플 이미지 목록

import tqdm

model.eval()  # Set model to evaluation mode

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

for img_name in tqdm.tqdm(img_list, desc="Segmentation"):

    input_image = Image.open(img_name)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    plt.imshow(input_image)
    plt.show()

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    plt.imshow(r)
    plt.show()

"""# 실습과제

## Kaggle 에서 open dataset 다운로드 받아서 segmentation 결과 출력하기
- 객체 탐지용 open dataset 목록: https://www.kaggle.com/datasets?search=object+detection
- 1) 사람, 자동차, 개, 자전거, 의자, 말, 새 등 탐지할만한 데이터셋 찾아서 다운로드 (pretrained 모델은 COCO 데이터셋 중 20개 클래스에 대해서만 학습 진행함)
- 2) 데이터셋에 포함된 이미지가 너무 많다면 랜덤하게 20장만 선별하기
- 3) FCN-ResNet101 모델로 segmentation 결과 출력하기(예제 코드 수정)
- 4) Segmentation 결과 분석하기
* 유의사항: 데이터셋을 코랩에 업로드한 이후에 구글 드라이브 마운트하기 (파일 업로드 이후에 마운트해야 파일 접근 가능)

실습문제 1. Segmentation 결과 출력(예제 코드 수정하여 출력결과 보이기)

실습문제 2. 선택한 데이터셋 정보(이름, 다운로드 링크): 

Car Object Detection

https://www.kaggle.com/sshikamaru/car-object-detection

실습문제 3. Segmentation 결과 분석(어떤 경우에 Segmentation 결과가 우수/미흡한지?): 

도로에 달리는 차같은 경우 다른 동물들에 비해서 결과가 미흡하다.
"""