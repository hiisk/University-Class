id = '20187100'
name = '강현석'
print(id, name)

"""코랩 메뉴 -> 수정 -> 노트 설정 -> 하드웨어 가속기 GPU 권장(행렬 연산이 많기 때문)

구글 드라이브 연동
"""

from google.colab import drive
drive.mount('/gdrive')

"""폴더 경로 설정"""

import os
workspace_path = "/gdrive/My Drive/한밭대 20187100/4-1/컴퓨터비전/CV_과제9"  # 실습파일 저장 경로 반영
yolov3_path = os.path.join(workspace_path, 'PyTorch-YOLOv3-master')
sample_path = os.path.join(yolov3_path, 'data/samples')

"""YOLOv3 소스코드 경로 설정"""

import sys
sys.path.append(yolov3_path)  # YOLOv3 소스코드 경로 설정

"""필요 패키지 설치 후, 메뉴 -> 런타임 -> 런타임 다시 시작"""

!pip3 install numpy
!pip3 install torch>=1.2
!pip3 install torchvision
!pip3 install matplotlib
!pip3 install tensorboard>=1.15
!pip3 install terminaltables
!pip3 uninstall pillow -y
!pip3 install pillow
!pip3 install tqdm
!pip3 uninstall imgaug -y
!pip3 install imgaug
!pip3 install torchsummary

"""YOLOv3 pretrained model 다운로드"""

# Download weights for vanilla YOLOv3
!wget -c "https://pjreddie.com/media/files/yolov3.weights" --header "Referer: pjreddie.com"

ls  # yolov3.weights 파일 다운로드 확인

"""# YOLOv3 Inference

파이썬 패키지 로드
"""

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *

import os
import argparse
import tqdm
import numpy as np

from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

"""객체 탐지 함수 정의"""

def detect_directory(model_path, weights_path, img_path, classes, output_path, 
    batch_size=8, img_size=416, n_cpu=8, conf_thres=0.5, nms_thres=0.5):
    """Detects objects on all images in specified directory and saves output images with drawn detections.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to directory with images to inference
    :type img_path: str
    :param classes: List of class names
    :type classes: [str]
    :param output_path: Path to output directory
    :type output_path: str
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    """
    dataloader = _create_data_loader(img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    img_detections, imgs = detect(
        model,
        dataloader,
        output_path,
        img_size,
        conf_thres,
        nms_thres)
    _draw_and_save_output_images(img_detections, imgs, img_size, output_path)

def detect_image(model, image,
    img_size=416, conf_thres=0.5, nms_thres=0.5):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image
    :rtype: nd.array
    """
    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS, 
        Resize(img_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return to_cpu(detections).numpy()

def detect(model, dataloader, output_path,
    img_size, conf_thres, nms_thres):
    """Inferences images with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images to inference
    :type dataloader: DataLoader
    :param output_path: Path to output directory
    :type output_path: str
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: List of detections. The coordinates are given for the padded image that is provided by the dataloader. 
        Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
        List of input image paths
    :rtype: [Tensor], [str]
    """
    # Create output directory, if missing
    os.makedirs(output_path, exist_ok=True)

    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections
        img_detections.extend(detections)
        imgs.extend(img_paths)
    return img_detections, imgs

"""결과 출력 함수 정의"""

def _draw_and_save_output_images(img_detections, imgs, img_size, output_path):
    """Draws detections in output images and stores them.

    :param img_detections: List of detections
    :type img_detections: [Tensor]
    :param imgs: List of paths to image files
    :type imgs: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    """
    # TODO: Draw class names...
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Iterate through images and save plot of detections
    for (image_path, detections) in tqdm.tqdm(zip(imgs, img_detections), desc="Saving output images"):
        _draw_and_save_output_image(image_path, detections, img_size, colors, output_path)

def _draw_and_save_output_image(image_path, detections, img_size, colors, output_path):
    """Draws detections in output image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param colors: List of colors used to draw detections
    :type colors: []
    :param output_path: Path of output directory
    :type output_path: str
    """
    # Create plot
    img = np.array(Image.open(image_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_pred in detections:

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")
        
        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=classes[int(cls_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0})

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{filename}.png")
    plt.show()  # colab 상에서 이미지 출력
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)  # 이미지 파일 출력
    plt.close()

"""데이터로더 정의"""

def _create_data_loader(img_path, batch_size, img_size, n_cpu):
    """Creates a DataLoader for inferencing.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader

"""하이퍼파라미터 설정"""

CFG = {
    'model': os.path.join(yolov3_path, "config/yolov3.cfg"),  # YOLOv3 모델 하이퍼파라미터 설정
    'weights': "yolov3.weights",  # pretrained model 가중치 경로
    'images': os.path.join(yolov3_path, "data_car/testing"),  # 테스트 이미지 경로
    'classes': os.path.join(yolov3_path, "data/coco.names"),  # 테스트 레이블 이름
    'output': workspace_path,  # 테스트 출력 결과 저장 경로
    'batch_size': 1,  # 미니배치 크기
    'img_size': 416,  # 입력 이미지 크기
    'n_cpu': 1,  # 데이터로딩 시 사용할 CPU thread 개수
    'conf_thres': 0.5,  # 객체 탐지 confidence 임계치
    'nms_thres': 0.4  # Non-maximum supression을 위한 IoU 임계치
}
print(CFG)

# Extract class names from file
classes = load_classes(CFG['classes'])  # List of class names

"""YOLOv3 모델 학습에 사용한 데이터셋 레이블 정보 (탐지 가능한 객체 종류)"""

print(classes)

"""YOLOv3 이용한 탐지"""

import random
detect_directory(
    CFG['model'],
    CFG['weights'],
    CFG['images'],
    classes,
    CFG['output'],
    batch_size=CFG['batch_size'],
    img_size=CFG['img_size'],
    n_cpu=CFG['n_cpu'],
    conf_thres=CFG['conf_thres'],
    nms_thres=CFG['nms_thres'])

"""# 실습과제

## Kaggle 에서 open dataset 다운로드 받아서 탐지 결과 출력하기
- 객체 탐지용 open dataset 목록: https://www.kaggle.com/datasets?search=object+detection
- 1) YOLOv3 모델 학습에 사용한 데이터셋 레이블 정보 참고하여 탐지할만한 데이터셋 찾아서 다운로드
- 2) 데이터셋에 포함된 이미지가 너무 많다면 랜덤하게 20장만 선별하기
- 3) YOLOv3 모델로 탐지 결과 출력하기(예제 코드 수정)
- 4) 탐지 결과 분석하기
* 유의사항: 데이터셋을 코랩에 업로드한 이후에 구글 드라이브 마운트하기 (파일 업로드 이후에 마운트해야 파일 접근 가능)

실습문제 1. 탐지 결과 출력(예제 코드 수정하여 출력결과 보이기)

실습문제 2. 선택한 데이터셋 정보(이름, 다운로드 링크): 

Car Object Detection

https://www.kaggle.com/sshikamaru/car-object-detection

실습문제 3. 탐지 결과 분석(어떤 경우에 탐지 결과가 우수/미흡한지?): 

임계치가 0.5인경우가 0.2인경우보다 Confidence값이 높았다.

0.2같은경우 그림에서 잘안보이는 자동차까지 인식을 시킨다.

### YOLOv3 코드자료 출처: https://github.com/eriklindernoren/PyTorch-YOLOv3
"""