#상대경로
sam_path = "./segment-anything"
sam_model_path = "./model/sam_vit_l_0b3195.pth"
yolo_model_path = "./model/ship.pt"
test_image_path = '/workspace/dataset/'
yolo = './model/yolo'

#load package
import sys
sys.path.append(sam_path)

import gc

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset, Dataset,DatasetDict
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
from io import BytesIO
import matplotlib.pyplot as plt

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
from torchvision.models import resnet50

import cv2
import numpy as np
import supervision as sv
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import glob
import os
import pandas as pd

from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# dataset
file_list = os.listdir(test_image_path)
df = pd.DataFrame({"image_name":file_list})

class CustomDataset():
  def __init__(self, df, transform=None):
    self.df = df
    self.transform = transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    image_name = self.df['image_name'][idx]
    image_path = os.path.join(test_image_path, image_name)
    image = Image.open(image_path)

    original_size = image.size  # (width, height)

    if self.transform is not None:
      image = self.transform(image)

    if isinstance(image, Image.Image):
      image_bgr = np.array(image)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_name, image_rgb, original_size

transform = transforms.Compose([
    transforms.Resize((4096, 4096))
])


dataset = CustomDataset(df,transform = transform)

# model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_l"
SAM_CHECKPOINT_PATH = sam_model_path

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Yolo
feature_extractor = YolosFeatureExtractor.from_pretrained(yolo)

MODEL_NAME = yolo_model_path
# Load a model
model = YOLO(MODEL_NAME)

def callback(image: np.ndarray) -> sv.Detections:
    # YOLOv8 모델에 이미지 입력 후 결과 받기 (confidence threshold 포함)
    result = model(image, conf=0.4)[0]  # YOLOv8의 예측 결과
    return sv.Detections.from_ultralytics(result)


slicer = sv.InferenceSlicer(
    callback=callback,
    slice_wh=(24, 24),            # 슬라이스 크기 (512x512)
    overlap_wh=(0.2, 0.2)          # 슬라이스 간 겹침 비율 (20%)
)

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def detect_bbox(image_name, image, original_size):
  detections = slicer(image=image)
  # prediction_num = len(detections.xyxy)

  probas = detections.confidence
  idx = np.array([0] * len(probas.tolist()))

  xyxy = detections.xyxy

  # convert detections to masks
  mask = segment(
      sam_predictor=sam_predictor,
      image = image, #cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
      xyxy=xyxy
  )

  detections = sv.Detections(xyxy)
  detections.mask = mask
  detections.confidence = probas
  detections.class_id = idx
  detections.tracker_id = None
  detections.data = {}

  # convert detections to masks
  detections.mask = segment(
      sam_predictor=sam_predictor,
      image = image, #cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
      xyxy=detections.xyxy
  )

  total_len = len(detections.mask)

  center_x = []
  center_y = []
  widths = []
  heights = []
  angle = []

  for i in range(total_len):
    # Apply the mask on the original image

    height, width, channels = image.shape

    # 흰색으로 채워진 이미지 생성 (픽셀 값이 255로 설정됨)
    white_image = np.ones((height, width, channels), dtype=np.uint8) * 255

    inverted_mask = cv2.bitwise_not(detections.mask[i].astype(np.uint8) * 255)

    # 반전된 마스크를 적용한 이미지
    inverted_masked_image = cv2.bitwise_and(white_image, white_image, mask=inverted_mask)

    # 이미지의 검정색 부분을 찾기 위해 이진화
    gray = cv2.cvtColor(inverted_masked_image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

    # 검은색 부분을 감싸는 최소 회전 바운딩 박스 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
      pass
    else:
      cnt = contours[0]
      rect = cv2.minAreaRect(cnt)

      cx , cy = rect[0]
      w , h = rect[1]
      ang = rect[2]

      ratio = w / h if h != 0 else 0

      # 정사각형 또는 직사각형 판단
      if 0.5 <= ratio <= 2:
        pass
      else:
        resize_ratio_x = original_size[0] / 4096
        resize_ratio_y = original_size[1] / 4096

        center_x.append(cx * resize_ratio_x)
        center_y.append(cy * resize_ratio_y)
        widths.append(w * resize_ratio_x * 1.1)
        heights.append(h * resize_ratio_y * 1.1)
        angle.append(ang)

  df =  pd.DataFrame({'image_name': image_name, 'cx':center_x,'cy':center_y,'width':widths,'height':heights,'angle':angle})
  return df

total_df = pd.DataFrame()

for i in range(len(dataset)):
  image_name, image, original_size = dataset[i]
  df1 = detect_bbox(image_name, image, original_size)
  total_df = pd.concat([total_df,df1])
  torch.cuda.empty_cache()
  gc.collect()

total_df.to_csv('submission.csv',index=False)