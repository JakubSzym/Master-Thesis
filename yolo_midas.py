#!/usr/bin/env python3

from ultralytics import YOLO
import cv2
import torch
from ultralytics.utils.plotting import Annotator
import os
import time

DATA_DIR = os.path.abspath('./test_data')
MIDAS_DIR = os.path.abspath('./MidasImages')
OUT_DIR = os.path.abspath('./test_results')
YOLO_MODEL = os.path.abspath("./yolov8/runs/detect/train3/weights/best.pt")

if not os.path.exists(MIDAS_DIR):
  os.mkdir(MIDAS_DIR)

if not os.path.exists(OUT_DIR):
  os.mkdir(OUT_DIR)

midas = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
midas.to('cpu')
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS','transforms')
transform = transforms.small_transform

yolo = YOLO(YOLO_MODEL)

times = 0
images = len(os.listdir(DATA_DIR))

for filename in os.listdir(DATA_DIR):
  start = time.time()

  imgname = os.path.join(DATA_DIR, filename)
  detection = yolo(imgname)
  imgbatch = transform(detection[0].orig_img).to('cpu')

  with torch.no_grad():
    prediction = midas(imgbatch)
    prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=detection[0].orig_shape[:2],
    mode='bicubic',
    align_corners=False
        ).squeeze()

  midas_output = prediction.cpu().numpy()

  midas_output = 255 * cv2.normalize(midas_output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  # cv2.imwrite(os.path.join(MIDAS_DIR, filename), midas_output)
  print(f"Image: {filename}")
  for r in detection:
    annotator = Annotator(r.orig_img)

    width = midas_output.shape[1]
    left_pivot = int(width / 3)
    right_pivot = width - left_pivot

    boxes = r.boxes
    for box in boxes:
            
      b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
      left, top, right, bottom = b

      x = int((left + right) / 2)
      y = int((top + bottom) / 2)
    
      side = ""
      if x <= left_pivot:
        side = "LEFT"
      elif x <= right_pivot:
        side = "CENTER"
      else:
        side = "RIGHT"

      value = midas_output[y][x]

      print(f"Object: {r.names[int(box.cls)]}, Depth Value: {int(value)}")
      value /= 2.55

      c = box.cls

      label = ""
      if value > 50:
        label = "VERY CLOSE"
      elif value > 30:
        label = "CLOSE"
      elif value > 15:
        label = "MEDIUM"
      elif value > 5:
        label = "FAR"
      else:
        label = "VERY FAR"

      annotator.box_label(b, f"{r.names[int(c)]}, {label}, {side}")
  
  end = time.time() - start
  times += end
  img = annotator.result()
  cv2.imwrite(os.path.join(OUT_DIR, filename), img)

print(f"FPS: {images / times}")