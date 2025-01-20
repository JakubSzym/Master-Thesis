#!/usr/bin/env python3

from ultralytics import YOLO
from monodepth2 import monodepth2
import cv2
from ultralytics.utils.plotting import Annotator
import os
import numpy as np
import time

DATA_DIR = os.path.abspath('./test_data')
MD_DIR = os.path.abspath('./Monodepth2Images')
OUT_DIR = os.path.abspath('./test_md_results')
YOLO_MODEL = os.path.abspath("./yolov8/runs/detect/train3/weights/best.pt")

if not os.path.exists(MD_DIR):
  os.mkdir(MD_DIR)

if not os.path.exists(OUT_DIR):
  os.mkdir(OUT_DIR)

md = monodepth2()
yolo = YOLO(YOLO_MODEL)

times = 0
images = len(os.listdir(DATA_DIR))

for filename in os.listdir(DATA_DIR):
  start = time.time()
  imgname = os.path.join(DATA_DIR, filename)
  detection_results = yolo.predict(imgname)
  depth_results = md.eval(detection_results[0].orig_img)

  width = depth_results.shape[1]
  left_pivot = int(width / 3)
  right_pivot = 2 * left_pivot

  # cv2.imwrite(os.path.join(MD_DIR, filename), depth_results)
  print(f"Image: {filename}")
  for r in detection_results:
    annotator = Annotator(r.orig_img)
  
    boxes = r.boxes
    for box in boxes:
            
      b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
      left, top, right, bottom = b

      x = int((left + right) / 2)
      y = int((top + bottom) / 2)

      center = depth_results[y][x]
    
      side = ""
      if x <= left_pivot:
        side = "LEFT"
      elif x <= right_pivot:
        side = "CENTER"
      else:
        side = "RIGHT"

      value = int(np.mean(center))

      print(f"Object: {r.names[int(box.cls)]}, Depth Value: {value}")

      c = box.cls
    
      value /= 2.55

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
