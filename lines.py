#!/usr/bin/env python3

import cv2
import numpy as np
from argparse import ArgumentParser
import time

# Sources:
# https://medium.com/analytics-vidhya/lane-detection-for-a-self-driving-car-using-opencv-e2aa95105b89
# https://www.labellerr.com/blog/real-time-lane-detection-for-self-driving-cars-using-opencv/

def grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def blur(image):
  return cv2.GaussianBlur(image, (3,3), 0)

def edge_detector(image):
  return cv2.Canny(image, 50, 150)

def get_roi(image):
  height = image.shape[0]
  width = image.shape[1]
  polygon = np.array([[(0, height), 
                         (0, int(height*0.66)), (int(width*0.33), int(height*0.4)), 
                         (int(width*0.66),int(height*0.4)), (width, int(height*0.66)), 
                         (width, height)]])
  
  # triangle = np.array([[(0,height), (int(width*0.5), int(height*0.4)), (width, height)]])

  # trapezoid = np.array([[(0,height), (int(width*0.33), int(height*0.4)), 
  #                        (int(width*0.66), int(height*0.4)), (width, height)]])

  black_image = np.zeros_like(image)
  mask = cv2.fillPoly(black_image, polygon, 255)
  return cv2.bitwise_and(image, mask)

def get_lines(image):
  return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20,
                         minLineLength=10, maxLineGap=100)

def average_slope_intercept(lines):
  left_lines = []
  left_weights = []
  right_lines = [] 
  right_weights = []
	
  for line in lines:
    for x1, y1, x2, y2 in line:
      if x1 == x2:
        continue
			
      slope = (y2 - y1) / (x2 - x1)
			
      intercept = y1 - (slope * x1)
			
      length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
			
      if slope < 0:
        left_lines.append((slope, intercept))
        left_weights.append((length))
      else:
        right_lines.append((slope, intercept))
        right_weights.append((length))

  left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
  right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
  return left_lane, right_lane

def pixel_points(y1, y2, line):
	if line is None:
		return None
	slope, intercept = line
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	y1 = int(y1)
	y2 = int(y2)
	return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
	left_lane, right_lane = average_slope_intercept(lines)
	y1 = image.shape[0]
	y2 = y1 * 0.6
	left_line = pixel_points(y1, y2, left_lane)
	right_line = pixel_points(y1, y2, right_lane)
	return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
	line_image = np.zeros_like(image)
	for line in lines:
		if line is not None:
			cv2.line(line_image, *line, color, thickness)
	return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def draw_hough_lines(image, lines, thickness=2):
  line_image = np.zeros_like(image)
  for line in lines:
    x1, y1, x2, y2 = line[0]

    if x1 == x2:
      continue
			
    slope = (y2 - y1) / (x2 - x1)
    if slope < 0:
      cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), thickness)
    else:
      cv2.line(line_image, (x1, y1), (x2, y2), (0,0,255), thickness)
  return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def on_track(image, lines):
  if len(lines) != 2:
    return False
  
  left_line, right_line = lines
  ((xr1, yr1), (xr2, yr2)) = right_line
  ((xl1, yl1), (xl2, yl2)) = left_line

  slope_right = (yr2 - yr1)/(xr2 - xr1)
  intercept_right = yr1 - slope_right * xr1

  slope_left = (yl2 - yl1)/(xl2 - xl1)
  intercept_left = yl1 - slope_left * xl1

  if slope_left == slope_right:
    return False

  x = (intercept_right - intercept_left)/(slope_left - slope_right)
  y = slope_left * x + intercept_left

  width, height = image.shape[0], image.shape[1]

  if x >= width * 0.25 and x <= width * 0.75 and y <= height * 0.6:
    return True
  
  return False

parser = ArgumentParser()
parser.add_argument("--img")

args = parser.parse_args()

imgname = args.img

image = cv2.imread(imgname)

start = time.time()

# gray = grayscale(image)

# cv2.imwrite("Gray.jpg", gray)

blurred = blur(image)

# cv2.imwrite("Blurred.jpg", blurred)

edged_image = edge_detector(blurred)

# cv2.imwrite("Edges_img2_80_180.jpg", edged_image)

roi_image = get_roi(edged_image)

# cv2.imwrite("ROI_testImage_trapezoid.jpg", roi_image)

lines = get_lines(roi_image)

lines_img = draw_hough_lines(image, lines)

cv2.imwrite("HoughLines.jpg", lines_img)

lines = lane_lines(image, lines)

image_with_lines = draw_lane_lines(image, lines)

print(f"Time: {time.time() - start}")

cv2.imwrite("hough_50_60_testImage.jpg", image_with_lines)

if on_track(image, lines):
  print("On track")
else:
  print("Out of track")