import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import cv2
import pyzed.sl as sl
from sort.sort import *

# Load ZED Stereo
init = sl.InitParameters()

init.camera_resolution = sl.RESOLUTION.HD1080
init.camera_fps = 30

runtime = sl.RuntimeParameters()
cam = sl.Camera()

if not cam.is_opened:
    print("\nCannot open ZED Stereo!")
    exit()

status = cam.open(init)

if status != sl.ERROR_CODE.SUCCESS:
    print("\nCannot open ZED Stereo!")
    exit()

print("Successfully running ZED Stereo!")
# End of Load ZED Stereo




# Load Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model.classes = [0]
# End of Load Model

# Initial SORT
mot_tracker = Sort()

# Start looping
while True:
    # Take image from left and right lens
    left_image = sl.Mat()
    right_image = sl.Mat()

    err = cam.grab(runtime)

    if err != sl.ERROR_CODE.SUCCESS:
        print("Cannot load more images from ZED Stereo Camera!")
        break
    
    cam.retrieve_image(left_image, sl.VIEW.LEFT)
    result_left = left_image.get_data()

    cam.retrieve_image(right_image, sl.VIEW.RIGHT)
    result_right = right_image.get_data()
    # End of Take image from left and right lens


    # Start detecting with PyTorch Hub
    results = model([result_left, result_right])
    print(results.pandas().xyxy[0])
    print("\n")
    print(results.pandas().xyxy[1])
    # End of Start detecting with PyTorch Hub


    # Feed results into SORT for left to return unique id
    detections_left = results.pred[0].cpu().numpy()
    track_id_left = mot_tracker.update(detections_left)
    # End of Feed results into SORT for left to return unique id
    
    # Feed results into SORT for left to return unique id
    detections_right = results.pred[1].cpu().numpy()
    track_id_right = mot_tracker.update(detections_right)
    # End of Feed results into SORT for left to return unique id


    # Put result on image left
    for j in range(len(track_id_left.tolist())):
        coords = track_id_left.tolist()[j]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        name = f"ID: {int(coords[4])}"

        # Put bounding box and text
        color = (255, 0, 0)
        cv2.rectangle(result_left, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result_left, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2.9, color, 5)
        # End of Put bounding box and text
    # End of Put result on image left
    
    # Put result on image right
    for j in range(len(track_id_right.tolist())):
        coords = track_id_right.tolist()[j]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        name = f"ID: {int(coords[4])}"

        # Put bounding box and text
        color = (255, 0, 0)
        cv2.rectangle(result_right, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result_right, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2.9, color, 5)
        # End of Put bounding box and text
    # End of Put result on image right



    # Show images
    result_left = cv2.resize(result_left, (672, 376))
    result_right = cv2.resize(result_right, (672, 376))
    cv2.imshow('kamera kiri', result_left)
    cv2.imshow('kamera kanan', result_right)
    # End of Show images



    # Exit if pressed 'q'
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    # End of Exit if pressed 'q'

# Exit everything
cv2.destroyAllWindows()