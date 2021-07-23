import torch
import cv2
import numpy as np
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
import torch.optim as optim

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
NUM_WORKERS = 2
WEIGHT_DECAY = 0
LOAD_MODEL = True
LOAD_MODEL_FILE = "checkpoints/chkpnt48.pth.tar"

model = Yolov1(split_size=7, num_boxes=2, num_classes=8).to(DEVICE)
optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

vid = cv2.VideoCapture('vidd.mp4')
  
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    image = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = cv2.resize(image,(448,448))
    x = image.reshape(3,448,448)
    x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

    bboxes = cellboxes_to_boxes(model(x))
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    plot_image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), bboxes)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()