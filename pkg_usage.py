import os
import logging
import numpy as np
import urllib
import time
import cv2
from pathlib import Path
from yolact_edge.inference import YOLACTEdgeInference
from yolact_edge.utils.logging_helper import setup_logger


setup_logger(logging_level=logging.INFO)


weights = "weights/yolact_edge_resnet50_54_800000.pth"
# All available model configs, depends on which weights
# you use. More info could be found in data/config.py.
model_configs = [
    'yolact_edge_mobilenetv2_config',
    'yolact_edge_vid_config',
    'yolact_edge_vid_minimal_config',
    'yolact_edge_vid_trainflow_config',
    'yolact_edge_youtubevis_config',
    'yolact_resnet50_config',
    'yolact_resnet152_config',
    'yolact_edge_resnet50_config',
    'yolact_edge_vid_resnet50_config',
    'yolact_edge_vid_trainflow_resnet50_config',
    'yolact_edge_youtubevis_resnet50_config',
]
config = model_configs[7]
print("config:", config)
# All available model datasets, depends on which weights
# you use. More info could be found in data/config.py.
datasets = [
    'coco2014_dataset',
    'coco2017_dataset',
    'coco2017_testdev_dataset',
    'flying_chairs_dataset',
    'youtube_vis_dataset',
]
dataset = datasets[1]
# Used tensorrt calibration
calib_images = "./data/calib_images"
# Override some default configuration
config_ovr = {
    'use_fast_nms': True,  # Does not work with regular nms
    'mask_proto_debug': False
}
model_inference = YOLACTEdgeInference(
    weights, config, dataset, calib_images, config_ovr)

img = None

# try:
#     with urllib.request.urlopen("http://images.cocodataset.org/val2017/000000439715.jpg") as f:
#         img = np.asarray(bytearray(f.read()), dtype="uint8")
#         img = cv2.imdecode(img, cv2.IMREAD_COLOR)
# except:
#     pass

image_path = Path("data/programmatic_inference_testing/swarmfarm")
image_path = Path("data/programmatic_inference_testing/coco")

for im_file in image_path.glob("**/*"):
    print("Image:", im_file.name)
    img = cv2.imread(str(im_file), cv2.IMREAD_COLOR)
    print(img.shape)

    if img is None:
        print("Couldn't retrieve image for benchmark...")
        exit(1)

    print("Benchmarking performance...")
    start = time.time()
    samples = 200
    exit()
    for i in range(samples):
        p = model_inference.predict(img, False)

    break

print(f"Average {1 / ( (time.time() - start) / samples )} FPS")
