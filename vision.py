#!/usr/bin/env python3

from pathlib import Path
# import cv2
import depthai as dai
import numpy as np
import time
import argparse

def send_velocity_command(yaw_velocity):
    ### TODO: Add your code here to send the velocity command to the pupper
    ### Write the velocity command to the file "velocity_command"
    pass

nnPathDefault = str((Path(__file__).parent / Path('mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-s', '--sync', action="store_true", help="Sync RGB output with NN output", default=False)
args = parser.parse_args()

if not Path(nnPathDefault).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")
nnNetworkOut.setStreamName("nnNetwork");

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setFps(40)
# Define a neural network that will make predictions based on the source frames
nn.setConfidenceThreshold(0.1)
nn.setBlobPath(args.nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Linking
if args.sync:
    nn.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

camRgb.preview.link(nn.input)
nn.out.link(nnOut.input)
nn.outNetwork.link(nnNetworkOut.input);

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qNN = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    printOutputLayersOnce = True

    yaw_velocity = 0.0

    while True:
        if args.sync:
            inRgb = qRgb.get()
            inDet = qDet.get()
            inNN = qNN.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()
            inNN = qNN.tryGet()

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        ### TODO: Add your code here to detect a person and run a visual servoing controller
        ### Steps:
        ### 1. Detect a person
        ### 2. Compute the centroid of the bounding box
        ### 3. Compute the error between the centroid and the center of the image
        ### 4. Compute the velocity command using a proportional controller
        ### 5. Send the velocity command to the pupper