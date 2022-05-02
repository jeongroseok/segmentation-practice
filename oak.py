#!/usr/bin/env python3

import argparse
import random
import time
from datetime import datetime, timedelta

import cv2
import depthai as dai
import matplotlib.pyplot as plt
import numpy as np

nn_shape = 256
nn_path = (
    fr"C:\Users\jeong\Desktop\machine-learning\segmentation-practice\test_256_13.blob"
)
TARGET_SHAPE = (400, 400)


class_colors = [
    (random.random() * 256, random.random() * 256, random.random() * 256)
    for i in range(21)
]
class_colors = np.asarray(class_colors, dtype=np.uint8)


def decode_deeplabv3p(output_tensor):
    output = output_tensor.reshape(nn_shape, nn_shape)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors


class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


class HostSync:
    def __init__(self):
        self.arrays = {}

    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        # Add msg to array
        self.arrays[name].append({"msg": msg})
        # Try finding synced msgs
        ts = msg.getTimestamp()
        synced = {}
        for name, arr in self.arrays.items():
            for i, obj in enumerate(arr):
                time_diff = abs(obj["msg"].getTimestamp() - ts)
                # 20ms since we add rgb/depth frames at 30FPS => 33ms. If
                # time difference is below 20ms, it's considered as synced
                if time_diff < timedelta(milliseconds=33):
                    synced[name] = obj["msg"]
                    # print(f"{name}: {i}/{len(arr)}")
                    break
        # If there are 3 (all) synced msgs, remove all old msgs
        # and return synced msgs
        if len(synced) == 2:  # color, depth, nn

            def remove(t1, t2):
                return timedelta(milliseconds=500) < abs(t1 - t2)

            # Remove old msgs
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if remove(obj["msg"].getTimestamp(), ts):
                        arr.remove(obj)
                    else:
                        break
            return synced
        return False


def crop_to_square(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    delta = int((width - height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta : width - delta]


# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# Color cam: 1920x1080
# Mono cam: 640x400
cam.setIspScale(2, 3)  # To match 400P mono cameras
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.initialControl.setManualFocus(130)

# For deeplabv3
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(nn_shape, nn_shape)
cam.setInterleaved(False)

# NN output linked to XLinkOut
isp_xout = pipeline.create(dai.node.XLinkOut)
isp_xout.setStreamName("cam")
cam.isp.link(isp_xout.input)

# Define a neural network that will make predictions based on the source frames
seg_nn = pipeline.create(dai.node.NeuralNetwork)
seg_nn.setBlobPath(nn_path)
seg_nn.input.setBlocking(False)
seg_nn.setNumInferenceThreads(2)
cam.preview.link(seg_nn.input)

# NN output linked to XLinkOut
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
seg_nn.out.link(xout_nn.input)


# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline.getOpenVINOVersion()) as device:
    cams = device.getConnectedCameras()
    depth_enabled = (
        dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    )
    if not depth_enabled:
        raise RuntimeError(
            "Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(
                cams
            )
        )
    device.startPipeline(pipeline)
    # Output queues will be used to get the outputs from the device
    q_color = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    fps = FPSHandler()
    sync = HostSync()
    disp_frame = None

    frame = None
    depth = None
    depth_weighted = None
    frames = {}

    while True:
        msgs = False
        if q_color.has():
            msgs = msgs or sync.add_msg("color", q_color.get())
        if q_nn.has():
            msgs = msgs or sync.add_msg("nn", q_nn.get())

        if msgs:
            fps.next_iter()
            # get layer1 data
            layer1 = msgs["nn"].getFirstLayerFp16()
            # reshape to numpy array
            lay1 = np.asarray(layer1).reshape((21, nn_shape, nn_shape)).argmax(0)
            lay1 = lay1.astype(np.int32)
            output_colors = decode_deeplabv3p(lay1)

            # To match depth frames
            output_colors = cv2.resize(output_colors, TARGET_SHAPE)

            frame = msgs["color"].getCvFrame()
            frame = crop_to_square(frame)
            frame = cv2.resize(frame, TARGET_SHAPE)
            frames["frame"] = frame
            frame = cv2.addWeighted(frame, 1, output_colors, 0.5, 0)
            cv2.putText(
                frame,
                "Fps: {:.2f}".format(fps.fps()),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.4,
                color=(255, 255, 255),
            )
            frames["colored_frame"] = frame
            frames["output_colors"] = output_colors

        if len(frames) == 3:
            show = np.concatenate(
                (frames["frame"], frames["output_colors"], frames["colored_frame"]),
                axis=1,
            )
            cv2.imshow("Combined frame", show)

        if cv2.waitKey(1) == ord("q"):
            break
