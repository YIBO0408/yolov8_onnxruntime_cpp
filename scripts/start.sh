#!/bin/bash

docker run \
--privileged \
--gpus all \
--rm \
-it \
-v /home/yibo/yolov8_cpp/YOLOv8-ONNXRuntime-CPP:/home/yibo/yolov8_cpp/YOLOv8-ONNXRuntime-CPP \
hub.micro-i.com.cn:9443/dad/tritonserver_dev:23.09-py3 bash
