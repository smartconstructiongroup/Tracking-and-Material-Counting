# Tracking-and-Material-Counting

This repo is an implementation of our paper:

[Deep-Learning Based Construction Progress Tracking and Material Counting Using Existing Site Surveillance Camera](https://)

Main steps of the framework are:
- Detection
- Classification
- Couting
- Prediction

## Configuration
#### Detection
For object detection, we used [Yolov4 code](https://github.com/AlexeyAB/darknet) as a basic code and then improving loss function on it.
For running Yolov4 on own dataset, we modified **obj.name**, **obj.dat**, **yolov4.cfg**, and **src/**.
