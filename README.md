# Tracking and Material Counting

This repo is an implementation of our paper:

[Deep-Learning Based Construction Progress Tracking and Material Counting Using Existing Site Surveillance Camera](https://)

Main steps of the framework are:
- Detection
- Classification
- Couting
- Prediction

![main_figure1](https://user-images.githubusercontent.com/119409598/216583159-1b5eb1fe-525b-44cf-a351-941d647f44de.gif)

## Configuration
#### Detection
For object detection, we used [Yolov4 code](https://github.com/AlexeyAB/darknet) as a basic code and then improving loss function on it.
For running Yolov4 on own dataset, we modified **obj.name**, **obj.dat**, **yolov4.cfg**, and **src/**.
We put google colab code for this part on the Detection folder.
#### Classification
We finetuned [DenseNet-161](https://github.com/flyyufelix/DenseNet-Keras) in keras platform for our three classes of materials (0: *left*, 1: *frontal*, and 2: *right*). We put the finetuned denseNet for own data, implementing google colab on the Classificatin folder.
#### Counting
Site operatives counted by the results of the detection phase. However, the materials counted by MC module, which is based on the morphology operations, Hough Transform, and post processing algorithm.
#### Prediction
Finally, our framework could predict the rate of waste, installed, and imported materials and also the number of operative sites in a scene.
![results](https://user-images.githubusercontent.com/119409598/216585717-81a7a32e-8050-4155-b782-1755b1631eb7.gif)
