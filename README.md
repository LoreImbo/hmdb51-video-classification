# VIDEO CLASSIFICATION ON HMDB51 DATASET: HUMAN ACTIVITY RECOGNITION



GOAL: Develope a classification algorithm (LRCN) that exploits the main deep learning techniques in order to predict and recognize the simplest human actions.
 
 Check out the [report](https://github.com/LoreImbo/hmdb51-video-classification/blob/513476b621053613ae8424d3c72f5e7decf54620/Report/DLproject.pdf)

## DATASET
The selected dataset is named 'HMDB - Human Emotion DB'. Each observation corresponds to one video, for a total of 6849 clips. Each video has associated one of 51 possible classes, each of which identifies a specific human behavior. Moreover the classes of actions can be grouped into: 
1) general facial actions such as smiling or laughing; 
2) facial actions with object manipulation such as smoking; 
3) general body movements such as running; 
4) body movements withi object interaction such as golfing; 
5) body movements for human interaction such as kissing. 

Due to computational problems we have chosen only 19 classes (general body movements) on which to train the human activity recognition algorithm.

[HMDB51 dataset site](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)

## LRCN APPROACH

LRCN is a class of architectures which combines Convolutional layers and Long Short-Term Memory (LSTM).

BASIC LRCN

- Convolutional2D Layer
- LSTM Layer
- Dense Layer [fully - connected]

ADVANCED LRCN

- 3 Convolutional2D Layers
- LSTM Layer
- Dense Layer [fully - connected]

## MOVENET APPROACH

MoveNet is an ultra fast and accurate model that detects 17 keypoints of a body. The model is offered with two variants, known as Lightning and Thunder. Lightning is intended for latency-critical applications, while Thunder is intended for applications that require high accuracy.

MoveNet is a bottom-up estimation model, using heatmaps to accurately localize human keypoints. The architecture consists of two components: a feature extractor and a set of prediction heads.

The feature extractor in MoveNet is MobileNetV2 with an attached feature pyramid network (FPN), which allows for a high resolution, semantically rich feature map output. There are four prediction heads attached to the feature extractor, responsible for densely predicting a:
- Person center heatmap: predicts the geometric center of person instances;
- Keypoint regression field: predicts full set of keypoints for a person, used for grouping keypoints into instances;
- Person keypoint heatmap: predicts the location of all keypoints, independent of person instances;
- 2D per-keypoint offset field: predicts local offsets from each output feature map pixel to the precise sub-pixel location of each keypoint.

MOVENET ARCHITECTURE

<img width="583" alt="Schermata 2023-03-16 alle 17 00 46" src="https://user-images.githubusercontent.com/96497366/225679688-abdbc201-8b36-40f4-8ab9-db7262ed827d.png">


RESULTS

| Network       | Valid Accuracy |
| ------------- | -------------- |
| Basic LRCN    |       34%      |
| Adavnced LRCN |       41%      |
| MoveNet       |       70%      |  



## REFERENCES

[1] [Deep Learning Models for Human Activity Recognition](https://machinelearningmastery.com/deep-learning-models-for-human-activity-recognition/)

[2] [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/abs/1411.4389?source=post_pagel)

[3] [Long-term Recurrent Convolutional Network for Video Regression](https://towardsdatascience.com/long-term-recurrent-convolutional-network-for-video-regression-12138f8b4713)

[4] [Long-term Recurrent Convolutional Networks](https://jeffdonahue.com/lrcn/)

[5] [Next-Generation Pose Detection with MoveNet](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)
