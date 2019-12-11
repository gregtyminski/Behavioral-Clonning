# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

[//]: # (Image References)

[ds-udacity-non-balanced]: ./examples/ds-udacity-non-balanced.png "Udacity dataset non-balanced"
[ds-udacity-balanced]: ./examples/ds-udacity-balanced.png "Udacity dataset balanced"
[ds-my-non-balanced]: ./examples/ds-my-non-balanced.png "Own dataset non-balanced"
[ds-my-balanced]: ./examples/ds-my-balanced.png "Own dataset balanced"

The point of this project to train deep neural network to clone driving behavior. Trained model outputs a steering angle to an autonomous vehicle based on the view from camera.

The simulator for steering a car around a track for data collection and for model verification is provided and available from [here](https://github.com/udacity/self-driving-car-sim).


# Environment preparation

This is actually the most difficult part of the project. The Simulator provided by Udacity is outdated and makes lots of troubles.

To be able to use simulator:

* download and install [Unity](https://unity.com/)
* install [git-lfs](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage)
* use `git lfs clone` command to download [simulator code](https://github.com/udacity/self-driving-car-sim).
* open a [Assets/Standard Assets/Cameras/Scripts/TargetFieldOfView.cs](https://github.com/udacity/self-driving-car-sim/blob/master/Assets/Standard%20Assets/Cameras/Scripts/TargetFieldOfView.cs#L62) file in any text editor and replace line 62:

```python
if (!((r is TrailRenderer) || (r is ParticleRenderer) || (r is ParticleSystemRenderer)))
```

to the file

```python
if (!((r is TrailRenderer) || (r is ParticleSystemRenderer)))
```

... or in another words, remove the `(r is ParticleRenderer) ||` part, as this causes troubles.
* next, using `pip` install `python-socketio` library (__don't install `socketio`__) and also `eventlet` library.<br />
This operation crashed my conda environment with Python 3.7 twice. After the project is passed, I'll have to reinstall conda from scratch.<br />
If you try to run `drive.py` script provided for this project it will fail with the message:<br />
`AttributeError: module 'importlib._bootstrap' has no attribute 'SourceFileLoader'`<br />
Next step corrects that temporarily.
* open the `%python_path%/python3.7/site-packages/pkg_resources.py` with any text editor and add following code in line 75:<br />

```python
importlib_bootstrap = None
```

# Dataset
I pretrained my network with the dataset provided by Udacity.<br />
Next I trained with my own data.

### Dataset provided by Udacity

#### Udacity's non-balanced dataset
The dataset from Udacity has following stats:<br />
![alt text][ds-udacity-non-balanced]
This clearly shows, that most of the time the car was driving straight:<br />

* values 0.0 are for images from center camera
* values 0.2 are for images from left camera
* values -0.2 are for images from right camera

Such data could lead to getting the model being able mostly to drive ahead.<br />

#### Udacity's balanced dataset

To overcome this balancing the dataset is necessary. This is implemented in `DataGenerator.balance_dataset()`. This method drop random 90% of entries in steering ranges `(-0.22, -0.18)`, `(0.18, 0.22)` and `(-0.01, 0.01)`.<br />
![alt text][ds-udacity-balanced]


### My dataset
To be able to train meaningful model I suggest to record at least 2-3 laps in each direction. While recording, steering is very important. You can control your car with keys _A,S,D,W_ or _UP, LEFT, DOWN, RIGHT, UP_ arrows. However you can control the car with the mouse as well. I suggest using the mouse. While using keys, steering is not smooth. In contradiction you can turn slightly, constantly and steady with a mouse. This will allow the model to train smoothly.<br />
My dataset contains:

#### My non-balanced dataset
Dataset recorded by me has following stats:<br />
![alt text][ds-my-non-balanced]

It's pretty balanced, but not fully. It still has peaks, which are not welcome.

#### My balanced dataset
Running same method `DataGenerator.balance_dataset()` to balance dataset gave following result:<br />
![alt text][ds-my-balanced]


# Build model

The model I have proposed after 

# Train

# Verify

# Potential improvements
 
---


We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

# Solution Writeup
---
The code uses Keras API in TensorFlow.

## Prepare model
In first step we are going to prepare model for training.

