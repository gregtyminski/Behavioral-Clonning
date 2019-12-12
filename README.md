# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

[//]: # (Image References)

[ds-udacity-non-balanced]: ./examples/ds-udacity-non-balanced.png "Udacity dataset non-balanced"
[ds-udacity-balanced]: ./examples/ds-udacity-balanced.png "Udacity dataset balanced"
[ds-my-non-balanced]: ./examples/ds-my-non-balanced.png "Own dataset non-balanced"
[ds-my-balanced]: ./examples/ds-my-balanced.png "Own dataset balanced"
[image-mean]: ./examples/mean_image.png "Mean of central images"
[image-mean-box]: ./examples/mean_image_box.png "Mean of central images with box"
[image-mean-all]: ./examples/mean_of_all.png "Mean of all images"
[training_job]: ./examples/training_job.png "Loss function over epochs"
[autonomous_mode_1]: ./examples/autonomous_mode_1.png "Autonomous mode"
[manual_mode]: ./examples/manual_mode.png "Manual mode"

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
To be able to record the dataset, [simulator](https://github.com/udacity/self-driving-car-sim) has been used, which looks like during dataset collection:<br />
![alt text][manual_mode]

#### My non-balanced dataset
Dataset recorded by me has following stats:<br />
![alt text][ds-my-non-balanced]

It's pretty balanced, but not fully. It still has peaks, which are not welcome.

#### My balanced dataset
Running same method `DataGenerator.balance_dataset()` to balance dataset gave following result:<br />
![alt text][ds-my-balanced]


# Build model

The model I have proposed after many trials:<br />

```python
def saturation_converter(x):
    hsv = tf.image.rgb_to_hsv(x)
    return hsv[: , : , : , :1: ]
    
model = Sequential([
    Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3), name='normalize'),
    Cropping2D(cropping=((65,25), (0,0)), name='cropping_65_25'),
    Lambda(saturation_converter, name='saturation'),
    BatchNormalization(),
    Conv2D(8,(3,3),padding='valid', activation='relu'),
    BatchNormalization(),
    Conv2D(16,(3,3),padding='valid', activation='relu'),
    BatchNormalization(),
    Conv2D(32,(3,3),padding='valid', activation='relu'),
    GlobalAveragePooling2D(),
    Dense(16, activation='linear', name='dense-64'),
    Dropout(0.5),
    Dense(1, activation='linear', name='dense-1')
])
```
... which is very similar to the one proposed by Udacity in the training (from nvidia). The difference is in the input normalization.<br />
I propose to do:<br />

* normalize the input to the range of value <-0.5, 0.5> - `Lambda` layer named `normalize`
* crop 65px from top, 25px from bottom - `Cropping2D` layer named `cropping_65_25`
* change the RGB color map to HSV and pick saturation channel only - `Lambda` layer named `saturation`
... and then 3 x convolutions layers with batch normalization with 0.5 dropout before last `Dense` layer to 1 value.<br />
<br />
This model learns to find edges of the road first and changes 3-channel to 1-channel image to speed up calculations.<br />
<br />
The model has final shape:<br />

```text
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
normalize (Lambda)           (None, 160, 320, 3)       0         
_________________________________________________________________
cropping_65_25 (Cropping2D)  (None, 70, 320, 3)        0         
_________________________________________________________________
saturation (Lambda)          (None, 70, 320, 1)        0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 70, 320, 1)        4         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 68, 318, 8)        80        
_________________________________________________________________
batch_normalization_4 (Batch (None, 68, 318, 8)        32        
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 66, 316, 16)       1168      
_________________________________________________________________
batch_normalization_5 (Batch (None, 66, 316, 16)       64        
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 64, 314, 32)       4640      
_________________________________________________________________
global_average_pooling2d_1 ( (None, 32)                0         
_________________________________________________________________
dense-64 (Dense)             (None, 16)                528       
_________________________________________________________________
dropout_1 (Dropout)          (None, 16)                0         
_________________________________________________________________
dense-1 (Dense)              (None, 1)                 17        
=================================================================
Total params: 6,533
Trainable params: 6,483
Non-trainable params: 50
_________________________________________________________________
```
... and the model consists of following files:

* [model.h5](model.h5) - trained model
* [model.json](model.json) - model description file

#### Cropping layer - why?
If you'd ask, why cropping 65px from top and 25px from bottom, please have a look on mean value of all central images:<br >
![alt text][image-mean]<br />
or mean value of all images (left, central, right):<br />
![alt text][image-mean-all]<br />
To get this mean of images just run `DataGenerator.mean_of_all()` code.
The picture below shows exactly, that the only meaningful information to make a decision if the vehicle should drive ahead or turn left or right is in the red bounding box. Top 65px and bottom 25px are useless:<br />
![alt text][image-mean-box]


# Train

The model has been trained with following parameters:<br />

* batch_size = 32
* learn_rate = 1e-04
* epochs = 120
* patience = 10  used for early stopping and reducing learning rate

As I have used also following callbacks:<br />

* `early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience+1)` to stop training when the model stops to learn
* `reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=patience, min_lr=1e-6, verbose=1)` to decrease learning rate if loss function stops to get smaller with each next epoch.

<br />
The model has trained with all 120 epochs and still could improve if we would keep training for more epochs. That could be seen on the plot of `loss` function over epochs:<br />
![alt text][training_job]<br />
The training took ~12 hours on CPU (Macbook Pro) with around 360s per epoch.<br />

```text
Epoch 1/120
573/573 [==============================] - 417s 728ms/step - loss: 0.0396
Epoch 2/120
573/573 [==============================] - 404s 705ms/step - loss: 0.0337
Epoch 3/120
573/573 [==============================] - 399s 696ms/step - loss: 0.0329
Epoch 4/120
573/573 [==============================] - 395s 690ms/step - loss: 0.0326
Epoch 5/120
573/573 [==============================] - 396s 692ms/step - loss: 0.0322
...
Epoch 117/120
573/573 [==============================] - 350s 612ms/step - loss: 0.0264
Epoch 118/120
573/573 [==============================] - 358s 624ms/step - loss: 0.0261
Epoch 119/120
573/573 [==============================] - 358s 625ms/step - loss: 0.0256
Epoch 120/120
573/573 [==============================] - 357s 623ms/step - loss: 0.0254
```


# Verify
To verify the model:<br />

* run `python drive.py image_folder/` script to feed simulator with model (over 4567 port) and keep the source images
* run the [simulator](https://github.com/udacity/self-driving-car-sim) in `autonomous mode` which looks like:<br />
![alt text][autonomous_mode_1]
* when the vehicle drives at least 1 full lap around the track, run `python movie.py image_folder/` code to generate the movie from all pictures collected in autonomous mode
<br />

As the output you get the movie showing your car driving over the track.<br />
Mine is available here in the `output_video.mp4` file:<br />
![output_video.mp4](output_video.mp4)

# Potential improvements

We could:

* make much smaller (thus faster) model by decreasing the size (scale down) of input image to even ~ 17x80 pixels and use just 1 convolution layer keeping usage of saturation channel of course. This choice would give us also a change to load entire dataset to the memmory and use `fit()` training method instead of `fit_generator`. This could be fairly enough.
* collect data also from differnt tracks. Dataset I've used contains only 1 track, while simulator has 2 tracks/.
* the model uses only a current video frame as an input for prediction and don't use information about current velocity or current steering (from previous video frame). For this reason the car does not steer smoothly but rather in quite nervous style. In real world this would lead to loosing traction control and to accident. The better way to run smoothly would be to use also as an input previous steering, velocity as well as previous video frame and train the RNN model.
