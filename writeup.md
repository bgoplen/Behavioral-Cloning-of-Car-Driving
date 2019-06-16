# **Behavioral Cloning Writeup** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py contains the script to create and train the model
* model.h5 contains the trained convolution neural network 
* writeup.md/writeup.html summarizes the results
* video.mp4 is the video of the test drive around track 1 using model.h5
* drive.py for driving the car in autonomous mode but was not modified


#### 2. Submission includes functional code
The command "python drive.py model.h5" can be used with the Udacity simulator to drive the car autonomously around the track.  video.mp4 shows the video of the resulting autonomous drive around track 1.


#### 3. Submission code is usable and readable

model.py is executed using "python model.py" to train and save the convolution neural network as model.h5. "batch_mode" in line 11 can be set to run in batch mode using a Python generator.  I was able to create the model without using the generator so I did not use that mode to create model.h5.  A lot more memory was used without the generator, but the GPU was able to handle it.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I modified the convolution neural network from NVIDIA's "End to End Learning for Self-Driving Cars" paper for use as the model as shown in model.py lines 103-140:

_________________________________________________________________
Layer (type)                 Output Shape              Param #

=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636
_________________________________________________________________
batch_normalization_1 (Batch (None, 17, 77, 36)        144
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712
_________________________________________________________________
batch_normalization_2 (Batch (None, 5, 35, 64)         256
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 6336)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              7376268
_________________________________________________________________
dropout_1 (Dropout)          (None, 1164)              0
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500
_________________________________________________________________
dense_3 (Dense)              (None, 10)                1010
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11

=================================================================
Total params: 7,625,537
Trainable params: 7,625,337
Non-trainable params: 200

ReLU activation function was used throughout the model, and normalization of the image data used a Keras lambda layer in line 104.


#### 2. Attempts to reduce overfitting in the model

I tried various schemes to reduce overfitting.  Initially, I tried adding a dropout layer with successively decreasing rates after each step, but it didn't seem to help.  I tried max pooling after each of the convolution step but with no discernible improvement.  After reading this paper: https://towardsdatascience.com/dont-use-dropout-in-convolutional-networks-81486c823c16, I tried various combinations of batch normalization steps between the convolution layers and settled on using just batch normalization after every other convolution layer.  I also removed all the dropout layers except after the dense 1164 layer, and I removed the dense 50 layer.  In line 147, a validation_split of 20% was used to ensure the model is trained and validated on different data sets.  Below shows the training and validation loss outputted from model.py

Epoch 1/10
8572/38572 [==============================] - 83s - loss: 3.1521 - val_loss: 0.4817
Epoch 2/10
38572/38572 [==============================] - 76s - loss: 0.3835 - val_loss: 0.4630
Epoch 3/10
38572/38572 [==============================] - 76s - loss: 0.3352 - val_loss: 0.4520
Epoch 4/10
38572/38572 [==============================] - 76s - loss: 0.3131 - val_loss: 0.4385
Epoch 5/10
38572/38572 [==============================] - 77s - loss: 0.2726 - val_loss: 0.4373
Epoch 6/10
38572/38572 [==============================] - 76s - loss: 0.2573 - val_loss: 0.4634
Epoch 7/10
38572/38572 [==============================] - 76s - loss: 0.2392 - val_loss: 0.3600
Epoch 8/10
38572/38572 [==============================] - 75s - loss: 0.2039 - val_loss: 0.2398
Epoch 9/10
38572/38572 [==============================] - 75s - loss: 0.1755 - val_loss: 0.2249
Epoch 10/10
38572/38572 [==============================] - 76s - loss: 0.1560 - val_loss: 0.2165

Both losses are significantly reduced by the end of training.  The model was also tested with the simulator to verify that the car would stay on the track.


#### 3. Model parameter tuning

The Adam optimizer was chosen in model.py line 142, but I did tune the initial learning rate used by it.  I tried 1e2, 1e3, 5e4, 4e4, 3e4, 2e4, 1e4, and 5e5.  I found that 3e4 worked best.  Too small of a learning rate, and the loss would decrease too slowly.  Too high of a learning rate, and the final loss value would not be as low.


#### 4. Appropriate training data

I augmented the supplied training data using not only the center images but also the right and left camera images. For the right and left images, a correction value (line 10) was used to adjust the steering values.  Steering corrections of 0.1, 0.15, 0.2, 0.225, 0.24, 0.25, 0.26, 0.27, 0.3, and 0.5 were tested. Low correction values show less turning speed, and higher correction values tended to create more oscillation, but the results didn't seem to be too sensitive on the value chosen.  A steering correction of 0.27 was ultimately used to add and subtract to the steering value for the left and right cameras respectively.  In addition, the data was further augmented by flipping the center, right, and left images and multiplying their steering values by -1.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The convolution neural network from NVIDIA's "End to End Learning for Self-Driving Cars" paper was used as the starting point in the developing the model architecture.  Initially, the model drove the car perfectly, but the validation loss showed no improvement during training so it was clearly just overfitting.  Different combinations of drop out, max pooling, and batch normalization layers were tried.  Adding too many of these layers made it difficult to train, so I setted on two batch normalization layers in the convolution layers and one dropout layer in the dense layers.  The dense 50 layer was removed from the model as its removal seemed to improve results.

For preprocess, the top 60 rows of pixels were removed so that the sky and background doesn't interfer with the results.  The bottom 20 rows of pixels were removed to remove the hood of the car from the image.  I tried both x/127.5-1.0 and x/255.0-0.5 formulas for normalizing the image data, both gave similar results as did converting from RGB to HSV.  The tensorflow tf.image.rgb_to_yuv function wasn't available with the version of tensorflow installed on the GPU so I didn't try the YUV image format.


#### 2. Final Model Architecture

The final model architecture is shown below and coded in model.py line 103-140:

Normalization: x/255.0-0.5, input_shape=(160,320,3)
Cropping2D: cropping={(60,20),(0,0)}
Convolution2D: filter=24, kernel=5x5, strides=2x2, activation=RELU
Convolution2D: filter=36, kernel=5x5, strides=2x2, activation=RELU
Batch normalization
Convolution2D: filter=48, kernel=5x5, strides=2x2, activation=RELU
Convolution2D: filter=64, kernel=3x3, strides=1x1, activation=RELU
Batch normalization
Convolution2D: filter=64, kernel=3x3, strides=1x1, activation=RELU
Dense: neurons=1164, activation=RELU
Drop out: rate=0.5
Dense: neurons=100, activation=RELU
Dense: neurons=10, activation=RELU
Dense: neurons=1, activation=RELU

RGB images were used as the input with the top 60 (sky and background) and bottom 20 (hood of car) rows of pixels removed.

#### 3. Creation of the Training Set & Training Process

Using the data in /opt/carnd_p3/data, image and steering data was extracted and augmented by using the right and left images in addition to the center camera image.  These three sets were duplicated and flipped to further augment the data.  There were 8036 lines of data listed in driving_log.csv so this resulted in 8036*6=48216 images with steering angles.  

Examples of the images are located here:
 /opt/carnd_p3/data/IMG/center_*.jpg
 /opt/carnd_p3/data/IMG/left_*.jpg
 /opt/carnd_p3/data/IMG/right_*.jpg

./examples/center_2016_12_01_13_31_13_381.jpg
./examples/left_2016_12_01_13_31_13_381.jpg
./examples/right_2016_12_01_13_31_13_381.jpg

The data was radomly shuffled with 20% extracted for use as the validation set.  I constantly had problems with the car going off the road or driving on the line.  Almost always, it would turn when it got close to the edge of the road in these instances but not strong enough.  To combat this, I created a stronger turning response by multiplying the steering angle with a steering multiplier: lines 12 54, & 90.  I ended up using a steering multiplier of 4, but smaller values were also effective.  Given the same basic architecture, the appropriate number of epochs depends on the learning rate and amount of overfitting preventation layers, but I set it to 10 because 5 seemed to be too short and 15 too long.  In my first attempts, the mean squared error for the validation set did not decrease that much across epochs so I had to add drop out and batch normalization layers to prevent overfitting.


### Simulation
The simulator was run in autonomous mode along with the command "python drive.py model.h5 run1".  The video was created with "python video.py run1" and "mv run1.mp4 video.mp4"

In my simulation, no tire left the drivable portion of the track surface. The car did not drive outside the lines or off the edge of the road.  The steering was a little jerky/jittery, but it was not a requirement of this project to make it smooth.  This can likely be solved by averaging steering values together in some way.  I found that models with smoother steering tended to be accompanied by a greater likelihood of the car being less responsive to the edges of the road.


