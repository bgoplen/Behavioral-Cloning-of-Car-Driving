import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

path='/opt/carnd_p3/data/'
csv_file=path+'driving_log.csv'

correction = 0.27 # this is a parameter to tune
batch_mode=0 # set to run in batch mode using a generator
steering_multiplier=4

lines=[]
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

lines=lines[1:]

def generator(samples, batch_size=128):
    import sklearn
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center=float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                img_center = np.asarray(mpimg.imread(path+batch_sample[0].lstrip()))
                img_left = np.asarray(mpimg.imread(path+batch_sample[1].lstrip()))
                img_right = np.asarray(mpimg.imread(path+batch_sample[2].lstrip()))

                #flip images for data augmentation
                flip_img_center=cv2.flip(img_center,1)
                flip_img_left=cv2.flip(img_left,1)
                flip_img_right=cv2.flip(img_right,1)
                # add images and angles to data set
                images.extend([img_center,img_left,img_right,flip_img_center,flip_img_left,flip_img_right])
                angles.extend([steering_center,steering_left,steering_right,-steering_center,-steering_left,-steering_right])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)*steering_multiplier
            yield sklearn.utils.shuffle(X_train, y_train)

car_images=[]
steering_angles=[]

if  batch_mode == 1:
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=100)
    validation_generator = generator(validation_samples, batch_size=100)
else:
    for line in lines:
        steering_center = float(line[3])

        # create adjusted steering measurements for the side camera images
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        img_center = np.asarray(mpimg.imread(path+line[0].lstrip()))
        img_left = np.asarray(mpimg.imread(path+line[1].lstrip()))
        img_right = np.asarray(mpimg.imread(path+line[2].lstrip()))

        # create flipped images
        flip_img_center=cv2.flip(img_center,1)
        flip_img_left=cv2.flip(img_left,1)
        flip_img_right=cv2.flip(img_right,1)

        # add images and angles to data set
        car_images.extend([img_center,img_left,img_right,flip_img_center,flip_img_left,flip_img_right])
        steering_angles.extend([steering_center,steering_left,steering_right,-steering_center,-steering_left,-steering_right])

    X_train=np.array(car_images)
    y_train=np.array(steering_angles)*steering_multiplier

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout,BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

def convert_norm(x):
    import tensorflow as tf
    return tf.image.rgb_to_hsv(x)/255.0-0.5
#    return tf.image.rgb_to_yuv(x)/255.0-0.5

model=Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(160,320,3)))
#model.add(Lambda(lambda x: x/127.5-1.0,input_shape=(160,320,3))) #similar results with above
#model.add(Lambda(convert_norm,input_shape=(160,320,3))) #didn't seem to help
#model.add(Cropping2D(cropping=((63,23),(0,0))))
model.add(Cropping2D(cropping=((60,20),(0,0))))
#model.add(Cropping2D(cropping=((55,15),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
#model.add(Dropout(.5))
#model.add(BatchNormalization())
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
#model.add(Dropout(.45))
model.add(BatchNormalization())
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
#model.add(Dropout(.4))
#model.add(BatchNormalization())
model.add(Convolution2D(64,3,3,activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
#model.add(Dropout(.35))
model.add(BatchNormalization())
model.add(Convolution2D(64,3,3,activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
#model.add(Dropout(.3))
#model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1164))
#model.add(Dropout(.25))
model.add(Dropout(.5))
model.add(Dense(100))
#model.add(Dropout(.2))
#model.add(Dense(50))
#model.add(Dropout(.15))
model.add(Dense(10))
#model.add(Dropout(.1))
model.add(Dense(1))

model.compile(loss='mse',optimizer=Adam(lr=3e-4))
#model.compile(loss='mse',optimizer=Adam(lr=1e-4))
if  batch_mode == 1:
    history_object = model.fit_generator(train_generator,steps_per_epoch=len(train_samples),validation_data=validation_generator,validation_steps=len(validation_samples),epochs=5,verbose = 1)
else:
    history_object=model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=10,verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())
model.save('model.h5')

print(model.summary())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
