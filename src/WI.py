#!/usr/bin/env python
# coding: utf-8

# In[2]:


#this is ours
from __future__ import division
import numpy as np
import os
import glob

from random import *
from PIL import Image
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop

from collections import Counter 


# In[15]:


from google.colab import files
files.upload()


# In[ ]:


get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')


# In[18]:


get_ipython().system('kaggle datasets download -d spbourne95/iamhandwritingtop50simplified')


# In[ ]:


get_ipython().system('chmod 600 /root/.kaggle/kaggle.json')


# In[19]:


from zipfile import ZipFile 
file_name="iamhandwritingtop50simplified.zip"
with ZipFile(file_name,"r") as zip :
  zip.extractall()
  print("done")


# In[ ]:


from zipfile import ZipFile 
file_name="data_subset.zip"
with ZipFile(file_name,"r") as zip :
  zip.extractall()
  print("done")





# Create array of file names and corresponding target writer names

tmp = []
target_list = []
path_to_files = os.path.join('new_dataset/data_subset', '*')
for filename in sorted(glob.glob(path_to_files)):
    #print(filename)
    tmp.append(filename)
    image_name = filename.split('/')[-1]
    #print(image_name)
    #file, ext = os.path.splitext(image_name)
    #print(file,ext)
   # print(file)
    parts = image_name.split('-')
    
    #print(parts)
    writer= parts[0] 
    print (writer)
    #print(form)
    #i=0
    #for key in d:
        #if key == form:
           # print(d[form])
    target_list.append(writer)
           # print(target_list[i])
            #i=i+1
#print(target_list)            
img_files = np.asarray(tmp)

img_targets = np.asarray(target_list)
#print(img_targets[50])
#print(img_files[1])
#print(img_targets.shape)


# In[4]:



# Visualizing the data
for filename in img_files[:3]:
                          
    img = mpimg.imread(filename)
    plt.figure(figsize=(10,10))
    plt.imshow(img,cmap="gray")
    plt.show()


# In[5]:


# Label Encode writer names 


encoder = LabelEncoder()
encoder.fit(img_targets)
#print(img_targets)
encoded_Y = encoder.transform(img_targets)
#print(encoded_Yåpp)

print(img_files[:3], img_targets[:], encoded_Y[:])






from sklearn.model_selection import train_test_split
#print(len(train_files))
#print(len(rem_files))
train_files, validation_files, train_targets, validation_targets = train_test_split(
        img_files, encoded_Y, train_size=0.8, random_state=52, shuffle= True)

#validation_files, test_files, validation_targets, test_targets = train_test_split(
        #rem_files, rem_targets, train_size=0.5, random_state=22, shuffle=True)

#print(train_files.shape, validation_files.shape, test_files.shape)
#print(train_targets.shape, validation_targets.shape, test_targets.shape)
#print(len(train_files))
print(len(train_files))
print(len(validation_files))


# In[7]:


# Generator function for generating random crops from each sentence

# # Now create generators for randomly cropping 113x113 patches from these images

batch_size = 16
num_classes = 50

# Start with train generator shared in the class and add image augmentations
def generate_data(samples, target_files,  batch_size=batch_size, factor = 0.1 ):
    num_samples = len(samples)
    from sklearn.utils import shuffle
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_targets = target_files[offset:offset+batch_size]
            print(batch_samples)  
            images = []
            targets = []
            for i in range(len(batch_samples)):
                batch_sample = batch_samples[i]
                batch_target = batch_targets[i]
                im = Image.open(batch_sample)
                cur_width = im.size[0]
                cur_height = im.size[1]

                #print('middle')
                height_fac = 113 / cur_height

                new_width = int(cur_width * height_fac)
                size = new_width, 113

                imresize = im.resize((size), Image.ANTIALIAS)  # Resize so height = 113 while keeping aspect ratio
                now_width = imresize.size[0]
                now_height = imresize.size[1]
                # Generate crops of size 113x113 from this resized image and keep random 10% of crops

                
                avail_x_points = list(range(0, now_width - 113 ))# total x start points are from 0 to width -113

                # Pick random x%
                pick_num = int(len(avail_x_points)*factor)

                # Now pick
                random_startx = sample(avail_x_points,  pick_num)

                for start in random_startx:
                    imcrop = imresize.crop((start, 0, start+113, 113))
                    images.append(np.asarray(imcrop))
                    targets.append(batch_target)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(targets)

            #reshape X_train for feeding in later
            X_train = X_train.reshape(X_train.shape[0], 113, 113, 1)
            #convert to float and normalize
            X_train = X_train.astype('float32')
            X_train /= 255

            #One hot encode y
            y_train = to_categorical(y_train, num_classes)

            yield shuffle(X_train, y_train)


# In[8]:


# Generate data for training and validation
train_generator = generate_data(train_files, train_targets, batch_size, factor = 0.3)
validation_generator = generate_data(validation_files, validation_targets, batch_size=batch_size, factor = 0.3)
#test_generator = generate_data(test_files, test_targets, batch_size=batch_size, factor = 0.1)


# In[9]:


# Build a neural network in Keras

# Function to resize image to 56x56
def resize_image(image):
    import tensorflow as tf
    return tf.image.resize_images(image,[56,56])

# Function to resize image to 64x64
row, col, ch = 113, 113, 1

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(row, col, ch)))

# Resise data within the neural network
model.add(Lambda(resize_image))  #resize images to allow for easy computation

# CNN model - Building the model suggested in paper

model.add(Convolution2D(filters= 32, kernel_size =(5,5), strides= (2,2), padding='same', name='conv1')) #96
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool1'))

model.add(Convolution2D(filters= 64, kernel_size =(3,3), strides= (1,1), padding='same', name='conv2'))  #256
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool2'))

model.add(Convolution2D(filters= 128, kernel_size =(3,3), strides= (1,1), padding='same', name='conv3'))  #256
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool3'))


model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(512, name='dense1'))  #1024
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256, name='dense2'))  #1024
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes,name='output'))
model.add(Activation('softmax'))  #softmax since output is within 50 classes

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
print(model.summary())






nb_epoch = 2

samples_per_epoch = 3268
nb_samples = 842


#save every model using Keras checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint
filepath="sample_data/check-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath= filepath, verbose=1, save_best_only=False)
callbacks_list = [checkpoint]

# Model fit generator
#history_object = model.fit_generator(generator=train_generator, steps_per_epoch= samples_per_epoch,
                                     # validation_data=validation_generator,
                                      #validation_steps=nb_samples, epochs=nb_epoch, verbose=1, callbacks=callbacks_list)
history_objec=model.fit_generator(train_generator,steps_per_epoch=3268,epochs=5,verbose=1,callbacks=callbacks_list,validation_data=validation_generator,validation_steps=842)





# Load save model and use for prediction on test set
model.load_weights('check-08-0.3848.hdf5')
#scores = model.evaluate_generator(train_generator,842) 
#print("Accuracy = ", scores[1])




#loading in Of test Data
def interfacer(value)
 image = value
 images=[]
#view_img=np.asarray(test_files)

 im = Image.open(value)
     #print(filename)
 cur_width = im.size[0]   
 cur_height = im.size[1]
    
 height_fac = 113 / cur_height

 new_width = int(cur_width * height_fac)
 size = new_width, 113

 imresize = im.resize((size), Image.ANTIALIAS)  # Resize so height = 113 while keeping aspect ratio
 now_width = imresize.size[0]
 now_height = imresize.size[1]
     # Generate crops of size 113x113 from this resized image and keep random 10% of crops

 avail_x_points = list(range(0, now_width - 113 ))# total x start points are from 0 to width -113

## Pick random x%
 factor = 0.1
 pick_num = int(len(avail_x_points)*factor)
    
 random_startx = sample(avail_x_points,  pick_num)
     
 print(random_startx)
 count=0
 for start in random_startx:
    ++count
    imcrop = imresize.crop((start, 0, start+113, 113))   
    imcrop1 = np.asarray(imcrop)
    images.append(imcrop1)  
         #imcrop1=np.asarray(images)
    cropped_image=imcrop1.squeeze()  
    plt.figure(figsize=(2,2))
    plt.imshow(cropped_image, cmap ='gray')
    plt.show()
 X_test = np.asarray(images)
     
 X_test = X_test.reshape(X_test.shape[0], 113, 113, 1)
     #convert to float and normalize
 X_test = X_test.astype('float32')
 X_test /= 255
 shuffle(X_test)
 print(len(X_test))


# print(X_test.shape)


# In[14]:



# Play with results from model 
 predictions = model.predict(X_test, verbose =1)

 predicted_writer = []
 for pred in predictions:
    predicted_writer.append(np.argmax(pred))

 i=0

  
 def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 
    
 print("the writer is ", most_frequent(predicted_writer))
return most_frequent(predicted_writer) 






