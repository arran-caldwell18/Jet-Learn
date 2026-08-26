#------------------
#Imports
#------------------
import tensorflow as tf
from keras.layers import Flatten , Dense , Dropout , Conv2D , MaxPool2D
from keras.applications import VGG16

import os
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from PIL import Image

import kagglehub as kh


#-----------------
#Consts
#-----------------
DATA_PATH = kh.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

NUM_CLASSES = 43

INPUT_SHAPE = (30 , 30 , 3)

#-----------------
#Pre Processing
#----------------

#Local Use Variables
all_img_path = os.path.join(DATA_PATH , "Train")
img_data    = []
labels      = []

#Looping over every image
for i in range(NUM_CLASSES):

    #Creating Path For Each Label
    image_path = os.path.join(all_img_path , str(i))

    for j in os.listdir(image_path):
        #Get File Path Using os.path.join (instead of adding strings together)
        file_path = os.path.join(image_path, j)

        opened = Image.open(file_path)

        #Resizing Image (For Model To Accept(Model Only takes ? X 30 X 30))
        opened = opened.resize((30 , 30))

        #Convert to NP array
        opened = np.array(opened)
        img_data.append(opened)
        labels.append(i)

#Converting to NP array
img_data    = np.array(img_data)
labels      = np.array(labels)




#---------------------
#Train Test Split
#---------------------

X_train , X_test , y_train , y_test  =  train_test_split(img_data , labels , test_size = 0.3 , random_state = 42 )


#---------------------
#Normalize (train) Data
#---------------------
X_test  =   X_test  / 255
X_train =   X_train / 255


#--------------------
#To Categorical
#--------------------
y_train = to_categorical(y_train , NUM_CLASSES)
y_test  = to_categorical(y_test  , NUM_CLASSES)



#------------------------
#Create Model
#-------------------------
cnn = Sequential()

cnn.add(Conv2D(filters = 32 , kernel_size = (5 , 5) , activation = 'relu' , input_shape = INPUT_SHAPE))
cnn.add(Conv2D(filters = 32 , kernel_size = (5 , 5) , activation = 'relu' ))

cnn.add(MaxPool2D(pool_size = (2 , 2)))
cnn.add(Dropout(rate = 0.25))

cnn.add(Conv2D(filters = 64 , kernel_size = (5 , 5) , activation = 'relu' ))
cnn.add(Conv2D(filters = 64 , kernel_size = (5 , 5) , activation = 'relu' ))

cnn.add(MaxPool2D(pool_size = (2 , 2)))
cnn.add(Dropout(rate = 0.25))

cnn.add(Flatten())

cnn.add(Dense(256 , activation = 'relu'))
cnn.add(Dropout(rate = 0.25))

cnn.add(Dense(NUM_CLASSES , activation = 'softmax'))


#Compile
cnn.compile( optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['Accuracy'])

#Display Model Architecture
cnn.summary()

#Run Model
history = cnn.fit(X_train , y_train , epochs = 10 , validation_data = (X_test , y_test) , batch_size = 64)


#Evaluate
loss , acc = cnn.evaluate(X_train , y_test)

print(f"Loss : {loss} | Acc : {acc}!")

#Save Model
cnn.save() #Save To Current Path i assume if left empty