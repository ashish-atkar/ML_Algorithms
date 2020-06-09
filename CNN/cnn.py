"""
This is very basic CNN architecture made just to learn the basic important steps in the CNN:
    1: Convolution Layer (Applying filters(feature) and getting various feature maps)
    2: Pooling (Basically reducing the size by either MaxPooling or MinPooling)
    3: Flattening (Converting into vector row by row)
    4: Full Connection (Adding further layers as done in ANN)
    
Dataset is structured and divided into training and testing folder(extract zip and know more)
    
Note : Here the purpose was to learn basics important layers in CNN soo we will not achieve 
       great accuracy but we can certainly increase accuracy by adding more layers which can
       be found in other code in parent folder.

"""

#Part 1:Building the CNN

#Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense 

#Initializing the CNN
classifier = Sequential()

#Step 1: Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=[64, 64, 3], activation="relu"))

#Step 2:Pooling
classifier.add(MaxPool2D(pool_size= (2,2), strides=2, padding='valid' ))

#Step 3:Flattening
classifier.add(Flatten())

#Step 4: Full Connection
classifier.add(Dense(units= 128, activation="relu"))
classifier.add(Dense(units= 1, activation="sigmoid"))

#Compiling the CNN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics=['accuracy']) 


#Part 2: Fitting the CNN  to the images
from keras.preprocessing.image import ImageDataGenerator

# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = train_datagen.flow_from_directory('dataset/test_set',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Training the CNN on the Training set and evaluating it on the Test set
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
                                   