from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from keras.utils import to_categorical, normalize
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn import datasets


def convolutional():
    digits = datasets.load_digits()

    X = digits.data/digits.data.max()
    y = digits.target
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)

    XTrain = XTrain.reshape(1437, 8, 8, 1) #shape of XTrain, 8, 8, 1
    XTrain = normalize(XTrain, axis = 1)
    XTest = XTest.reshape(360, 8, 8, 1) #shape of XTest 8, 8, 1
    XTest = normalize(XTest, axis = 1)
    #normalizes to between 1 and 0 (axis)

    #transforms image into array of size 10 where 1 in a specific
    #index specifies the image label, e.g. 1 in the 6th index indicates
    #that the label is a 7 (as zero-indexed (0 to 9))
    yTrainHot = to_categorical(yTrain)
    yTestHot = to_categorical(yTest)

    #CNN have multiple hidden layers, an input and output
    #2 convolutional layers, 64 -> 32 -> 16 -> flatten layer into 1D array
    classif = Sequential()
    #1st convolutional layer
    classif.add(Dense(64, activation = 'relu'))

    #second layer
    classif.add(Conv2D(32, (2,2), input_shape = (8, 8, 1))) #2x2 matrix
    classif.add(Activation('relu'))
    classif.add(MaxPooling2D(pool_size = (2,2)))
    #factors by which to downscale (vertical, horizontal) = (2,2)

    #third convolutional layer
    classif.add(Conv2D(20, (2,2)))
    classif.add(Activation('relu'))
    classif.add(MaxPooling2D(pool_size = (2,2)))
    #factors by which to downscale (vertical, horizontal) = (2,2)

    classif.add(Flatten())
    #flatten to 1D
    classif.add(Dense(10, activation = 'softmax'))

    #categorical_crossentropy for when more than 2 classes
    #adam optimiser controls learning rate
    #metrics provide info about whatever is set
    classif.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    ####################################################
    # To train the two models, uncomment the below code#
    ####################################################
    #classif.fit(XTrain, yTrainHot, validation_data = (XTest, yTestHot), epochs = 10, batch_size = 10)
    #pickle.dump(classif, open("convolutional.sav", "wb"))


def nonConv():
    digits = datasets.load_digits()

    X = digits.data/digits.data.max()
    y = digits.target
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Flatten the images
    XTrain = XTrain.reshape(1437, 8, 8, 1) #shape of XTrain, 8, 8, 1
    XTrain = normalize(XTrain, axis = 1)
    XTest = XTest.reshape(360, 8, 8, 1) #shape of XTest 8, 8, 1
    XTest = normalize(XTest, axis = 1)

    #one-hot encode
    yTrainHot = to_categorical(yTrain)
    yTestHot = to_categorical(yTest)

    classif = Sequential()
    classif.add(Dense(64, activation= "relu"))
    classif.add(Dense(32, activation = "relu"))
    classif.add(Flatten())
    classif.add(Dense(10, activation = 'softmax'))

    classif.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
    ####################################################
    # To train the two models, uncomment the below code#
    ####################################################
    #classif.fit(XTrain, yTrainHot, batch_size = 10, epochs = 10, validation_data = (XTest, yTestHot))
    #pickle.dump(classif, open("nonConvolutional.sav", "wb"))


def crossValidation():
    #1. shuffle dataset randomly
    #2. slit data set into k groups (k = 10)
    #3. for each k group,
        #1. take the group as a test dataset
        #2. take the remaining groups as a training data set
        #3. fit a model on the training set and evaluate it on the test set
        #4. retain evaluation score and discard model

    digits = datasets.load_digits()
    #dataset is already random
    X = digits.data/digits.data.max()
    y = digits.target
    X, X1, y, y1 = train_test_split(X, y, test_size=0.2)

    #creating folds for knn data
    knn2X, knn3X, knn4X, knn5X = np.array_split(X ,4)
    knn2y, knn3y, knn4y, knn5y = np.array_split(y, 4)

    knnXFolds = [X1, knn2X, knn3X, knn4X, knn5X]
    knnyFolds = [y1, knn2y, knn3y, knn4y, knn5y]

    #normalise data and create folds for conv and
    #non conv nn
    X = X.reshape(1437, 8, 8, 1) #shape of XTrain, 8, 8, 1
    X = normalize(X, axis = 1)
    X1 = X1.reshape(360, 8, 8, 1) #shape of XTest 8, 8, 1
    X1 = normalize(X1, axis = 1)
    #one-hot encode
    yHot = to_categorical(y)
    y1Hot = to_categorical(y1)

    y5, y2, y3, y4 = np.array_split(yHot, 4)
    X5, X2, X3, X4 = np.array_split(X, 4)
    #XTrain and yTrain hold 80% of training data
    #XTrain1 and yTrain1 = 20% = 1 group

    #data for each fold
    XFolds = [X1, X2, X3, X4, X5]
    yFolds = [y1Hot, y2, y3, y4, y5]

    #open models
    convModel = pickle.load(open("convolutional.sav", "rb"))
    nonConv = pickle.load(open("nonConvolutional.sav", "rb"))
    knnModel = pickle.load(open("knn.sav", "rb"))
    #knnImp = pickle.load(open("knnImplementation.sav", "rb"))
    nonConvAccuracy = 0
    convAccuracy = 0
    knnAccuracy = 0
    #knnImpAcc = 0
    for i in range(5): #5 folds
        #creates copies so a previous fold wont affect next fold
        copyX = XFolds
        copyY = yFolds
        knnCopyX = knnXFolds
        knnCopyY = knnyFolds
        #get single test data
        XTest = copyX[i]
        yTest = copyY[i]
        knnXTest = knnCopyX[i]
        knnyTest = knnCopyY[i]
        #training data is created by deleting the test data from the fold
        #the remaining data is concatenated so it can be used to fit
        XTrain = np.delete(copyX, i)
        XTrain = np.concatenate(XTrain)
        yTrain = np.delete(copyY, i)
        yTrain = np.concatenate(yTrain)
        knnXTrain = np.delete(knnCopyX, i)
        knnXTrain = np.concatenate(knnXTrain)
        knnyTrain = np.delete(knnCopyY, i)
        knnyTrain = np.concatenate(knnyTrain)

        knnModel.fit(knnXTrain, knnyTrain)
        knnAccuracy += (knnModel.score(knnXTest, knnyTest))

        history = nonConv.fit(XTrain, yTrain, batch_size = 10, epochs = 10, validation_data = (XTest, yTest))
        nonConvAccuracy += (sum(history.history['accuracy']))/10
        #calc accuracy for model for each fold

        history = convModel.fit(XTrain, yTrain, batch_size = 10, epochs = 10, validation_data = (XTest, yTest))
        convAccuracy += (sum(history.history['accuracy']))/10
        #calc accuracy for model for each fold

    print("\nAccuracy for cross validation")
    print("Convolutional Model: {:.0%}".format(convAccuracy/5))
    print("Non Convolutional Model: {:.0%}".format(nonConvAccuracy/5))
    print("KNN Model: {:.00%}".format(knnAccuracy/5))

while __name__ == '__main__':
    print("Enter number to choose option")
    print('''
          1) Train convolutional model
          2) Train non convolutional model
          3) Cross validation for all models
          9) Exit
    ''')
    chosen = int(input())
    if chosen == 1:
        convolutional()

    elif chosen == 2:
        nonConv()

    elif chosen == 3:
        crossValidation()

    elif chosen == 9:
        break
