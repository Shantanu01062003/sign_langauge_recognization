[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<h1>Sign Language Recognition Using OpenCV</h1>
<h2>Dependencies:</h2>
  <p>1. Tensorflow</p>
  <p>2. Keras</p>
  <p>3. OpenCV</p>
  
  
<h3>Dataset:</h3>
<link>https://www.kaggle.com/datamunge/sign-language-mnist</link>

<h3>Tools:</h3>
  <p>Google Colab</p>

<h2>How to run</h2>
<p>Run ROIinOpenCV.py</p>
Pytorch Implementation is given in sign_language_pytorch.ipynb


<h1>Sign Language Recognition Using CNN and OpenCV<h1>

<h5>To build a SLR (Sign Language Recognition) we will need three things:
</h5>

<h3>1)Dataset</h3>
<h3>2)Model (In this case we will use a CNN)</h3>
<h3>3)Platform to apply our model (We are going to use OpenCV)
</h3>


<p>1) Dataset
We will use MNIST (Modified National Institute of Standards and Technology)dataset.

You can download the dataset here.

Basically, our dataset consists of many images of 24 (except J and Z) American Sign Language alphabets. Each image has size 28×28 pixel which means total 784 pixels per image.


English Alphabets for Sign Language
Loading the dataset to colab
To load the dataset into colab use this code:

from keras.datasets import mnist
(X_train, Y_train) , (X_test , Y_test) = mnist.load_data()
Our dataset is in CSV(Comma-separated values) format. train_X and test_X contain the values of each pixel. train_Y and test_Y contain the label of image. You can use the following code to see the dataset:



    display(X_train.info())
    display(X_test.info())
    display(X_train.head(n = 2))
    display(X_test.head(n = 2))


Preprocessing
train_X and test_X consists of an array of all the pixel pixel values. We have to create an image from these values. Our image size is 28×28 hence we have to divide the array into 28×28 pixel groups. To do that we will use the following code:


    X_train = np.array(X_train.iloc[:,:])
    X_train = np.array([np.reshape(i, (28,28)) for i in X_train])
    X_test = np.array(X_test.iloc[:,:])
    X_test = np.array([np.reshape(i, (28,28)) for i in X_test])num_classes = 26
    y_train = np.array(y_train).reshape(-1)
    y_test = np.array(y_test).reshape(-1)y_train = np.eye(num_classes)[y_train]
    y_test = np.eye(num_classes)[y_test]X_train = X_train.reshape((27455, 28, 28, 1))
    X_test = X_test.reshape((7172, 28, 28, 1))

Now we can use this dataset to train our model.

2) Build and Train the Model
We will use CNN (Convolutional Neural Network) to recognise the alphabets. We are going to use keras.

If you are building this project then you should know how CNN works. If you are not familiar with CNN then I would highly recommend you this course Andrew Ng’s Convolutional Neural Networks on Coursera. Or you can follow my own blog from here.


Here’s our model:

    classifier = Sequential()
    classifier.add(Conv2D(filters=8, kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(28,28,1),activation='relu', data_format='channels_last'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(filters=16, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
    classifier.add(Dropout(0.5))<br>classifier.add(MaxPooling2D(pool_size=(4,4)))
    classifier.add(Dense(128, activation='relu'))<br>classifier.add(Flatten())
    classifier.add(Dense(26, activation='softmax'))

As you can observe, like any other CNN our model consists of couple of Conv2D and MaxPooling layers followed by some fully connected layers (Dense).
The first Conv2D (Convolutional) layer takes input image of shape (28,28,1). The last fully connected layer gives us output for 26 alphabets.
We are using a Dropout after 2nd Conv2D layer to regularise our training.
We are using softmax activation function in the final layer. Which will give us probability for each alphabet as an output.
At the end our model looks like this:


Model summary
We have to compile and fit our model. To do that we will use this:




    classifier.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(X_train, y_train, epochs=50, batch_size=100)


We are using SGD optimiser to compile our model. You may decrease the epochs to 25.

Finally to check the accuracy we will use this:



    accuracy = classifier.evaluate(x=X_test,y=y_test,batch_size=32)
    print("Accuracy: ",accuracy[1])


Now to download the trained model on our PC we can use this:


    classifier.save('CNNmodel.h5')
    weights_file = drive.CreateFile({'title' : 'CNNmodel.h5'})
    weights_file.SetContentFile('CNNmodel.h5')<br>weights_file.Upload()
    drive.CreateFile({'id': weights_file.get('id')})


It will save the trained model to your drive.

3) OpenCV
Create a Window.
We have to create a window to take the input from our webcam. The image which we are taking as an input should be 28×28 grayscale image. Because we trained our model on 28×28 size image.

To create the window


    def main():
        while True:  

       # capturing the image from webcam 
       cam_capture = cv2.VideoCapture(0)
       _, image_frame = cam_capture.read()
  
       # to crop required part
       im2 = crop_image(image_frame, 300,300,300,300)

       # convert to grayscale 
       image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
       # blurring the image 
       image_grayscale_blurred =cv2.GaussianBlur(image_grayscale, (15,15), 0)

       # resize the image to 28x28
       im3 = cv2.resize(image_grayscale_blurred, (28,28), interpolation = cv2.INTER_AREA)

       # expand the dimensions from 28x28 to 1x28x28x1
       im4 = np.resize(im3, (28, 28, 1))
       im5 = np.expand_dims(im4, axis=0)


Prediction
Now we have to predict the alphabet from the input image. Our model will give outputs as integers rather than alphabets that’s because the labels are given as integers (1 for A, 2 for B, 3 for C and so on..)


#model is our classifier and image is input image we are passing
 
    
    def keras_predict(model, image):
        data = np.asarray( image, dtype="int32" )
        pred_probab = model.predict(data)[0]
    
    # softmax gives probability for all the alphabets hence we have to choose the maximum probability alphabet
    
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class
Our model’s accuracy is 94% so it should recognise alphabets without any problem with plain background and descent lights.

Done!
</p>
