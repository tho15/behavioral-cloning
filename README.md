# Behavrioal Cloning Project - Udacity Self Driving Car Nanodegree Project 3

## The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Train the model on data set from track one and test it on track two

## Results
I tried to collect data from the simulator which I navigated using keyboard, but it was difficult to generate a smooth driving. The model does not perform well with my own dataset. Later I only use the sample dataset from Udacity project site for training my model. By training on the sample dataset, my model is able to drive successfully around both track one and two without leaving the road.

## Rubric Points

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md describing the network modle, training approach and summarizing the results

#### 2. Submssion includes functional code
Using the simulator provided by Udacity and my drive.py file, the car can be driven autonomously around both tracks in simulator by executing 
```sh
python drive.py model.h5
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The code in model.py (line 86-130) uses a Python generator, which randomly samples the set of available images from the CSV file, preprocess the images and stores the data in memory.


### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

I borrow the model from NVIDIA's End to End Learning for Self-Driving Cars paper. The network consists of 10 layers, including a normalization layer, 5 convolutional layers and 4 fully connected layers. The first layer of the network performs image normalization using Keras lambda layer (code line 139). The model uses strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers. Maxpooling layer is added after first four convolutional layers. Follow the five convolutional layers are four fully connected layers (1164, 100, 50, 10) leading to an output angle.

The model includes ELU layer after each convolutional layer and fully connected layer to introduce nonlinearity.

#### 2. Final Model Architecture

Here is a visualization of the final architecture (model.py lines 139-179)

![Architecture](image/model.png)


#### 3. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 149, 154, 159 and 168). 

The sample data set is separated to two different sets, 80% of data was used for generating training images, while the other 20% of data was for validation (code line 199). The model was tested by running it through the simulator and ensuring that the car could stay on the track.

#### 4. Model parameter tuning

The model uses an adam optimizer with learning rate 0.0001 when doing traing. When fine turning the model, I use learning rate 0.00001. This was emperically noted when using transfer learning.

#### 5. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The simulator recorded three images taken from left, center, and right cameras in each data. I choose images randomly between center, left and right images, with 50% probability of choosing center, and 25% probability of either left or right images. The idea of using left and right images is to teach car to recover from the left and right sides. A small angle adjustment is added to steering angle when using left or right camera images.

For details about how I created the training data, see the next section. 

### Model Development and Data Generation

#### 1. Solution Design Approach

I started with a model that consists two convolutional layers and three fully connected layers. This is the model I used successfully for my traffic sign recognition project. But the model does not work well after many tries on tuning parameters. It failed to converge to a reasonable loss value and the result is not stable. Nevertheless it gave me some idea on the complexity of the problem.

I decided to use the architecture described in the Nvidia paper End to End Learning for Self-Driving Cars. This model resulted in a much lower mean squared error comparing to my first one, and it performed much better on my first test in simulation. However it did not handle the sharp turn well. To further improve the model, I took following steps:

* Added Maxpooling layer following each of the first 4 convolutional layers
* Added Dropout layers to avoid overfitting
* Implemented image cropping that remove part of top and low pixes. The ideal is that those pixels do not provide valueable information on driving angle

After making these changes, the car is able to sucessfully drive around track one. Then I tested the model on track two, it failed in early turns. Increasing the number of training epochs and fine-tuning parameters did not help. After reading some posts on forum and blogs ([Kaspar's nice blog](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.kkvdh7ig7)), I added following image processing before fed the images to model:

* Random image shearing. The large portion of dataset have zero steering angle, image shearing create a new image, and the steering angle is adjusted with sheering angle. This creats new image with different steering angle.
* Brightness adjustment. The images from second track has much lower brightness. To teach network to handle darker images, Random brightness adjustment is applied to all images before network training.

The car is able to successfully complete track two simulation after applying these two image processing steps and run the model with 30 epochs. However, to complete track two, I have to increase the throttle to 0.3, but I found the car tends to wobble on track one when using 0.3 throttle. To combat this issue, I first made the throttle in the drive.py varying linearly with steering angle ([an idea from this post](https://carnd-forums.udacity.com/questions/36904752/behavioral-cloning-mysteries-)), the fomula is `throttle = max(0.2, -0.45*abs(steering_angle)+0.32)`. Then I fine tune the model without image shearing processing, and with lower angle adjustment for left/right camera. After taking these two steps, the vehicle is able to move smoother on both track.


#### 3. Creation of the Training Set & Training Process

The training set is based on the sample training data from Udacity. The original training set consists of 8036 records, each record has 3 images that captured by center, left and right camera. Here is an example record of images:

center | left | right
-------|------|-------
![center](image/center.jpg) | ![left](image/left.jpg) | ![right](image/right.jpg)


The majority of steering angles in the dataset are zeros. The zero steering angle will bia the car to drive straight. To train on a more balanced dataset, one way is to use left/right images, and modify the steering angle. When selected image for training, 50% images are selected from left or right record images. Another solution is to apply image shearing and adjust the steering angle proportional to shearing angle. This operation creates a new distorted image with non-zero steering angle. The image shearing is applied to train image with 50% probablity in the first step of image processing. Here is the sample of sheared image:

input image | sheared image
------------|--------------
![center](image/center.jpg) | ![sheared](image/sheared.jpg)

The training track one has more left turning curves than right ones. To train our model for a more general driving track, our second step is to randomly flipped images and angles with probability of 50%. Following shows the flipped image with original one:

left image | flipped image
-----------|--------------
![left](image/left.jpg) | ![flipped](image/flipped.jpg)

The next step of image augmentation is to crop image. I remove the top 54 pixels and bottom 25 pixels. The following figures show the result of a cropped image:

input image | cropped image
------------|--------------
![center](image/center.jpg) | ![flipped](image/cropped.jpg)

After crop operation, we changed image's brightness by appling gamma correction. Following figures show image after a random brightness change:

cropped image | brightness changed image
--------------|-------------------------
![cropped](image/cropped.jpg) | ![gamma](image/gamma.jpg)

The final stop is to rescal the image to 64x64 to reduce training time. Before train, I randomly shuffled the data set and put 20% of the data into a validation set. 

The ideal number of epochs was 30 for first training, which I found by running the training in 5, 10, 20, 30 and 50 epochs. After running first training, I fine turned the model as discussed in the previous section.

