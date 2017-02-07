#Behavrioal Cloning Project - Udacity Self Driving Car Nanodegree Project 3

##The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Train the model on track one driving data and test it on track two

##Results
I tried to collect data from the simulator which I navigated using keyboard, but it was difficult to generate a smooth driving. The model does not perform well with the dataset. Later I only use the sample dataset from Udacity project site. By training on the sample dataset, my model works on both tracks even though the dataset only consist data from track one.

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md describing the network modle, training approach and summarizing the results

####2. Submssion includes functional code
Using the simulator provided by Udacity and my drive.py file, the car can be driven autonomously around both tracks in simulator by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The code in model.py (line 86-130) uses a Python generator, which randomly samples the set of available images from the CSV file, preprocess the images and stores the data in memory.


###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

I borrow the model from NVIDIA's End to End Learning for Self-Driving Cars paper. The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. The first layer of the network performs image normalization using Keras lambda layer (code line 139). The model uses strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers. Maxpooling layer is added after first four convolutional layers. Follow the five convolutional layers are four fully connected layers (1164, 100, 50, 10) leading to an output angle.

The model includes ELU layer after each convolutional layer and fully connected layer to introduce nonlinearity.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 149, 154, 159 and 168). 

The sample data set is separated to two different sets, 80% of data was used for generating training images, while the other 20% of data was for validation (code line 199). The model was tested by running it through the simulator and ensuring that the car could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer with learning rate 0.0001 when doing traing. When fine turning the model, I use learning rate 0.00001. This was emperically noted when using transfer learning.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The simulator recorded three images taken from left, center, and right cameras in each data. I choose images randomly between center, left and right images, with 50% probability of choosing center, and 25% probability of either left or right images. The idea of using left and right images is to teach car to recover from the left and right sides. A small angle adjustment is added to steering angle when using left or right camera images.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with a model that consists two convolutional layers and three fully connected layers. This is the model I used successfully for my traffic sign recognition project. But the model does not work well after many tries on tuning parameters. It failed to converge to a reasonable loss value and the result is not stable. Nevertheless it gave me some idea on the complexity of the problem.

I decided to use the architecture described in the Nvidia paper End to End Learning for Self-Driving Cars. This model resulted in a much lower mean squared error comparing to my first one, and it performed much better on my first test in simulation. However it did not handle the sharp turn well. To further improve the model, I took following steps:

* Added Maxpooling layer follow the first 4 convolutional layers
* Added Dropout layers to avoid overfitting
* Implemented image cropping that remove part of top and low pixes. The ideal is that those pixels do not provide valueable information on driving angle

After making these changes, the car is able to sucessfully drive around track one. Then I tested the model on track two, it failed in early turns. Increasing the number of training epochs and fine-tuning parameters did not help. After reading some posts on forum and blogs [credit to Kaspar](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.kkvdh7ig7), I added following image processing before fed the images to model:

* Random image shearing. The large portion of dataset have 0 steering angle, image shearing create a new image, and the steering angle is adjusted with sheering angle. This creats new image with different steering angle.
* Brightness adjustment. The images from second track has much lower brightness. To teach network to handle darker images, Random brightness adjustment is applied to all images before network training.

The car is able to successfully complete track two simulation after applying these two image processing steps and run the model with 30 epochs. However, to complete track two, I have to increase the throttle to 0.3, but I found the car tends to wobble on track one when using 0.3 throttle. To combat this issue, I first make the throttle in the drive.py varying linearly with steering angle [post](https://carnd-forums.udacity.com/questions/36904752/behavioral-cloning-mysteries-), the fomula is 'throttle = max(0.2, -0.45*abs(steering_angle)+0.32)'. Then I fine tune the model without image shearing processing, and lower angle adjustment for left/right camera. After taking these two steps, the vehicle is able to move smoother on both track.


####2. Final Model Architecture

The final model architecture (model.py lines 139-179) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

