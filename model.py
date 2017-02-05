import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import csv
import math
import os
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, ELU, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json

num_epochs = 30
batch_size = 64


def random_gamma(image):
	"""
	Random gamma correction is used as an alternative method changing the brightness of training images.
	http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
	"""
	gamma = np.random.uniform(0.4, 1.5)
	inv_gamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** inv_gamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")
	
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def random_shear(image, steering_angle, shear_range = 100):
	"""
	Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
	"""
	
	randi = np.random.randint(0, 2)
	if randi == 1:
		rows, cols, ch = image.shape
		dx = np.random.randint(-shear_range, shear_range + 1)
		random_point = [cols / 2 + dx, rows / 2]
		
		pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
		pts2 = np.float32([[0, rows], [cols, rows], random_point])
		dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
		
		M = cv2.getAffineTransform(pts1, pts2)
		image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
		steering_angle += dsteering
		
	return image, steering_angle


# preprocess image: random flip, crop and resize
def preporcess_img(img, angle):
	# randomly shear image
	img, angle = random_shear(img, angle)
	
	# randomly flip the image
	randi = np.random.randint(0, 2)
	if randi == 0:
		# flip the image
		img   = cv2.flip(img, 1)
		angle = -1.0 * angle
	
	# crop the image, remove the top 54, and lower 25 pixels
	img = img[55:135, :, :]
	
	# adjust brightness
	img = random_gamma(img)
	
	# use opencv to resize image to new dimension
	img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_CUBIC)
	
	return img, angle


# generate training/validation batch
def get_batch(X, batch_size = 64):
	# randomly pickup training data to create a batch
	while(True):
		X_batch = []
		y_batch = []
		
		picked = []
		n_imgs = 0
		
		# randomly selected batch size images and steering angles
		while n_imgs < batch_size:
			i = np.random.randint(0, len(X))
			if (i in picked):
				continue  # skip if this image has been picked
			y_angle  = float(X[i][3])
			
			# if the angle is very small, drop it in 70% probablity
			# if (y_angle < 0.01):
			#    keep_prob = np.random.uniform()
			#    if (keep_prob < 0.5):
			#        continue
			
			picked.append(i)
			# randomly pick one of 3 images (center, left, right)
			# take 50% images from center camera
			img2pick = np.random.randint(0, 2)
			if img2pick == 1:
				img2pick = np.random.randint(1, 3)
				# compensate steering angle when using left/right camera
				if img2pick == 1:
					y_angle += 0.225
				elif img2pick == 2:
					y_angle -= 0.225
			
			img_path = './data/' + X[i][img2pick].strip()
			drv_img  = plt.imread(img_path)
			
			# preprocess image
			
			drv_img, y_angle = preporcess_img(drv_img, y_angle)
			X_batch.append(drv_img)
			y_batch.append(y_angle)
			n_imgs += 1
		
		yield np.array(X_batch), np.array(y_batch)


# create model
def get_model():
	
	model = Sequential()
	
	# normalization layer
	model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))
	
	# convolution 2D with filter 5x5
	model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	
	model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Dropout(0.4))
	
	model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Dropout(0.25))
	
	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Dropout(0.25))
	
	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(ELU())
	
	model.add(Flatten())
	
	model.add(Dense(1164))
	model.add(ELU())
	model.add(Dropout(0.5))
	
	model.add(Dense(100))
	model.add(ELU())
	
	model.add(Dense(50))
	model.add(ELU())
	
	model.add(Dense(10))
	model.add(ELU())
	
	model.add(Dense(1))
	
	return model


def get_samples_per_epoch(num_samples, batch_size):
	# return samples per epoch that is multiple of batch_size
	return math.ceil(num_samples/batch_size)*batch_size


driving_data = []
# create a list of image paths and angles
with open('data/driving_log.csv') as drvfile:
	reader = csv.DictReader(drvfile)
	for row in reader:
		driving_data.append((row['center'], row['left'], row['right'], row['steering']))


driving_data = shuffle(driving_data)
# split the data, 20% for validation
X_train, X_validation = train_test_split(driving_data, test_size = 0.2, random_state = 7898)


train_generator = get_batch(X_train)
val_generator   = get_batch(X_validation)

ans = 'n'
if os.path.exists(os.path.join('.', 'model.json')):
	ans = input('A model already exists, do you want to reuse? (y/n): ')

if ans == 'y' or ans == 'Y':
	with open(os.path.join('.', 'model.json'), 'r') as jfile:
		model = model_from_json(json.load(jfile))
        
	weights_file = os.path.join('.', 'model.h5')
	model.load_weights(weights_file)
	print('Get model from the disk')
	
	model.summary()
	num_epochs = 10
else:
	model = get_model()
    
model.compile(optimizer = Adam(lr = 0.0001), loss='mse')

print("Start training...")
model.fit_generator(train_generator,
		samples_per_epoch = get_samples_per_epoch(len(X_train), batch_size),
		nb_epoch = num_epochs,
		validation_data = val_generator,
		nb_val_samples = get_samples_per_epoch(len(X_validation), batch_size),
		verbose = 1)

# save model and weights
model_json = model.to_json()
with open("./model.json", "w") as json_file:
	json.dump(model_json, json_file)
model.save_weights("./model.h5")
print("Saved model to disk")

