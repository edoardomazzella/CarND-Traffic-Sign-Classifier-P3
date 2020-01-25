# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/signs.png "Visualization"
[image2]: ./examples/histogram1.png "Histogram 1"
[image3]: ./examples/histogram2.png "Histogram 2"
[image4]: ./examples/preproc.png "Pre Processing"
[image5]: ./examples/01.png "Traffic Sign 1"
[image6]: ./examples/02.png "Traffic Sign 2"
[image7]: ./examples/03.png "Traffic Sign 3"
[image8]: ./examples/04.png "Traffic Sign 4"

[image9]: ./examples/05.png	"Traffic Sign 5"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used  few line of python code in order to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

 ![alt text][image1]

The bar chart shows the number of occurrences for each of the 43 classes of the traffic signs.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to equalize the histogram in order to obtain an uniform distribution with 4000 occurrences per sign.

![alt text][image3]

The data set augmentation has been applied by adding randomly selected images from the training data set with modified lightness  or orientation. 

Here is an example of an image Pre-processing which modifies its lightness. 

![alt text][image4]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I choosed LeNet as final model. The following table shows its architecture:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
| RELU	|												|
| Max Pooling	      | 2x2 kernel, 2x2 stride,  outputs 14x14x6 |
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16 |
| RELU	|         									|
| Max Pooling	| 2x2 kernel, 2x2 stride,  outputs 5x5x16 |
| Fully connected | Input = 400, Output = 120 |
| RELU |												|
| Fully connected | Input = 120, Output = 84 |
| RELU |	|
| Fully connected | Input = 84, Output = 43 |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with a learning rate of 0.0005 and I set 20 epochs with a bath size of 128. This is the result of a parameter tuning based on the validation set accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 93.5%
* test set accuracy of 90.8%

I choosed LeNet architecture for my model as presented in the previous lessons. In order to reach the required validation accuracy i tuned the learning rate, the number of epochs, the batch size in addition to the data pre-processing which has been the main key for success. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

* The first image might be difficult to classify because of the shadows on the sign.
* The second and third are good qualities images.
* The fourth and fifth are a little bit dark.

Depending on the training set the model could have problems in correctly predict the classes of these signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit 70 | Speed limit 60 |
| Keep right | Speed limit 60 |
| Speed limit 120	| Speed limit 120	|
| Speed limit 100	| Speed limit 60	|
| Speed limit 100	| Speed limit 60 |


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. The accuracy is very low respect to the test set and it demonstrates that the training set shall be further augmented with different pre-processing techniques in order to get good results.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last cell of the Ipython notebook.

The top five soft max probabilities for the Speed Limit 70km/h sign were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .10777834  | Speed limit (60km/h) |
| .09465593 | Speed limit (80km/h) |
| .08328136	| Speed limit (120km/h)	|
| .0635726	 | End of speed limit (80km/h)	|
| .03677879	| Dangerous curve to the right |

The top five soft max probabilities for the Keep Right sign were

| Probability |          Prediction          |
| :---------: | :--------------------------: |
|  .12940538  |     Speed limit (60km/h)     |
|  .09520334  |    Speed limit (120km/h)     |
|  .07817145  |     Speed limit (80km/h)     |
|  .05288851  |    Speed limit (100km/h)     |
|  .05018421  | Dangerous curve to the right |

The top five soft max probabilities for the Speed Limit 120km/h sign were

| Probability |          Prediction          |
| :---------: | :--------------------------: |
|  .14798599  |    Speed limit (120km/h)     |
|  .13001789  |     Speed limit (60km/h)     |
|  .09946342  |     Speed limit (80km/h)     |
|  .05718621  |    Speed limit (100km/h)     |
|  .0459808   | Dangerous curve to the right |

The top five soft max probabilities for the Speed Limit 100km/h sign were

| Probability |          Prediction          |
| :---------: | :--------------------------: |
|  .09513693  |     Speed limit (60km/h)     |
|  .0830178   |    Speed limit (120km/h)     |
|  .06556524  |     Speed limit (80km/h)     |
|  .05923611  | Dangerous curve to the right |
|  .04210715  |    Speed limit (100km/h)     |

The top five soft max probabilities for the last Speed Limit 100km/h sign were

| Probability |             Prediction             |
| :---------: | :--------------------------------: |
|  .14798599  |        Speed limit (60km/h)        |
|  .13001789  |       Speed limit (120km/h)        |
|  .09946342  |        Speed limit (80km/h)        |
|  .05718621  |       Speed limit (100km/h)        |
|  .0459808   | Dangerous curve to the rightEnd of |

From this results it can be deduced that the neural network shall be further improved from an architectural point of view and that the training data set is not adequate to reach good results. Augmenting the data set with by preprocessing the already existing images and by adding new images should be the key.
