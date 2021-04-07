# Image Caption Generator
 
Given an image, generates a caption for it using two different neural networks; Convolutional Neural Network (CNN) and Long Short Term Memory Network (LSTM).

It uses transfer learning using Xception model to leverage the model's trained parameters to encode an image to a 2048 feature vector which is then fed into an LSTM to predict a caption based on the features extracted by Xception.
<br /><br />


## Flow of the project

**a. Cleaning the caption data**

**b. Extracting features from images using VGG-16**

**c. Merging the captions and images**

**d. Building LSTM model for training**
**e  Predicting on test data**
**f. Evaluating the captions using BLEU scores as the metric**


## Model Architecture
<br />

<img src="model_plot.png"></img>
<br /><br />

- We remove the last layer of Xception network
- Image is fed into this modified network to generate a 2048 length encoding corresponding to it
- The 2048 length vector is then fed into a second neural network along with a caption for the image (while training)
- This second network consists of an LSTM which tries to generate a caption for the image
<br /><br />

## Examples

Here are some captions generated by the network:
<br />

<div style="float:left">
<div style="float:left"><img width="45%" src="https://i.imgur.com/Nmeg3DD.jpg" />
<img width="45%" src="https://i.imgur.com/J4mnPsG.jpg" /></div>
</div>
<br /><br />

