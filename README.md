# Evaluation of Machine Learning algorithms and Bias Variance Trade off analysis

## I - Introduction
### What is Bias?

Bias is the difference between the average prediction of a model and the correct value which we are trying to predict. Model with high bias means very little attention is given to the training data and oversimplifies the model. This always leads to high error on training and test data.

### What is Variance?
Variance is the variability of model prediction for a given data point  which tells us spread of our data.  Model with high variance pays a lot of attention to the training data and does not generalize on the test data(unseen data) .Thus, such models perform  really well on training data, but has high error rates on test data.

## II - Description of the experment

The main goal of this project to evaluate machine learning algorithms on a particular data set
called abalone and analyzing the bias variance tradeoff as complexity of the model increases to fit
the data using different polynomial regressions.

The selected dataset consists of different attributes of the plant called **abalone**. The main
task of the experiment is predicting the age of the plant from different physical
measurements like height, width, height and others. The data sets consist of 4177
instances and 8 features and one target variable called rings which is the age of the plant.
The data is mainly collected for regression purpose.

The link for the abalone data sets [abalone](https://archive.ics.uci.edu/ml/datasets/abalone)


For solving the probelem we used Python  and polynomial regression and collection of
functions and classes which are defined inside the sklearn machine learning library. 
Firstly, we modeled the relationship between the independent and response variable using simple
linear mode and polynomial regression to fit the relationship between independent
variables and the response variable. 

## III - Design of the experiment

Firstly, I performed prepressing tasks to drop categorical variables (Sex) and some of
features which are not used in the model such as Shell weight and Shucked weight. After
preprocessing to handle the prediction task, I used cross validation with 5 folds. I used
polynomial features to generate feature matrix consisting of all polynomial combinations
from the linear repressor. To make it explicit, I selected three random rows from the data
sets for computing bias and variance values and I tested the bias and variance using the
records. Furthermore, to store the variance and bias values while predicting, we created a
matrix of with number of test records and number of polynomials as main dimensions of
the matrix.

To store the predicted values we created a matrix of results with test size and number of folds as
dimensions of the matrix. These values are used for calculating the bias and variance. Using the
selected number of cross validations, we predicted the value of each test data 5 times.

 The following example for the linear regression (polynomial degree 1)

![Regreesion](https://github.com/htefera/Bias-Variance-Tradeoff/blob/master/Bias%20Variance%20Images/4.PNG)
 <div align="center">
  Figure 1: Linear regression using cross validation for selected data
 </div>
 <br>
   
Where the rows numbers are the test records and column number’s the number of cross
validations. Even if we used different polynomial regression with various degrees, but there is still
error between the actual data points and predicted values. When I use simple modes, they do not
represent the actual relationship between the response variable (ring) and the feature variable
variables. To decide which model is perfect for my regression I performed bias variance analysis by
varying complexities.

## IV - Discussion of the results and findings from the experiment

After we trained the model on the train data, we evaluated the performance of the model on
the unseen data(aka test data) and registered the following bias variance values for the selected test data.
Where X0, X1 and X2 are the records at position 5,60 and 100 of the [abalone](https://archive.ics.uci.edu/ml/datasets/abalone) dataset. We can use different records and evaluate the model.

Below table shows the bias variance trade off

![Bias Variance Comparision](https://github.com/htefera/Bias-Variance-Tradeoff/blob/master/Bias%20Variance%20Images/5.png)

 <div aling="center">
  Table 1- Table of Bias and Variance for selected test data’s (row 5, 60,100)
 </div>

Note that both bias and variance are calculated column wise (record wise calculation) taking into
consideration all the values of the independent variables. 

### V - Graph of actual data vs predicted data using the model



![Fitting the data](https://github.com/htefera/Bias-Variance-Tradeoff/blob/master/Bias%20Variance%20Images/Bv.png)


<div align="center">
 Figure 2. Fitting the data using different complexities
</div>

<br>


Even if I used different polynomial regression with various degrees, but there is error between the
actual data points and predicted value of using the model. When I use simple modes, they do not
represent the actual relationship between the response variable (ring) and the feature variable
(diameter). Using complex models are also sensitive to small change in the data. Therefore, to
decide which model is perfect for my data set I performed bias variance analysis by varying
complexities and I obtained the following complexity vs bias variance.


![Bias Variance Graph](https://github.com/htefera/Bias-Variance-Tradeoff/blob/master/Bias%20Variance%20Images/bvmain.png)
<div align= "center">
   Figure : Bias variance graph
   </div>               


<br>

As we can see from the graph the variance changes slightly from complexity to complexity but the
bias is almost remains the same.
According the [Occam's razor principle](https://machinelearningmastery.com/ensemble-learning-and-occams-razor/)  bias and variance is low at complexity 1  balances between the bias and variance errors.

## VI- Reference

 1. [Scikit Learn](http://scikit-learn.org/stable/) <br>
 2. [The Bias-Variance Tradeoff](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229) 
 3. [Understanding the Bias-Variance Tradeof](http://scott.fortmann-roe.com/docs/BiasVariance.html)
 4. [Bias-Variance Tradeoff in Machine Learning](https://www.learnopencv.com/bias-variance-tradeoff-in-machine-learning/)














