# Evaluation of Machine Learning algorithms and Bias Variance Trade off analysis

## I: Description of the experment
The objective of this assignment to evaluate machine learning algorithms on a given data
set and analyzing the bias variance tradeoff as complexity of the model increases to fit the
data using different polynomial regressions

The selected dataset consists of different attributes of the plant called abalone. The main
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

## II- Design of the experiment

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

###### the following example for the linear regression (polynomial degree 1) I created a variable called p1_results and
the result looks like 

![Regreesion]()


