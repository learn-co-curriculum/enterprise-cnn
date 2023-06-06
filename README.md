# TensorFlow and TensorBoard with Evaluation



## Purpose

The purpose of this lab is twofold.  

1.   to review using `TensorFlow` for modeling and evaluation with neural networks
2.   to learn about [`TensorBoard`](https://www.tensorflow.org/tensorboard)

`TensorBoard` is `TensorFlow`'s visualization toolkit, so it is a dashboard that provides visualization and tooling that is needed for machine learning experimentation. 

We'll be using the canonical [Titanic Data Set](https://www.kaggle.com/competitions/titanic/overview).

## The Titanic

### The Titanic and it's data



RMS Titanic was a British passenger liner built by Harland and Wolf and operated by the White Star Line. It sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton, England to New York City, USA.

Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. 

Though there were about 2,224 passengers and crew members, we are given data of about 1,300 passengers. Out of these 1,300 passengers details, about 900 data is used for training purpose and remaining 400 is used for test purpose. The test data has had the survived column removed and we'll use neural networks to predict whether the passengers in the test data survived or not. Both training and test data are not perfectly clean as we'll see.

Below is a picture of the Titanic Museum in Belfast, Northern Ireland.

### Data Dictionary

*   *Survival* : 0 = No, 1 = Yes
*   *Pclass* : A proxy for socio-economic status (SES)
  *   1st = Upper
  *   2nd = Middle
  *   3rd = Lower
*   *sibsp* : The number of siblings / spouses aboard the Titanic
  *   Sibling = brother, sister, stepbrother, stepsister
  *   Spouse = husband, wife (mistresses and fiancés were ignored)
*   *parch* : The # of parents / children aboard the Titanic
  *   Parent = mother, father
  *   Child = daughter, son, stepdaughter, stepson
  *   Some children travelled only with a nanny, therefore *parch*=0 for them.
*   *Ticket* : Ticket number
*   *Fare* : Passenger fare (British pounds)
*   *Cabin* : Cabin number embarked
*   *Embarked* : Port of Embarkation
  *   C = Cherbourg (now Cherbourg-en-Cotentin), France
  *   Q = Queenstown (now Cobh), Ireland
  *   S = Southampton, England
*   *Name*, *Sex*, *Age* (years) are all self-explanatory

## Libraries and the Data



### Importing libraries

### Loading the data

## EDA and Preprocessing

### Exploratory Data Analysis

It is your choice how much or how little EDA that you perform. But you should do enough EDA that you feel comfortable with the data and what you'll need to do to make it so that you can run a neural network on it.

What follows in some suggestions for EDA. This is by no means exhaustive, but just a sampling of what you can do. It is important for each EDA code chunk that you run that you reflect on what the output is telling you.

#### Overview

We notice that the total number of columns in the training data (12) is one more tha test data (11). The former has the "survived" column whereas the latter does not since that is what we want the neural network to be able to predict.

#### Visualizations

There are a number of plots that we could do. Here are a few with brief comments about each.

Since '0' represents not surviving and '1' represents surviving, then we can see that females were much more likely to survice than males.

And now let's look at survival by class (frist, second, or third).

While the differences between second and third class doesn't seem very diffrent, those in third class seem more likely to die.

We can see this more clealry in the next graph.

What conclusions can you draw from the above grid?

Finally, let's look at a heatmap to see which columns are correlated.

What columns are correlated? Does this surprise you? Which columns are not correlated? Does this surprise you?

#### Data Analysis

Again there are a myriad of data analysis that we can do, but let's show some examples of what we may want to look it. Again, do as many as you think are appropriate and you should interpret the results explicitly.

Females had about 74% chance of survival where as men had only about a 19%.

Now let's do the same, but with class as well.

We can see that as the class increases, the likelihood of survivial increases; and again females are more likely to survive then males.

### Preprocessing

#### Missing Data

In order for our neural network to run properly, we need to see if we have any missing data... and if so, we need to deal with it.

So we have a decent amount missing data for *Age* in both the train and tests sets. There are also a couple from *Embarked* from the train set and one in *Fare* from the test set.

There are different ways to deal with missing data, which we'll turn to next.

#### Combining numeric features and converting categorical feasture

One possibility for dealing with the missing data is to
*    fill the nan values for *Age* and *Fare* with the median of each respective column
*   the embarked column with the most frequent value, which is *S* for Southampton

Once we clean up *Age* we can create a variable from it called *Child*, that allows us to look at the data as adults or children.

We have two categorical features, viz. *Sex* and *Embarked*. We need to make these numeric.

For *Sex*, let
*   Male = 0
*   Female = 1

For *Embarked*, let
*   Q = 0 (Queenstown)
*   S = 1 (Southampton)
*   C = 2 (Cherbourg)

If you did the heatmap above, you can see that there are some correlations between variables. You can choose modify the data set if you like. For the purposes of this exercise, we won't.

While the names of the individual passengers is not germane to our neural network model, the titles of the passengers may be worthwhile.

Finally, we ned to drop the feature columns that we do not need for our model.

#### Resplitting the data and scaling the data.

We need to split our data back into the training and test sets now that we have done all but one aspect of the preprocessing.

We need to get the training data into the predictors and the predicted variables, and scale the data.

## Neural Network Model

### Building the model

#### Define the model as a pipeline

Let's use the data science pipeline for our neural network model.

#### Fitting the optimal model and evaluating with `TensorBoard`

#### `TensorBoard`

`TensorBoard` is `TensorFlow`'s visualization toolkit. It is a dashboard that provides visualization and tooling that is needed for machine learning experimentation. The code immediately below will allow us to use TensorBoard.

N.B. When we loaded the libraries, we loaded the TensorBoard notebook extension. (It is the last line of code in the first code chunk.)

#### Fitting the optimal model and evaluating with `TensorBoaard`

#### Results and Predictions

Continue to tweak your model until you are happy with the results based on model evaluation.

## Conclusion

Now that you have the `TensorBoard` to help you look at your model, you can better understand how to tweak your model.

We'll continue with this for the next lesson when we learn about model regularization.
