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

It is prudent to investigate the attributes of the data frames, create visualizations, and perform data analysis.

### Preprocessing


```python
# Print the first rows of the train data
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Look at the information regarding the train data
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB



```python
# Print the first rows of the test data
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Look at the information regarding the test data
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  418 non-null    int64  
     1   Pclass       418 non-null    int64  
     2   Name         418 non-null    object 
     3   Sex          418 non-null    object 
     4   Age          332 non-null    float64
     5   SibSp        418 non-null    int64  
     6   Parch        418 non-null    int64  
     7   Ticket       418 non-null    object 
     8   Fare         417 non-null    float64
     9   Cabin        91 non-null     object 
     10  Embarked     418 non-null    object 
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB



```python
# Print the number of rows and columns in each data set
print("Total number of rows in training data ", train.shape[0])
print("Total number of columns in training data ", train.shape[1])
print("Total number of rows in test data ", test.shape[0])
print("Total number of columns in test data ", test.shape[1])
```

    Total number of rows in training data  891
    Total number of columns in training data  12
    Total number of rows in test data  418
    Total number of columns in test data  11


We notice that the total number of columns in the training data (12) is one more tha test data (11). The former has the "survived" column whereas the latter does not since that is what we want the neural network to be able to predict.

#### Visualizations

There are a number of plots that we could do. Here are a few with brief comments about each.

Since '0' represents not surviving and '1' represents surviving, then we can see that females were much more likely to survice than males.

And now let's look at survival by class (first, second, or third).

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


```python
# Check for nulls in the train set
train.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
# Check for nulls in the test set
test.isnull().sum()
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64



So we have a decent amount missing data for *Age* in both the train and tests sets. There are also a couple from *Embarked* from the train set and one in *Fare* from the test set.

There are different ways to deal with missing data, which we'll turn to next.

#### Combining numeric features and converting categorical feasture

One possibility for dealing with the missing data is to
*    fill the nan values for *Age* and *Fare* with the median of each respective column
*   the embarked column with the most frequent value, which is *S* for Southampton

Once we clean up *Age* we can create a variable from it called *Child*, that allows us to look at the data as adults or children.


```python
# view the current status of the combined dataframe
combined.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 1309 entries, 0 to 417
    Data columns (total 13 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  1309 non-null   int64  
     1   Survived     891 non-null    float64
     2   Pclass       1309 non-null   object 
     3   Name         1309 non-null   object 
     4   Sex          1309 non-null   object 
     5   Age          1309 non-null   float64
     6   SibSp        1309 non-null   int64  
     7   Parch        1309 non-null   int64  
     8   Ticket       1309 non-null   object 
     9   Fare         1309 non-null   float64
     10  Cabin        295 non-null    object 
     11  Embarked     1309 non-null   object 
     12  Child        1309 non-null   int64  
    dtypes: float64(3), int64(4), object(6)
    memory usage: 143.2+ KB


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

Finally, we ned to drop the feature columns that we do not need for our model (`PassengerID`, `Name`, `Ticket`, and `Cabin`).

#### Resplitting the data and scaling the data.

We need to split our data back into the training and test sets now that we have done all but one aspect of the preprocessing.


```python
# Separate the data back into train and test sets
test = combined[combined['Survived'].isnull()].drop(['Survived'],axis=1)
train = combined[combined['Survived'].notnull()]
```

We need to get the training data into the predictors and the predicted variables, and scale the data.

## Neural Network Model

### Building the model

#### Define the model as a pipeline

Let's use the data science pipeline for our neural network model.


```python
# It will help to define our model in terms of a pipeline
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=10,kernel_initializer='uniform',activation='relu',input_dim=14))
    classifier.add(Dense(units=128,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
```

#### Fitting the optimal model and evaluating with `TensorBoaard`

#### `TensorBoard`

`TensorBoard` is `TensorFlow`'s visualization toolkit. It is a dashboard that provides visualization and tooling that is needed for machine learning experimentation. The code immediately below will allow us to use TensorBoard.

N.B. When we loaded the libraries, we loaded the TensorBoard notebook extension. (It is the last line of code in the first code chunk.)

#### Fitting the optimal model and evaluating with `TensorBoaard`

#### Results and Predictions

Continue to tweak your model until you are happy with the results based on model evaluation.

## Conclusion

Now that you have the `TensorBoard` to help you look at your model, you can better understand how to tweak your model.

We'll continue with this for the next lesson when we learn about model regularization.
