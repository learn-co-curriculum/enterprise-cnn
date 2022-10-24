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


```python
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://upload.wikimedia.org/wikipedia/commons/c/c0/Titanic_Belfast_HDR.jpg", width=400, height=400)
```




<img src="https://upload.wikimedia.org/wikipedia/commons/c/c0/Titanic_Belfast_HDR.jpg" width="400" height="400"/>



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


```python
# Load the germane libraries

import pandas as pd
import numpy as np
import seaborn as sns 
from pandas._libs.tslibs import timestamps
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras 
from keras import models
from keras.layers import Dense
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.wrappers.scikit_learn import KerasClassifier

# Load the TensorBoard notebook extension and related libraries
%load_ext tensorboard
import datetime
```

### Loading the data


```python
# Load the data

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# We need to do this for when we mamke our predictions from the test data at the end
ids = test[['PassengerId']]
```

## EDA and Preprocessing

### Exploratory Data Analysis

It is your choice how much or how little EDA that you perform. But you should do enough EDA that you feel comfortable with the data and what you'll need to do to make it so that you can run a neural network on it.

What follows in some suggestions for EDA. This is by no means exhaustive, but just a sampling of what you can do. It is important for each EDA code chunk that you run that you reflect on what the output is telling you.

#### Overview


```python
# Print the first rows of the train data
train.head()
```





  <div id="df-2e617e68-d460-47d1-b63f-f7f163e97132">
    <div class="colab-df-container">
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-2e617e68-d460-47d1-b63f-f7f163e97132')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2e617e68-d460-47d1-b63f-f7f163e97132 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2e617e68-d460-47d1-b63f-f7f163e97132');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
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





  <div id="df-a024cd02-17e3-4a73-ac85-af924be7835a">
    <div class="colab-df-container">
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-a024cd02-17e3-4a73-ac85-af924be7835a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a024cd02-17e3-4a73-ac85-af924be7835a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a024cd02-17e3-4a73-ac85-af924be7835a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
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


```python
# Historgram with survival and sex
sns.histplot(x='Survived', hue = 'Sex', stat = 'percent', data = train)
plt.show()
```


    
![png](output_27_0.png)
    


Since '0' represents not surviving and '1' represents surviving, then we can see that females were much more likely to survice than males.

And now let's look at survival by class (frist, second, or third).


```python
# Historgram with survival and class
sns.histplot(x="Survived", hue = 'Pclass', stat = 'percent',  data = train)
plt.show()
```


    
![png](output_29_0.png)
    


While the differences between second and third class doesn't seem very diffrent, those in third class seem more likely to die.

We can see this more clealry in the next graph.


```python
# FaceGrid of Class and Survived with Age
grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()
```


    
![png](output_31_0.png)
    


What conclusions can you draw from the above grid?

Finally, let's look at a heatmap to see which columns are correlated.


```python
# Heatmap
corr_train = train.corr()
sns.heatmap(corr_train)
plt.show()
```


    
![png](output_33_0.png)
    


What columns are correlated? Does this surprise you? Which columns are not correlated? Does this surprise you?

#### Data Analysis

Again there are a myriad of data analysis that we can do, but let's show some examples of what we may want to look it. Again, do as many as you think are appropriate and you should interpret the results explicitly.


```python
# Surival rate by sex
((train.groupby(['Sex','Survived']).Survived.count() * 100) / train.groupby('Sex').Survived.count())
```




    Sex     Survived
    female  0           25.796178
            1           74.203822
    male    0           81.109185
            1           18.890815
    Name: Survived, dtype: float64



Females had about 74% chance of survival where as men had only about a 19%.

Now let's do the same, but with class as well.


```python
# Survival rate by class
(train.groupby(['Pclass','Survived', 'Sex']).Survived.count() * 100) / train.groupby('Pclass').Survived.count()
```




    Pclass  Survived  Sex   
    1       0         female     1.388889
                      male      35.648148
            1         female    42.129630
                      male      20.833333
    2       0         female     3.260870
                      male      49.456522
            1         female    38.043478
                      male       9.239130
    3       0         female    14.663951
                      male      61.099796
            1         female    14.663951
                      male       9.572301
    Name: Survived, dtype: float64



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
# Check for nulls in teh test set
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


```python
# Performing preprocessing on the train and test data will be more effecient if we combine the two date sets.
combined = pd.concat([train, test], axis=0, sort=False)
```

There are different ways to deal with missing data, which we'll turn to next.

#### Combining numeric features and converting categorical feasture

One possibility for dealing with the missing data is to
*    fill the nan values for *Age* and *Fare* with the median of each respective column
*   the embarked column with the most frequent value, which is *S* for Southampton

Once we clean up *Age* we can create a variable from it called *Child*, that allows us to look at the data as adults or children.


```python
#Age column
combined['Age'].fillna(combined['Age'].median(),inplace=True) # Age

# Embarked column
combined['Embarked'].fillna(combined['Embarked'].value_counts().index[0], inplace=True) # Embarked
combined['Fare'].fillna(combined['Fare'].median(),inplace=True)

# Class column
d = {1:'1st',2:'2nd',3:'3rd'} #Pclass
combined['Pclass'] = combined['Pclass'].map(d) #Pclass

# Making Age into adult (1) and child (0)
combined['Child'] = combined['Age'].apply(lambda age: 1 if age>=18 else 0) 
```


```python
combined.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 0 to 417
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


```python
# Break up the string that has the title and names
combined['Title'] = combined['Name'].str.split('.').str.get(0)  # output : 'Futrelle, Mrs'
combined['Title'] = combined['Title'].str.split(',').str.get(1) # output : 'Mrs '
combined['Title'] = combined['Title'].str.strip()               # output : 'Mrs'
combined.groupby('Title').count()

# Replace the French titles with Enlgish
french_titles = ['Don', 'Dona', 'Mme', 'Ms', 'Mra','Mlle']
english_titles = ['Mr', 'Mrs','Mrs','Mrs','Mrs','Miss']
for i in range(len(french_titles)):
    for j in range(len(english_titles)):
        if i == j:
            combined['Title'] = combined['Title'].str.replace(french_titles[i],english_titles[j])

# Seperate the titles into "major" and "others", the latter would be, e.g., Reverend
major_titles = ['Mr','Mrs','Miss','Master']
combined['Title'] = combined['Title'].apply(lambda title: title if title in major_titles else 'Others')
```

Finally, we ned to drop the feature columns that we do not need for our model.


```python
#Dropping the Irrelevant Columns
combined.drop(['PassengerId','Name','Ticket','Cabin'], 1, inplace=True)

# Getting Dummy Variables and Dropping the Original Categorical Variables
categorical_vars = combined[['Pclass','Sex','Embarked','Title','Child']] # Get Dummies of Categorical Variables 
dummies = pd.get_dummies(categorical_vars,drop_first=True)
combined = combined.drop(['Pclass','Sex','Embarked','Title','Child'],axis=1)
combined = pd.concat([combined, dummies],axis=1)
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      


#### Resplitting the data and scaling the data.

We need to split our data back into the training and test sets now that we have done all but one aspect of the preprocessing.


```python
# Separating the data back into train and test sets
test = combined[combined['Survived'].isnull()].drop(['Survived'],axis=1)
train = combined[combined['Survived'].notnull()]
```

We need to get the training data into the predictors and the predicted variables, and scale the data.


```python
# Training
X_train = train.drop(['Survived'],1)
y_train = train['Survived']

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
test = sc.fit_transform(test)
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
      


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


```python
# Clear out any prior log data.
!rm -rf logs
# Be careful not to run this command if already have trained your model and you want to use TensorBoard.

# Sets up a timestamped log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(log_dir)


# The callback function, which will be called in the fit()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
```

#### Fitting the optimal model and evaluating with `TensorBoaard`


```python
# Using KerasClassifier

classifier = KerasClassifier(build_fn = build_classifier,
                             optimizer='Adam',
                             batch_size=10,
                             epochs=16)

# Fit the model with the tensorboard_callback
classifier.fit(X_train,
               y_train,
               verbose=1,
               callbacks=[tensorboard_callback])


# Warning: If verbose = 0 (silent) or 2 (one line per epoch), then on TensorBoard's Graphs tab there will be an error.
# The other tabs in TensorBoard will still be function, but if you want the graphs then verbose needs to be 1 (progress bar).
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:6: DeprecationWarning: KerasClassifier is deprecated, use Sci-Keras (https://github.com/adriangb/scikeras) instead. See https://www.adriangb.com/scikeras/stable/migration.html for help migrating.
      


    Epoch 1/16
    90/90 [==============================] - 2s 6ms/step - loss: 0.6402 - accuracy: 0.6611
    Epoch 2/16
    90/90 [==============================] - 1s 6ms/step - loss: 0.4650 - accuracy: 0.8047
    Epoch 3/16
    90/90 [==============================] - 1s 6ms/step - loss: 0.4232 - accuracy: 0.8238
    Epoch 4/16
    90/90 [==============================] - 1s 7ms/step - loss: 0.4127 - accuracy: 0.8339
    Epoch 5/16
    90/90 [==============================] - 1s 7ms/step - loss: 0.4068 - accuracy: 0.8283
    Epoch 6/16
    90/90 [==============================] - 0s 5ms/step - loss: 0.4006 - accuracy: 0.8272
    Epoch 7/16
    90/90 [==============================] - 1s 6ms/step - loss: 0.3987 - accuracy: 0.8316
    Epoch 8/16
    90/90 [==============================] - 1s 7ms/step - loss: 0.3970 - accuracy: 0.8328
    Epoch 9/16
    90/90 [==============================] - 0s 5ms/step - loss: 0.3939 - accuracy: 0.8328
    Epoch 10/16
    90/90 [==============================] - 0s 5ms/step - loss: 0.3914 - accuracy: 0.8418
    Epoch 11/16
    90/90 [==============================] - 1s 10ms/step - loss: 0.3870 - accuracy: 0.8440
    Epoch 12/16
    90/90 [==============================] - 1s 9ms/step - loss: 0.3903 - accuracy: 0.8361
    Epoch 13/16
    90/90 [==============================] - 1s 8ms/step - loss: 0.3872 - accuracy: 0.8406
    Epoch 14/16
    90/90 [==============================] - 1s 6ms/step - loss: 0.3861 - accuracy: 0.8418
    Epoch 15/16
    90/90 [==============================] - 1s 6ms/step - loss: 0.3853 - accuracy: 0.8474
    Epoch 16/16
    90/90 [==============================] - 1s 6ms/step - loss: 0.3848 - accuracy: 0.8418





    <keras.callbacks.History at 0x7f17289d2190>




```python
# Call TensorBoard
%tensorboard --logdir logs/fit
```


    <IPython.core.display.Javascript object>


#### Results and Predictions


```python
# results
preds = classifier.predict(test)
results = ids.assign(Survived=preds)
results['Survived'] = results['Survived'].apply(lambda row: 1 if row > 0.5 else 0)
results.to_csv('titanic_submission.csv',index=False)
results.head(20)
```

    14/14 [==============================] - 0s 2ms/step






  <div id="df-0ff333d8-ef32-40f0-b73a-04413121a248">
    <div class="colab-df-container">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>897</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>898</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>899</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>900</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>901</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>902</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>903</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>904</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>905</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>906</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>907</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>908</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>909</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>910</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>911</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0ff333d8-ef32-40f0-b73a-04413121a248')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0ff333d8-ef32-40f0-b73a-04413121a248 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0ff333d8-ef32-40f0-b73a-04413121a248');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Continue to tweak your model until you are happy with the results based on model evaluation.

## Conclusion

Now that you have the `TensorBoard` to help you look at your model, you can better understand how to tweak your model.

We'll continue with this for the next lesson when we learn about model regularization.
