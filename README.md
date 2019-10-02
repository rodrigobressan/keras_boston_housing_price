
# Keras 101: A simple Neural Network for House Pricing regression

In this post, we will be covering some basics of data exploration and building a model with Keras in order to help us on predicting the selling price of a given house in the Boston (MA) area. As an application of this model in the real world, you can think about being a real state agent looking for a tool to help you on your day-to-day duties, which for me, at least, sounds pretty good when compared to just gut-estimation.

For this exercise, we will be using the [Plotly](https://plot.ly/python/) library instead of the good ol' fashioned matplotlib, due to having more interactive plots, which for sure help in understanding the data. We will also use the [Scikit-Learn](https://scikit-learn.org/stable/) and [Keras](https://keras.io/) for building the models, [Pandas](https://pandas.pydata.org/) library to manipulate our data and the [SHAP library](https://github.com/slundberg/shap) to generate explanations for our trained model.

### Importing the dataset

In this example, we wil be using the sklearn.datasets module, which contains the Boston dataset. You could also use the keras.datasets module, but this one does not contain the labels of the features, so we decided to use scikits one. Let's also convert it to a Pandas DataFrame and print it's head.

```python
from sklearn.datasets import load_boston
import pandas as pd

boston_dataset = load_boston()

df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
df['MEDV'] = boston_dataset.target

df.head(n=10)
```

This should output the following dataframe:

|    |    CRIM |   ZN |   INDUS |   CHAS |   NOX |    RM |   AGE |    DIS |   RAD |   TAX |   PTRATIO |      B |   LSTAT |   MEDV |
|---:|--------:|-----:|--------:|-------:|------:|------:|------:|-------:|------:|------:|----------:|-------:|--------:|-------:|
|  0 | 0.00632 |   18 |    2.31 |      0 | 0.538 | 6.575 |  65.2 | 4.09   |     1 |   296 |      15.3 | 396.9  |    4.98 |   24   |
|  1 | 0.02731 |    0 |    7.07 |      0 | 0.469 | 6.421 |  78.9 | 4.9671 |     2 |   242 |      17.8 | 396.9  |    9.14 |   21.6 |
|  2 | 0.02729 |    0 |    7.07 |      0 | 0.469 | 7.185 |  61.1 | 4.9671 |     2 |   242 |      17.8 | 392.83 |    4.03 |   34.7 |
|  3 | 0.03237 |    0 |    2.18 |      0 | 0.458 | 6.998 |  45.8 | 6.0622 |     3 |   222 |      18.7 | 394.63 |    2.94 |   33.4 |
|  4 | 0.06905 |    0 |    2.18 |      0 | 0.458 | 7.147 |  54.2 | 6.0622 |     3 |   222 |      18.7 | 396.9  |    5.33 |   36.2 |

### Exploratory Data Analysis

Making yourself comfortable and familiar with your dataset is a fundamental step to help you comprehend your data and draw better conclusions and explanations from your results.

Initially, let's plot a few box plots, which will help us to better visualizate anomalies and/or outliers in data distribution. If you are confused about what is a box plot and how it can help us to better visualizate the distribution of our data, here is a brief description from Ross (1977):

> In descriptive statistics, a box plot is a method for graphically depicting groups of numerical data through their quartiles. Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability outside the upper and lower quartiles, hence the terms box-and-whisker plot and box-and-whisker diagram. Outliers may be plotted as individual points.

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

total_items = len(df.columns)
items_per_row = 3
total_rows = math.ceil(total_items / items_per_row)

fig = make_subplots(rows=total_rows, cols=items_per_row)

cur_row = 1
cur_col = 1

for index, column in enumerate(df.columns):
    fig.add_trace(go.Box(y=df[column], name=column), row=cur_row, col=cur_col)
    
    if cur_col % items_per_row == 0:
        cur_col = 1
        cur_row = cur_row + 1
    else:
        cur_col = cur_col + 1
    

fig.update_layout(height=1000, width=550,  showlegend=False)
fig.show()
```


<div style="width: 100%; text-align: center">
    <img style='width: 70%; object-fit: contain' src="/images/output_3_0.png"/>
</div>

These results do corroborate our initial assumptions about having outliers in some columns. Let's also plot some scatter plots for each feature and the target variable, as well as their intercept lines:

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import numpy as np

total_items = len(df.columns)
items_per_row = 3
total_rows = math.ceil(total_items / items_per_row)

fig = make_subplots(rows=total_rows, cols=items_per_row, subplot_titles=df.columns)

cur_row = 1
cur_col = 1

for index, column in enumerate(df.columns):
    fig.add_trace(go.Scattergl(x=df[column], 
                            y=df['MEDV'], 
                            mode="markers", 
                            marker=dict(size=3)), 
                  row=cur_row, 
                  col=cur_col)
    
    intercept = np.poly1d(np.polyfit(df[column], df['MEDV'], 1))(np.unique(df[column]))
    
    fig.add_trace(go.Scatter(x=np.unique(df[column]), 
                             y=intercept, 
                             line=dict(color='red', width=1)), 
                  row=cur_row, 
                  col=cur_col)
    
    if cur_col % items_per_row == 0:
        cur_col = 1
        cur_row = cur_row + 1
    else:
        cur_col = cur_col + 1
    

fig.update_layout(height=1000, width=550, showlegend=False)
fig.show()
```

<div style="width: 100%; text-align: center">
    <img style='width: 70%; object-fit: contain' src="/images/output_5_0.png"/>
</div>

From this initial data exploration, we can have two major conclusions:

- There is a strong linear correlation between the RM (average number of rooms per dwelling) and LSTAT (% lower status of the population) with the target variable, being the RM a positive and the LSTAT a negative correlation.
- There are some records containing outliers, which we could preprocess in order to input our model with more normalized data.

### Data preprocessing

Before we proceed into any data preprocessing, it's important to split our data into training and test sets. We should not apply any kind of preprocessing into our data without taking into account that we should not leak information from our test set. For this step, we can use the *train_test_split* method from scikit-learn. In this case, we will use a split of 70% of the data for training and 30% for testing. We also set a random_state seed, in order to allow reprocibility.

```python
from sklearn.model_selection import train_test_split

X = df.loc[:, df.columns != 'MEDV']
y = df.loc[:, df.columns == 'MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
```

In order to provide a standardized input to our neural network, we need the perform the normalization of our dataset. This can be seen as an step to reduce the differences in scale that may arise from the existent features. We perform this normalization by subtracting the mean from our data and dividing it by the standard deviation. **One more time,  this normalization should only be performed by using the mean and standard deviation from the training set, in order to avoid any information leak from the test set.**

```python
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
```

### Build our model

Due to the small amount of presented data in this dataset, we must be careful to not create an overly complex model, which could lead to overfitting our data. For this, we are going to adopt an architecture based on two Dense layers, the first with 128 and the second with 64 neurons, both using a ReLU activation function. A dense layer with a linear activation will be used as output layer.

In order to allow us to know if our model is properly learning, we will use a mean squared error loss function and to report the performance of it we will adopt the mean average error metric.

By using the summary method from Keras, we can see that we have a total of 10,113 parameters, which is acceptable for us.

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(128, input_shape=(13, ), activation='relu', name='dense_1'))
model.add(Dense(64, activation='relu', name='dense_2'))
model.add(Dense(1, activation='linear', name='dense_output'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
```

### Train our model

This step is pretty straightforward: fit our model with both our features and their labels, for a total amount of 100 epochs, separating 5% of the samples (18 records) as validation set.

```python
history = model.fit(X_train, y_train, epochs=100, validation_split=0.05)
```

By plotting both loss and mean average error, we can see that our model was capable of learning patterns in our data without overfitting taking place (as shown by the validation set curves):

```python
fig = go.Figure()
fig.add_trace(go.Scattergl(y=history.history['loss'],
                    name='Train'))

fig.add_trace(go.Scattergl(y=history.history['val_loss'],
                    name='Valid'))


fig.update_layout(height=500, width=700,
                  xaxis_title='Epoch',
                  yaxis_title='Loss')

fig.show()
```

<div style="width: 100%; text-align: center">
    <img style='width: 70%; object-fit: contain' src="/images/output_15_0.png"/>
</div>

```python
fig = go.Figure()
fig.add_trace(go.Scattergl(y=history.history['mean_absolute_error'],
                    name='Train'))

fig.add_trace(go.Scattergl(y=history.history['val_mean_absolute_error'],
                    name='Valid'))


fig.update_layout(height=500, width=700,
                  xaxis_title='Epoch',
                  yaxis_title='Mean Absolute Error')

fig.show()
```

<div style="width: 100%; text-align: center">
    <img style='width: 70%; object-fit: contain' src="/images/output_16_0.png"/>
</div>

### Evaluate our model

```python
mse_nn, mae_nn = model.evaluate(X_test, y_test)

print('Mean squared error on test data: ', mse_nn)
print('Mean absolute error on test data: ', mae_nn)
```

Output:
```
152/152 [==============================] - 0s 60us/step
Mean squared error on test data:  17.429732523466413
Mean absolute error on test data:  2.6727954964888725
```
### Comparison with traditional approaches

First let's try with a simple algorithm, the Linear Regression:

```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print('Mean squared error on test data: ', mse_lr)
print('Mean absolute error on test data: ', mae_lr)
```

```
Mean squared error on test data: 28.40585481050824
Mean absolute error on test data: 3.6913626771162575
```

And now with a Decision Tree:

```python
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_tree)
mae_dt = mean_absolute_error(y_test, y_pred_tree)

print('Mean squared error on test data: ', mse_dt)
print('Mean absolute error on test data: ', mae_dt)
```

```
Mean squared error on test data:  17.830657894736845
Mean absolute error on test data:  2.755263157894737
```

### Opening the Black Box (a.k.a. Explaining our Model)

Sometimes just a good result is enough for most of the people, but there are scenarios where we need to explain what are the major components used by our model to perform its prediction. For this task, we can rely on the SHAP library, which easily allows us to create a summary of our features and its impact on the model output. I won't dive deep into the details of SHAP, but if you are intered on it, you can check their [github page](https://github.com/slundberg/shap) or even give a look at its [paper](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)].

```python
import shap
shap.initjs()

explainer = shap.DeepExplainer(model, X_train[:100].values)
shap_values = explainer.shap_values(X_test[:100].values)

shap.summary_plot(shap_values, X_test, plot_type='bar')
```
<div style="width: 100%; text-align: center">
    <img style='width: 70%; object-fit: contain' src="/images/output_25_1.png"/>
</div>

From this simple plot, we can see that the major features that have an impact on the model output are:

- LSTAT: % lower status of the population
- RM: average number of rooms per dwelling
- RAD: index of accessibility to radial highways
- DIS: weighted distances to five Boston employment centres
- NOX: nitric oxides concentration (parts per 10 million) - this may more likely be correlated with greenness of the area
- CRIM: per capita crime rate by town

From this plot, we can clearly corroborate our initial EDA analysis in which we point out the LSTAT and RM features as having a high correlation with the model outcome.

### Conclusions

In this post, we have showed that by using a Neural Network, we can easily outperform traditional Machine Learning methods by a good margin. We also show that, even when using a more complex model, when compared to other techniques, we can still explain the outcomes of our model by using the SHAP library.

Furthermore, we need to have in mind that the explored dataset can be somehow outdated, and some feature engineering (such as correcting prices for inflaction) could be performed in order to better reflect current scenarios.

The Jupyter notebook for this post can be found [here](/notebook.ipynb).

#### References
Boston Dataset: [https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)

Plotly: [https://plot.ly/python/](https://plot.ly/python/)

ScikitLearn: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

Keras: [https://keras.io/](https://keras.io/)

Pandas: [https://pandas.pydata.org/](https://pandas.pydata.org/)

SHAP Project Page: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)

SHAP Paper: [https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)

Introduction to probability and statistics for engineers and scientists. [https://www.amazon.com.br/dp/B007ZBZN9U/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1](https://www.amazon.com.br/dp/B007ZBZN9U/ref=dp-kindle-redirect?_encoding=UTF8&btkr=1)

