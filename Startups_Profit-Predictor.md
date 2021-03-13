## A Basic Machine Learning Model using Multivariate Regression

Here we will be looking at the various basic data cleaning techniques, visualization techiniques and then finally build a model, to predict the profits of new startup companies, using Multivariate Regression(a type of linear regression in which more than one independent variable (predictors) and more than one dependent variable (responses), are linearly related).

Multivariate Regression in a quadratic equation is of the form

y = m1.a1 + m2.a2 + m3.a3 + ..... + mn.an + a0   where '.' represents multiplication


```python
#importing the require packages

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
```


```python
#loading the dataset and checking it

df = pd.read_csv('50_startups.csv')
df.head()
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
      <th>R&amp;D Spend</th>
      <th>Administration</th>
      <th>Marketing Spend</th>
      <th>State</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>165349.20</td>
      <td>136897.80</td>
      <td>471784.10</td>
      <td>New York</td>
      <td>192261.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>162597.70</td>
      <td>151377.59</td>
      <td>443898.53</td>
      <td>California</td>
      <td>191792.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>153441.51</td>
      <td>101145.55</td>
      <td>407934.54</td>
      <td>Florida</td>
      <td>191050.39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>144372.41</td>
      <td>118671.85</td>
      <td>383199.62</td>
      <td>New York</td>
      <td>182901.99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>140143.50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Florida</td>
      <td>134307.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
#looking at some of the statical values to better understand the data

df.describe()
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
      <th>R&amp;D Spend</th>
      <th>Administration</th>
      <th>Marketing Spend</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>53.000000</td>
      <td>51.000000</td>
      <td>52.000000</td>
      <td>52.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>75374.625283</td>
      <td>120805.798627</td>
      <td>206471.538269</td>
      <td>113097.331923</td>
    </tr>
    <tr>
      <th>std</th>
      <td>46133.031203</td>
      <td>28001.877707</td>
      <td>123424.029117</td>
      <td>39902.857207</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>51283.140000</td>
      <td>0.000000</td>
      <td>14681.400000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>44069.950000</td>
      <td>102101.520000</td>
      <td>125324.665000</td>
      <td>90518.427500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>73994.560000</td>
      <td>122616.840000</td>
      <td>208157.655000</td>
      <td>108643.015000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>114523.610000</td>
      <td>144606.780000</td>
      <td>298932.675000</td>
      <td>142253.990000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>165349.200000</td>
      <td>182645.560000</td>
      <td>471784.100000</td>
      <td>192261.830000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Data Preprocessing

#Removing all the rows with nan values, if not they will affect our model's accuracy

print(df.isnull().sum())
df = df.dropna()          #inbuilt function of pandas.Dataframe to remove all the nan values
df.describe()
```

    R&D Spend          0
    Administration     2
    Marketing Spend    1
    State              1
    Profit             1
    dtype: int64
    




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
      <th>R&amp;D Spend</th>
      <th>Administration</th>
      <th>Marketing Spend</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>73721.615600</td>
      <td>121344.639600</td>
      <td>211025.097800</td>
      <td>112012.639200</td>
    </tr>
    <tr>
      <th>std</th>
      <td>45902.256482</td>
      <td>28017.802755</td>
      <td>122290.310726</td>
      <td>40306.180338</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>51283.140000</td>
      <td>0.000000</td>
      <td>14681.400000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39936.370000</td>
      <td>103730.875000</td>
      <td>129300.132500</td>
      <td>90138.902500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>73051.080000</td>
      <td>122699.795000</td>
      <td>212716.240000</td>
      <td>107978.190000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>101602.800000</td>
      <td>144842.180000</td>
      <td>299469.085000</td>
      <td>139765.977500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>165349.200000</td>
      <td>182645.560000</td>
      <td>471784.100000</td>
      <td>192261.830000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Changing all the zeros to nan so that we can fill them with relevant values

df = df.replace(0.0, np.nan)
df.isnull().sum()
```




    R&D Spend          2
    Administration     0
    Marketing Spend    3
    State              0
    Profit             0
    dtype: int64




```python
#Replacing all the zeros that were converted into nan before by the mean of those columns

df['R&D Spend'].fillna(df['R&D Spend'].mean(), inplace = True)
df['Marketing Spend'].fillna(df['Marketing Spend'].mean(), inplace = True)
df.isnull().sum()
```




    R&D Spend          0
    Administration     0
    Marketing Spend    0
    State              0
    Profit             0
    dtype: int64




```python
#Checking the correlation matrix of the data

df.corr()
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
      <th>R&amp;D Spend</th>
      <th>Administration</th>
      <th>Marketing Spend</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R&amp;D Spend</th>
      <td>1.000000</td>
      <td>0.268979</td>
      <td>0.666533</td>
      <td>0.881354</td>
    </tr>
    <tr>
      <th>Administration</th>
      <td>0.268979</td>
      <td>1.000000</td>
      <td>-0.070590</td>
      <td>0.200717</td>
    </tr>
    <tr>
      <th>Marketing Spend</th>
      <td>0.666533</td>
      <td>-0.070590</td>
      <td>1.000000</td>
      <td>0.693088</td>
    </tr>
    <tr>
      <th>Profit</th>
      <td>0.881354</td>
      <td>0.200717</td>
      <td>0.693088</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Checking the number of startups in each State

Startups_in_New_York = df.loc[df['State']=='New York']
Startups_in_California = df.loc[df['State']=='California']
Startups_in_Florida = df.loc[df['State']=='Florida']
print("Number of startups in New York", len(Startups_in_New_York))
print("Number of startups in California", len(Startups_in_California))
print("Number of startups in Florida", len(Startups_in_Florida))
```

    Number of startups in New York 17
    Number of startups in California 17
    Number of startups in Florida 16
    


```python
#Data Visualization

#Here we draw the plot bar graph of the spendings by the startups organised by States

fig = plt.figure()
fig, ax = plt.subplots()
cities = [i for i in df['State'].unique()]

#Get the total of the profits and spendings of the startups in each State

Profits = [round(Startups_in_New_York['Profit'].sum()), round(Startups_in_California['Profit'].sum()), \
           round(Startups_in_Florida['Profit'].sum())]
RD_Spend = [round(Startups_in_New_York['R&D Spend'].sum()), round(Startups_in_California['R&D Spend'].sum()), \
            round(Startups_in_Florida['R&D Spend'].sum())]
Administration = [round(Startups_in_New_York['Administration'].sum()), round(Startups_in_California['Administration'].sum()), \
                  round(Startups_in_Florida['Administration'].sum())]
Marketing = [round(Startups_in_New_York['Marketing Spend'].sum()), round(Startups_in_California['Marketing Spend'].sum()), \
             round(Startups_in_Florida['Marketing Spend'].sum())]

New_york = [RD_Spend[0], Administration[0], Marketing[0], Profits[0]]
California = [RD_Spend[1], Administration[1], Marketing[1], Profits[1]]
Florida = [RD_Spend[2], Administration[2], Marketing[2], Profits[2]]

#Draw the bar graph and set proper labels, legend, etc.,.

labels = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profits']
X = np.arange(len(labels))
ax.set_title('Expenditure and Profits by States')
ax.bar(X - 0.25, New_york, color = 'b', width = 0.25, label='New York')
ax.bar(X + 0.00, California, color = 'g', width = 0.25, label='California')
ax.bar(X + 0.25, Florida, color = 'r', width = 0.25, label='Florida')

ax.set_ylabel('Expenditure (order of 10^6)')
ax.set_xticks(X)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
```


    <Figure size 432x288 with 0 Axes>



    
![png](output_11_1.png)
    



```python
#Draw the heat map of the data for analysis

plt.figure()
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.show()
```


    
![png](output_12_0.png)
    



```python
#Draw the graphs between each independant variable and dependant variable for analysis

sns.lmplot(x='R&D Spend', y='Profit', data=df).fig.suptitle("Plot between R&d Spendings and Profit", fontsize=12)
```




    Text(0.5, 0.98, 'Plot between R&d Spendings and Profit')




    
![png](output_13_1.png)
    



```python
fig = sns.lmplot(x='Administration', y='Profit', data=df)
fig = fig.fig
fig.suptitle("Plot between Administration Spendings and Profit", fontsize=12)
```




    Text(0.5, 0.98, 'Plot between Administration Spendings and Profit')




    
![png](output_14_1.png)
    



```python
sns.lmplot(x='Marketing Spend', y='Profit', data=df).fig.suptitle("Plot between Marketing Spendings and Profit", fontsize=12)
```




    Text(0.5, 0.98, 'Plot between Marketing Spendings and Profit')




    
![png](output_15_1.png)
    


#### Some intresting conclusions that can be observed are

1. As the spendings increase the profit obtained is also seen to increase from all the graphs but the rate of increase is seen to differ in each case with the R&D Spendings showing the highest slope followed by Marketing Spendings and Administrative spendings.

2. The above observation is also confirmed by the heatmap plotted above where the correlation values of the variables are seen. We can see that R&D Spendings has highest correlation with the profit while next is Marketing and followed by Administration Spending.

3. From the bar graph we can see that the startups in New York State spend the most and earn the most on average, than the startups in other states.


```python
#Data Preprocessing

#Perform one-hot encoding to reduce the ordinal data of State to a form that can be used to build the model

one_hot = pd.get_dummies(df['State'])
df = df.drop('State', axis=1)
df = df.join(one_hot)
prof = df['Profit']
df = df.drop('Profit', axis=1).join(prof)
df.head()
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
      <th>R&amp;D Spend</th>
      <th>Administration</th>
      <th>Marketing Spend</th>
      <th>California</th>
      <th>Florida</th>
      <th>New York</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>165349.20</td>
      <td>136897.80</td>
      <td>471784.10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>192261.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>162597.70</td>
      <td>151377.59</td>
      <td>443898.53</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>191792.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>153441.51</td>
      <td>101145.55</td>
      <td>407934.54</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>191050.39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>144372.41</td>
      <td>118671.85</td>
      <td>383199.62</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>182901.99</td>
    </tr>
    <tr>
      <th>5</th>
      <td>142107.34</td>
      <td>91391.77</td>
      <td>366168.42</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>166187.94</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Check the correlation values

df.corr()
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
      <th>R&amp;D Spend</th>
      <th>Administration</th>
      <th>Marketing Spend</th>
      <th>California</th>
      <th>Florida</th>
      <th>New York</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R&amp;D Spend</th>
      <td>1.000000</td>
      <td>0.268979</td>
      <td>0.666533</td>
      <td>-0.051912</td>
      <td>0.062887</td>
      <td>-0.010015</td>
      <td>0.881354</td>
    </tr>
    <tr>
      <th>Administration</th>
      <td>0.268979</td>
      <td>1.000000</td>
      <td>-0.070590</td>
      <td>-0.015478</td>
      <td>0.010493</td>
      <td>0.005145</td>
      <td>0.200717</td>
    </tr>
    <tr>
      <th>Marketing Spend</th>
      <td>0.666533</td>
      <td>-0.070590</td>
      <td>1.000000</td>
      <td>-0.189842</td>
      <td>0.144084</td>
      <td>0.047958</td>
      <td>0.693088</td>
    </tr>
    <tr>
      <th>California</th>
      <td>-0.051912</td>
      <td>-0.015478</td>
      <td>-0.189842</td>
      <td>1.000000</td>
      <td>-0.492366</td>
      <td>-0.515152</td>
      <td>-0.145837</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>0.062887</td>
      <td>0.010493</td>
      <td>0.144084</td>
      <td>-0.492366</td>
      <td>1.000000</td>
      <td>-0.492366</td>
      <td>0.116244</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>-0.010015</td>
      <td>0.005145</td>
      <td>0.047958</td>
      <td>-0.515152</td>
      <td>-0.492366</td>
      <td>1.000000</td>
      <td>0.031368</td>
    </tr>
    <tr>
      <th>Profit</th>
      <td>0.881354</td>
      <td>0.200717</td>
      <td>0.693088</td>
      <td>-0.145837</td>
      <td>0.116244</td>
      <td>0.031368</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Remove the column of New-York to prevent 'Dummy Variable Trap', which is caused due to the redundant data that
#was obtained due to the one hot encoding performed on States column

df = df.drop('New York', axis=1)
df.head()
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
      <th>R&amp;D Spend</th>
      <th>Administration</th>
      <th>Marketing Spend</th>
      <th>California</th>
      <th>Florida</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>165349.20</td>
      <td>136897.80</td>
      <td>471784.10</td>
      <td>0</td>
      <td>0</td>
      <td>192261.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>162597.70</td>
      <td>151377.59</td>
      <td>443898.53</td>
      <td>1</td>
      <td>0</td>
      <td>191792.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>153441.51</td>
      <td>101145.55</td>
      <td>407934.54</td>
      <td>0</td>
      <td>1</td>
      <td>191050.39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>144372.41</td>
      <td>118671.85</td>
      <td>383199.62</td>
      <td>0</td>
      <td>0</td>
      <td>182901.99</td>
    </tr>
    <tr>
      <th>5</th>
      <td>142107.34</td>
      <td>91391.77</td>
      <td>366168.42</td>
      <td>0</td>
      <td>1</td>
      <td>166187.94</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Building the model

#Split the data into train and test data(80% and 20%) 

X = df.iloc[:,:-1].values
y = df.iloc[:,5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
#Train the model with the train dataset

regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)
```

    33616.46584790526
    [ 6.45209703e-01  1.00425490e-01  8.15150660e-02 -6.68132451e+03
     -1.81773441e+03]
    


```python
#Run the model trained on the test values to obtain the predicted values

predictions = regressor.predict(X_test)
```


```python
#Plot the values obtained by predicting using the model trained and the actual test values to evaluate our model's performance

plt.grid(True)
plt.plot(predictions, color = 'red', label = 'Predicted')
plt.plot(y_test, color = 'blue', label = 'y_test')
plt.legend()
plt.ylabel('Profit')
plt.show()
```


    
![png](output_23_0.png)
    


#### Some conclusions are

1. The predicted values are close to the actual values which signifies that our model is a reasonable one and can be employed for real world problems.

2. Huge variation can be seen for a few values in the start but then that can be attributed to the small size of our dataset, the prediction can be made even more accurate if we have more data to train our model.
