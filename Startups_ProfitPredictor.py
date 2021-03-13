# importing the require packages

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

# loading the dataset and checking it

df = pd.read_csv('50_startups.csv')
df.head()

# looking at some of the statical values to better understand the data

df.describe()

# Data Preprocessing

# Removing all the rows with nan values, if not they will affect our model's accuracy

print(df.isnull().sum())
df = df.dropna()          # inbuilt function of pandas.Dataframe to remove all the nan values
df.describe()

# Changing all the zeros to nan so that we can fill them with relevant values

df = df.replace(0.0, np.nan)
df.isnull().sum()

# Replacing all the zeros that were converted into nan before by the mean of those columns

df['R&D Spend'].fillna(df['R&D Spend'].mean(), inplace = True)
df['Marketing Spend'].fillna(df['Marketing Spend'].mean(), inplace = True)
df.isnull().sum()

# Checking the correlation matrix of the data

df.corr()

# Checking the number of startups in each State

Startups_in_New_York = df.loc[df['State']=='New York']
Startups_in_California = df.loc[df['State']=='California']
Startups_in_Florida = df.loc[df['State']=='Florida']
print("Number of startups in New York", len(Startups_in_New_York))
print("Number of startups in California", len(Startups_in_California))
print("Number of startups in Florida", len(Startups_in_Florida))

# Data Visualization

# Here we draw the plot bar graph of the spendings by the startups organised by States

fig = plt.figure()
fig, ax = plt.subplots()
cities = [i for i in df['State'].unique()]

# Get the total of the profits and spendings of the startups in each State

Profits = [round(Startups_in_New_York['Profit'].sum()), round(Startups_in_California['Profit'].sum()),            round(Startups_in_Florida['Profit'].sum())]
RD_Spend = [round(Startups_in_New_York['R&D Spend'].sum()), round(Startups_in_California['R&D Spend'].sum()),             round(Startups_in_Florida['R&D Spend'].sum())]
Administration = [round(Startups_in_New_York['Administration'].sum()), round(Startups_in_California['Administration'].sum()),                   round(Startups_in_Florida['Administration'].sum())]
Marketing = [round(Startups_in_New_York['Marketing Spend'].sum()), round(Startups_in_California['Marketing Spend'].sum()),              round(Startups_in_Florida['Marketing Spend'].sum())]

New_york = [RD_Spend[0], Administration[0], Marketing[0], Profits[0]]
California = [RD_Spend[1], Administration[1], Marketing[1], Profits[1]]
Florida = [RD_Spend[2], Administration[2], Marketing[2], Profits[2]]

# Draw the bar graph and set proper labels, legend, etc.,.

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

# Draw the heat map of the data for analysis

plt.figure()
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.show()

# Draw the graphs between each independant variable and dependant variable for analysis

sns.lmplot(x='R&D Spend', y='Profit', data=df).fig.suptitle("Plot between R&d Spendings and Profit", fontsize=12)

fig = sns.lmplot(x='Administration', y='Profit', data=df)
fig = fig.fig
fig.suptitle("Plot between Administration Spendings and Profit", fontsize=12)

sns.lmplot(x='Marketing Spend', y='Profit', data=df).fig.suptitle("Plot between Marketing Spendings and Profit", fontsize=12)

# Data Preprocessing

# Perform one-hot encoding to reduce the ordinal data of State to a form that can be used to build the model

one_hot = pd.get_dummies(df['State'])
df = df.drop('State', axis=1)
df = df.join(one_hot)
prof = df['Profit']
df = df.drop('Profit', axis=1).join(prof)
df.head()

# Check the correlation values

df.corr()

# Remove the column of New-York to prevent 'Dummy Variable Trap', which is caused due to the redundant data that
# was obtained due to the one hot encoding performed on States column

df = df.drop('New York', axis=1)
df.head()

# Building the model

# Split the data into train and test data(80% and 20%)

X = df.iloc[:,:-1].values
y = df.iloc[:,5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Train the model with the train dataset

regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)

# Run the model trained on the test values to obtain the predicted values

predictions = regressor.predict(X_test)

# Plot the values obtained by predicting using the model trained and the actual test values to evaluate our model's
# performance

plt.grid(True)
plt.plot(predictions, color = 'red', label = 'Predicted')
plt.plot(y_test, color = 'blue', label = 'y_test')
plt.legend()
plt.ylabel('Profit')
plt.show()