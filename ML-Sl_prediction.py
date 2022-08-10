from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

## please download the dataset from here : https://docs.google.com/spreadsheets/d/1b50kg2PBGXD0Op3VXfdPtdEP-QwyMDYc/edit?usp=sharing&ouid=101206169502223149951&rtpof=true&sd=true

df = pd.read_csv("Downloads/new_dataset.xlsx - Sheet1.tsv",delimiter = "\t",names = ["phi","Pressure","Temperature","EGR_percentage","CH4_percentage","H2_percentage","Target"])

parameters = df.loc[:,["phi","Pressure","Temperature","EGR_percentage","CH4_percentage","H2_percentage"]]
target = df.loc[:,["Target"]]

x_train,x_test,y_train,y_test = train_test_split(parameters,target,test_size = 0.2,random_state = 10)
y_train = y_train.to_numpy()
y_train = y_train.reshape(628,)

reg = LinearRegression().fit(x_train,y_train)

regr = MLPRegressor(random_state = 1, max_iter = 500).fit(x_train,y_train)

vector = svm.SVR()
vector.fit(x_train,y_train)

dt = DecisionTreeRegressor(random_state = 0)

params={"splitter":["best","random"],
            "max_depth" : [2,3,5,7,10,20],
           "min_samples_leaf":[1,2,3,4,5,6,7,],
           "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30] }

grid_search = GridSearchCV(estimator = dt, param_grid = params, cv = 5, scoring = "neg_mean_squared_error")

grid_search.fit(x_train,y_train)

model = DecisionTreeRegressor(max_depth = 5, max_features = 'auto',max_leaf_nodes = None, min_samples_leaf = 1, min_weight_fraction_leaf = 0.1, splitter = 'best')

model.fit(x_train,y_train)
nn = MLPRegressor()
nn.fit(x_train,y_train)

LR_result = reg.predict(x_test)

plt.scatter(LR_result,np.random.rand(157),color = "orange",label = "Predicted value")
plt.scatter(y_test,np.random.rand(157),label = "Actual Value")
plt.legend()
plt.show()
plt.title("Prediction with LR")
plt.savefig("LR Prediction")

svm_result = regr.predict(x_test)

plt.scatter(svm_result,np.random.rand(157),color = "orange",label = "Predicted Value")
plt.scatter(y_test,np.random.rand(157),label = "Actual Value")
plt.xlabel("Prediction by Support Vector Machine")
plt.legend()
plt.show()
plt.savefig("SVM Prediction")

dt_result = model.predict(x_test)

plt.scatter(dt_result,np.random.rand(157),color = "orange",label = "Predicted Value")
plt.scatter(y_test,np.random.rand(157),label = "Actual Value")
plt.title("Prediction with Regression Tree")
plt.legend()
plt.show()
plt.savefig("Regression Tree Prediction")

nn_result = nn.predict(x_test)

plt.scatter(dt_result,np.random.rand(157),color = "orange",label = "Predicted Value")
plt.scatter(y_test,np.random.rand(157),label = "Actual Value")
plt.title("Prediction with Neural Network")
plt.legend()
plt.show()
plt.savefig("Neural Network Prediction")

df_1 = df[(df['Temperature']==300) & (df['Pressure']==1) & (df["EGR_percentage"]==0) & (df["CH4_percentage"]==1) & (df["H2_percentage"]==0)]

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.scatter(df_1["phi"],df_1["Target"])
ax.set_xlabel('phi')  # Add an x-label to the axes.
ax.set_ylabel('Burning Speed')  # Add a y-label to the axes.
ax.set_title("for Pure methane, 300K, Pressure=0.1MPa,EGR 0%")  # Add a title to the axes.
ax.legend()

df_2 = df[(df['Temperature']==300) & (df['Pressure']==1) & (df["EGR_percentage"]==0.2) & (df["CH4_percentage"]==1) & (df["H2_percentage"]==0)]

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.scatter(df_2["phi"],df_2["Target"])
ax.set_xlabel('phi')  # Add an x-label to the axes.
ax.set_ylabel('Burning Speed')  # Add a y-label to the axes.
ax.set_title("for Pure methane, 300K, Pressure=0.1MPa,EGR 20%")  # Add a title to the axes.
ax.legend()

df_3 = df[(df['Temperature']==800) & (df['H2_percentage']==0.15) & (df["CH4_percentage"]==0.85) & (df["phi"]==1)]

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.plot(df_3["Pressure"].where(df_3["EGR_percentage"]==0),df_3["Target"], label='0% EGR')  # Plot some data on the axes.
ax.plot(df_3["Pressure"].where(df_3["EGR_percentage"]==0.1),df_3["Target"], label='10% EGR')  # Plot more data on the axes...
ax.plot(df_3["Pressure"].where(df_3["EGR_percentage"]==0.2),df_3["Target"], label='20% EGR')  # ... and some more.
ax.set_xlabel('Pressure')  # Add an x-label to the axes.
ax.set_ylabel('Burning Speed')  # Add a y-label to the axes.
ax.set_title("for 85% methane,15% Hydrogen , 800K, phi=1")  # Add a title to the axes.
ax.legend()

df_4 = df[(df['Temperature']==800) & (df['H2_percentage']==0) & (df["CH4_percentage"]==1) & (df["phi"]==1)]

fig, ax = plt.subplots(figsize=(5,2.7))
ax.plot(df_4["Pressure"].where(df_4["EGR_percentage"]==0),df_4["Target"], label='0% EGR')  # Plot some data on the axes.
ax.plot(df_4["Pressure"].where(df_4["EGR_percentage"]==0.1),df_4["Target"], label='10% EGR')  # Plot more data on the axes...
ax.plot(df_4["Pressure"].where(df_4["EGR_percentage"]==0.2),df_4["Target"], label='20% EGR')  # ... and some more.
ax.set_xlabel('Pressure')  # Add an x-label to the axes.
ax.set_ylabel('Burning Speed')  # Add a y-label to the axes.
ax.set_title("for 100% methane,0% Hydrogen, 800K, phi=1")  # Add a title to the axes.
ax.legend()