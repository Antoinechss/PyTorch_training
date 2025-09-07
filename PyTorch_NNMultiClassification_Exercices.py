import torch
from torch import nn 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

import pandas as pd 


"""
1/ Make a binary classification dataset with Scikit-Learn's make_moons() function
"""

# Create the dataset 

X,y = make_moons(n_samples = 1000,
                 shuffle = True,
                 noise = 0.2,
                 random_state = 42)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Turn data into a DataFrame

x1 = X[:,0].numpy()
x2 = X[:,1].numpy()
y = y.numpy()

df = pd.DataFrame({"x1" : x1, "x2" : x2, "label" : y})
print(df.head())

# Visualize data on scatter plot 

plt.scatter(x1, x2, c = y, cmap=plt.cm.RdYlBu)
plt.show()

# split data into train and test set & send to device

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

"""
2/ Build a model by subclassing nn.Module that incorporates non-linear 
activation functions and is capable of fitting the data created 
"""

class MoonModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(), 
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(), 
            nn.Linear(in_features=hidden_units, out_features=output_features),)
    
    def forward(self,x):
        return self.linear_layer_stack(x)
    
moon_model = MoonModel(input_features = 2,output_features = 1,hidden_units = 10)
print (moon_model)