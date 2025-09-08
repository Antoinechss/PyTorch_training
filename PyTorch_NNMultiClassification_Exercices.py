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

# split data into train and test set & send to device

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

"""
2/ Build a model by subclassing nn.Module that incorporates non-linear 
activation functions and is capable of fitting the data created 
"""

# --- Accuracy model evaluation function --- 

def accuracy(y_pred,y_test):
    n_correct = torch.eq(y_test, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    return (n_correct/len(y_pred))*100

# --------------------------------------

def prob_to_class(y_pred_probs):
    return (y_pred_probs >= 0.5).float()

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

loss_function = nn.BCEWithLogitsLoss()
optimizer =  torch.optim.SGD(moon_model.parameters(), lr = 0.1)

epochs = 1000

for epoch in range (epochs):

    # train 
    moon_model.train()
    y_logits = moon_model(X_train)
    y_pred = torch.softmax(y_logits, dim = 1).argmax(dim=1)

    loss = loss_function(y_logits, y_train.squeeze())
    acc = accuracy(y_pred,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # test 
    moon_model.eval()
    with torch.inference_mode():
        test_logits = moon_model(X_test)
        test_pred = torch.softmax(test_logits, dim = 1).argmax(dim=1)

        test_loss = loss_function(test_logits, y_test.squeeze())
        test_acc = accuracy(test_pred,y_test)

    if epoch % 10 == 0 : 
        print(epoch, test_loss, test_acc)
