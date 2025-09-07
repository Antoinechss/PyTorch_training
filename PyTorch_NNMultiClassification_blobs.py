import matplotlib.pyplot as plt 

import torch 
from torch import nn 

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

# --- Creating a multi-class dataset --- 

N_CLASSES = 4
N_FEATURES = 2 

X,y = make_blobs(n_samples = 1000, 
                 n_features = N_FEATURES, 
                 centers = N_CLASSES,
                 cluster_std = 1.5, 
                 random_state = 42)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)

print (X[:5], y[:5])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

plt.scatter(X[:,0], X[:,1], c = y, cmap=plt.cm.RdYlBu)
plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# --- Accuracy model evaluation function --- 

def accuracy(y_pred,y_test):
    n_correct = torch.eq(y_test, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    return (n_correct/len(y_pred))*100

# --------------------------------------

class MultiClassModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(), 
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(), 
            nn.Linear(in_features=hidden_units, out_features=output_features),)


    def forward(self, x):
        return self.linear_layer_stack(x)
    

multi_model = MultiClassModel(input_features = N_FEATURES,
                              output_features = N_CLASSES,
                              hidden_units = 8).to(device)

# loss : Cross Entropy Loss function : builtin nn.CrossEntropyLoss()
loss_function = nn.CrossEntropyLoss()
# Optimizer : Stochastic Gradient Descent 
optimizer =  torch.optim.SGD(multi_model.parameters(), lr = 0.1)

epochs = 1000

for epoch in range (epochs):

    # train 
    multi_model.train()
    y_logits = multi_model(X_train)
    # Logits -> Prediction probabilities -> Prediction labels
    y_pred = torch.softmax(y_logits, dim = 1).argmax(dim=1)

    loss = loss_function(y_logits, y_train)
    acc = accuracy(y_pred,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # test 
    multi_model.eval()
    with torch.inference_mode():
        test_logits = multi_model(X_test)
        test_pred = torch.softmax(test_logits, dim = 1).argmax(dim=1)

        test_loss = loss_function(test_logits, y_test)
        test_acc = accuracy(test_pred,y_test)

    if epoch % 10 == 0 : 
        print (epoch, test_loss, test_acc)

# --- Visualize predictions --- 

from helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(multi_model, X_train, y_train)
plt.show()
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(multi_model, X_test, y_test)
plt.show()

