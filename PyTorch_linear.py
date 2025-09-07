import torch 
from torch import nn 
import matplotlib.pyplot as plt 

from pathlib import Path

# creating simple dataset by hand 

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02

X = torch.arange(start,end,step).unsqueeze(dim = 1) # Features 
y = weight * X + bias # Labels 

# Split data into training (80%) and test (20%)

split = int(len(X)*0.8)

X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Create a linear regression model with PyTorch 

class LinearRegressionModel(nn.Module): 
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)
    
    def forward(self, x):
        return self.weights * x + self.bias 

torch.manual_seed(42)
model_0 = LinearRegressionModel()

# ----- Display and visualize data --------

def display_prediction(train_data, train_labels, test_data, test_labels, predictions):
    plt.scatter(train_data, train_labels)
    plt.scatter(test_data, test_labels)
    if predictions is not None : 
        plt.scatter(test_data, predictions)
    plt.show()

# -----------------------------------------

# Loss function : Mean absolute error (MAE) for regression problems : builtin torch.nn.L1Loss
# Optimizer : Stochastic gradient descent (SGD) : builtin torch.optim.SGD

loss_function = nn.L1Loss()
optimizer = torch.optim.SGD(model_0.parameters(), lr = 0.01)

epochs = 100 # how many times model will pass through training set 

train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range (epochs):

    # train 
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_function(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # test 
    model_0.eval()
    with torch.inference_mode(): 
        test_pred = model_0(X_test)
        test_loss = loss_function(test_pred,y_test.type(torch.float))

        if epoch % 10 == 0 : 
            epoch_count.append(epoch)
            test_loss_values.append(test_loss.detach().cpu().item())
            train_loss_values.append(loss.detach().cpu().item())
            print(f"Epoch: {epoch}, MAE Train Loss: {loss}, MAE Test Loss: {test_loss} ")


# Displaying progress 

plt.plot(epoch_count,test_loss_values, label = "test loss values" )
plt.plot(epoch_count,train_loss_values, label = "train loss values" )
plt.show()

# Displaying data estimations 

with torch.inference_mode(): 
    y_pred = model_0(X_test)
display_prediction(X_train, y_train, X_test, y_test, y_pred)

# --- Saving the trained model ---

# create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# 2. Create model save path 
MODEL_NAME = "pytorch_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model_0.state_dict(), MODEL_SAVE_PATH)

# --- Loading the saved model ---

loaded_model_0 = LinearRegressionModel() # instantiate a new model 
loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Evaluate loaded model 
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_0_preds = loaded_model_0(X_test)
print (y_pred == loaded_model_0_preds)