from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import torch 
from torch import nn 

# Visualize on : https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=5,4&seed=0.47617&showTestData=false&discretize=true&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
# --- Creating dataset ---

n_samples = 1000
X,y = make_circles(n_samples, noise = 0.03, random_state = 42)

# Each label has two features : 
print(X[:5], y[:5]) # Each label (1 or 0) has a pair of features -> Binary classification 

# Visualizing data in plane  
plt.scatter(x=X[:, 0],y=X[:, 1],c=y,cmap=plt.cm.RdYlBu) 
plt.show()
# goal = classify wether dot is Red or Blue 

# turning data into tensors 
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# split into training and test sets with builting function in sklearn 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

# --- Building model 3 layers, 10 hidden units/neurons ---

# privilege use of GPU for parrallel calculations + send data to device 
device = "cuda" if torch.cuda.is_available() else "cpu"
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

class CircleModel_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 10) 
        self.layer_2 = nn.Linear(in_features = 10, out_features = 10) 
        self.layer_3 = nn.Linear(in_features = 10, out_features = 1)
        self.relu = nn.ReLU() # Add in non linear activation function ReLU (Rectified Linear Unit)

    def forward(self,x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))) # Adding non linear activation function in between hidden layers 

# create instance of model and send it to device
circle_model_0 = CircleModel_0().to(device)

# Loss function : binary cross entropy with logit function activation (more stable): builtin torch.nn.BCEWithLogitsLoss()
loss_function = nn.BCEWithLogitsLoss()
# Optimizer : Stochastic Gradient Descent
optimizer =  torch.optim.SGD(circle_model_0.parameters(), lr = 0.1)

# --- Evaluation metrics ----------

def accuracy(y_pred,y_test):
    n_correct = torch.eq(y_test, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    return (n_correct/len(y_pred))*100

# ---------------------------------

# turning probabilities into classification 

def prob_to_class(y_pred_probs):
    y_pred = []
    for pred_prob in y_pred_probs : 
        if pred_prob >= 0.5 : 
            y_pred.append(1)
        else : 
            y_pred.append(0)
    y_pred = torch.tensor(y_pred)
    return y_pred



torch.manual_seed(42)

epochs = 1000

for epoch in range(epochs): 

    # training 
    circle_model_0.train()
    y_logits = circle_model_0(X_train)
    y_pred_probs = torch.sigmoid(y_logits) # applying sigmoid activation 
    y_pred = prob_to_class(y_pred_probs)

    loss = loss_function(y_logits, y_train.unsqueeze(1))    
    acc = accuracy(y_pred,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # testing 
    circle_model_0.eval()
    with torch.inference_mode():
        test_logits = circle_model_0(X_test)
        test_pred_probs = torch.sigmoid(test_logits)
        test_pred = prob_to_class(test_pred_probs)

        test_loss = loss_function(test_logits, y_test.unsqueeze(1))
        test_acc = accuracy(test_pred,y_test)

    if epoch % 10 == 0 : 
        print (epoch, test_loss, test_acc)


# --- Visualizing performance --- 

from helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(circle_model_0, X_train, y_train)
plt.show()
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(circle_model_0, X_test, y_test)
plt.show()