import torch 
from torch import nn 
import torchvision
from torchvision import datasets 
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader



device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)

# Creating accuracy function 
def accuracy_fn(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    return (correct/len(y_true))*100

# DATA PROCESSING 
# Working with the FashionMNIST dataset 

train_data = datasets.FashionMNIST(root="data", 
                                   train=True, 
                                   download=True, 
                                   transform=ToTensor(),  
                                   target_transform=None)
test_data = datasets.FashionMNIST(root="data",
                                  train=False, 
                                  download=True,
                                  transform=ToTensor())

classes = train_data.classes

"""
Studying the shape of the image tensors : 
[color_channels=1, height=28, width=28]

len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets)
= (60000, 60000, 10000, 10000)
60,000 training samples and 10,000 testing samples

Classes : 10 different kinds of clothes 
['T-shirt/top',
 'Trouser',
 'Pullover',
 'Dress',
 'Coat',
 'Sandal',
 'Shirt',
 'Sneaker',
 'Bag',
 'Ankle boot']
"""

# Visualizing data 
#image, label = train_data[0]
#plt.imshow(image.squeeze(), cmap = "gray")
#plt.title(label)

BATCH_SIZE = 128

# splitting dataset into batches with Dataloader 
# Dataset is too large to forward and backward pass through the whole dataset 
# Batch sizes usually power of 2 : 32, 64, 128, 256, 512...

train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = True)
X_train_batch, y_train_batch = next(iter(train_dataloader))
X_test_batch, y_test_batch = next(iter(test_dataloader))
X_train_batch_flattened = nn.Flatten(X_train_batch) 

# shape after flattening [1, 784] = [colors, width * height]

# --- MODELS --- 

class FashionMNISTModel_V0(nn.Module): # First Linear Model 
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)
    
class FashionMNISTModel_V1(nn.Module): # Second model with non-linearity 
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer_stack(x)

class FashionMNISTModel_V2(nn.Module) : # CNN Model 
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, 
            out_features=output_shape)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x 
  
# HYPERPARAMETERS

INPUT_SHAPE = 784 # 28*28
INPUT_SHAPE_CNN = 1 # 28*28
HIDDEN_UNITS = 64
OUTPUT_SHAPE = len(train_data.classes)
lr = 0.01
EPOCHS = 2

# --- TRAINING FUNCTIONS --- 

def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device): 
    train_loss = 0
    train_acc = 0
    model.to(device)
    for batch, (X,y) in enumerate (data_loader): 
        model.train()
        X.to(device)
        y.to(device)
        y_pred_logits = fashion_model(X)
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()
        y_pred = torch.argmax(y_pred_logits, dim=1)
        train_acc += accuracy_fn(y, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0 : 
            print(f"looked at {batch*len(X)}/{len(data_loader.dataset)} samples")
        
    train_loss /= len(data_loader) # avg loss per batch  
    train_acc /= len(data_loader) # avg acc per batch 
    print (f"train loss : {train_loss}, train accuracy {train_acc}‰")

def test_step(model, data_loader, loss_fn, accuracy_fn, device): 
    test_loss = 0
    test_acc = 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader : 
            test_pred = fashion_model(X)
            test_loss += loss_fn(test_pred, y).item()
            test_pred = torch.argmax(test_pred, dim=1)  
            test_acc += accuracy_fn(y_true=y, y_pred = test_pred)

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print (f"test loss: {test_loss:.5f}, test accuracy: {test_acc:.5f}‰")

fashion_model_V0 = FashionMNISTModel_V0(INPUT_SHAPE,HIDDEN_UNITS,OUTPUT_SHAPE)
fashion_model_V1 = FashionMNISTModel_V1(INPUT_SHAPE,HIDDEN_UNITS,OUTPUT_SHAPE)
fashion_model_V2 = FashionMNISTModel_V2(INPUT_SHAPE_CNN,HIDDEN_UNITS,OUTPUT_SHAPE)
fashion_model = fashion_model_V2 # Choosing model 
# defining loss and optimizer 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fashion_model.parameters(), lr = lr)


if __name__ == "__main__":
    for epoch in range(1, EPOCHS+1): 
        print (f"epoch n° {epoch}")
        train_step(fashion_model,
                   train_dataloader,
                   loss_fn,
                   optimizer,
                   accuracy_fn,
                   device)
        test_step(fashion_model,
                  test_dataloader,
                  loss_fn,
                  accuracy_fn,
                  device)

    # save and load the trained model 
    torch.save(fashion_model.state_dict(), "fashion_cnn_v2.pt")
    fashion_prediction_model = FashionMNISTModel_V2(INPUT_SHAPE_CNN,HIDDEN_UNITS,OUTPUT_SHAPE)
    fashion_prediction_model.load_state_dict(torch.load("fashion_cnn_v2.pt", map_location=device))