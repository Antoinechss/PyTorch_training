import torch 
from torch import nn 
import torchvision
from torchvision import datasets 
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from pathlib import Path 
from PIL import Image
from PyTorch_ComputerVision_Model import FashionMNISTModel_V0, FashionMNISTModel_V1, FashionMNISTModel_V2


device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)

train_data = datasets.FashionMNIST(root="data", 
                                   train=True, 
                                   download=True, 
                                   transform=ToTensor(),  
                                   target_transform=None)

classes = train_data.classes

INPUT_SHAPE = 784 # 28*28
INPUT_SHAPE_CNN = 1 # 28*28
HIDDEN_UNITS = 64
OUTPUT_SHAPE = len(train_data.classes)


 #Import model 

fashion_prediction_model = FashionMNISTModel_V2(INPUT_SHAPE_CNN,HIDDEN_UNITS,OUTPUT_SHAPE)
fashion_prediction_model.load_state_dict(torch.load("fashion_cnn_v2.pt", map_location=device))

# Import and pre-processing image 

img_path = Path("/Users/antoinechosson/Desktop/PyTorch_training/computer vision/overalls.jpeg")

preprocess_fn = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize(28),
    torchvision.transforms.CenterCrop(28),
    torchvision.transforms.ToTensor()
])



def predict_cloth_type(img_path, model, device):
    model.eval()
    model.to(device)

    img = Image.open(img_path).convert("RGB")
    X = preprocess_fn(img).unsqueeze(0)
    with torch.inference_mode():
        pred_logits = model(X)
        pred_probs = torch.argmax(pred_logits, dim=1)
    idx = int(pred_probs.argmax())
    return f"this cloth is a : {classes[idx]}"

print (predict_cloth_type(img_path, fashion_prediction_model, device))