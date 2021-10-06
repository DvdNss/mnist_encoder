import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from network import MultiChannelEncoder

training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST( root="data", train=False, download=True, transform=ToTensor())

model = MultiChannelEncoder()
model.load_state_dict(torch.load('model.pth'))

model.eval()
x, y = training_data[0][0], training_data[0][1]
x, y = model.flatten(x), F.one_hot(torch.tensor(y), num_classes=10).type(torch.FloatTensor)
with torch.no_grad():
    img_pred, label_pred = model(x, y)
    print(label_pred, y)
