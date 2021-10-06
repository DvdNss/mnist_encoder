import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
from tqdm import tqdm

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 1

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Define model
class MultiChannelEncoder(nn.Module):
    def __init__(self):
        super(MultiChannelEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        self.img_encoder, self.label_encoder = nn.Linear(784, 195), nn.Linear(10, 5)
        self.total_encoder = nn.Linear(200, 50)
        self.total_decoder = nn.Linear(50, 200)
        self.img_decoder, self.label_decoder = nn.Linear(195, 784), nn.Linear(5, 10)

    def forward(self, img, label):
        img, label = self.img_encoder(img), self.label_encoder(label)
        img, label = self.sigmoid(img), self.sigmoid(label)

        encoding = torch.cat((img[0], label[0]), dim=0)
        encoding = self.total_encoder(encoding)
        encoding = self.sigmoid(encoding)

        decoding = self.total_decoder(encoding)
        decoding = self.sigmoid(decoding)

        split = torch.split(decoding, 195)
        img, label = self.img_decoder(split[0]), self.label_decoder(split[1])
        img_logits, label_logits = img, label

        return img_logits, label_logits


model = MultiChannelEncoder().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = model.flatten(X), F.one_hot(torch.tensor(y), num_classes=10).type(torch.FloatTensor)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        img_pred, label_pred = model(X, y)
        img_loss = loss_fn(img_pred, X)
        label_loss = loss_fn(label_pred, y)

        # Backpropagation
        optimizer.zero_grad()
        img_loss.backward(retain_graph=True)
        label_loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            img_loss, label_loss, current = img_loss.item(), label_loss.item(), batch * len(X)
            print(f"img_loss: {img_loss:>7f} || label_loss: {label_loss:>7f} || [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = model.flatten(X), F.one_hot(torch.tensor(y), num_classes=10).type(torch.FloatTensor)
            X, y = X.to(device), y.to(device)

            img_pred, label_pred = model(X, y)
            img_loss = loss_fn(img_pred, X).item()
            label_loss = loss_fn(label_pred, y).item()
            correct += (torch.argmax(label_pred) == torch.argmax(y)).type(torch.float).sum().item()
    label_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {label_loss:>8f} \n")


epochs = 3
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
