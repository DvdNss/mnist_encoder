from random import randint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

from mc_autoencoder import MultiChannelAutoEncoder


class Model:
    """Model training and testing. """

    def __init__(self, load_model: str = "", device: str = "", **kwargs):
        """
        Init model.

        :param load_model: whether to load model from path or create a new one
        :param kwargs: other arguments for MultiChannelAutoencoder
        """

        # Get cpu or gpu device for training.
        if device == "":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model with kwargs
        self.model = MultiChannelAutoEncoder(**kwargs).to(self.device)
        print(f"--> Model loaded on {self.device} device. ")

        # Load model with path if needed
        if load_model != "":
            self.model.load_state_dict(torch.load(load_model))

        self.args = kwargs

    @staticmethod
    def load_mnist(transform=ToTensor(), batch_size: int = 1):
        """
        Load MNIST dataset.

        :return:
        """

        # Load datasets
        train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

        # Create dataloaders
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        return train_data, test_data, train_dataloader, test_dataloader

    def train(self, dataloader, loss, optimizer, log_iter: int = 10000, mask_prob=0):
        """
        Train model.

        :param mask_prob: probability to mask output
        :param dataloader: dataloader
        :param loss: loss
        :param optimizer: optimizer
        :param log_iter: logs iretations
        :return:
        """

        size = len(dataloader.dataset)
        self.model.train()

        for batch, (x, y) in enumerate(dataloader):
            # Preprocessing
            x, y = self.model.flatten(x), F.one_hot(y.clone().detach(), num_classes=10).type(torch.FloatTensor)
            y_buff = y.to(self.device)

            # Forward pass
            mask = True if randint(0, 100) <= mask_prob else False
            if not mask:
                x, y = x.to(self.device), y.to(self.device)
            else:
                x, y = x.to(self.device), torch.zeros(1, 10).to(self.device)

            # Forward pass
            img_pred, label_pred = self.model(x, y)

            # Compute loss
            img_loss = loss(img_pred, x)
            label_loss = loss(label_pred, y_buff)

            # Backpropagation
            optimizer.zero_grad()
            label_loss.backward(retain_graph=True)
            img_loss.backward()
            optimizer.step()

            # Display
            if batch % log_iter == 0:
                img_loss, label_loss, current = img_loss.item(), label_loss.item(), batch * len(x)
                print(f"img_loss: {img_loss:>7f} || label_loss: {label_loss:>7f} || [{current:>5d}/{size:>5d}]")

    def eval(self, dataloader, loss, mask: bool = False):
        """
        Test model.

        :param mask: whether to mask input labels or not
        :param dataloader: dataloader
        :param loss: loss
        :return:
        """

        # Dataloader parameters
        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        # Trigger evaluation mode
        self.model.eval()

        # Init model props
        test_loss, correct = 0, 0

        with torch.no_grad():
            for x, y in tqdm(dataloader):
                # Preprocessing
                x, y = self.model.flatten(x), F.one_hot(y.clone().detach(), num_classes=10).type(torch.FloatTensor)
                y_buff = y.to(self.device)

                if not mask:
                    x, y = x.to(self.device), y_buff
                else:
                    x, y = x.to(self.device), torch.zeros(1, 10).to(self.device)

                # Forward pass
                img_pred, label_pred = self.model(x, y)

                # Compute loss
                img_loss = loss(img_pred, x).item()
                label_loss = loss(label_pred, y_buff).item()

                # Compute logs
                correct += (torch.argmax(label_pred) == torch.argmax(y_buff)).type(torch.float).sum().item()

        label_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {label_loss:>8f} \n")

    def save(self, path: str = 'model.pth'):
        """
        Save model to given path.

        :param path: path to save
        :return:
        """

        # Save model
        torch.save(self.model.state_dict(), f=path)
        print(f'Saved model to {path}. ')

    def infer(self, eval_data, random: bool = True, n_example: int = 0):
        """
        Quick inferance example with img save.

        :param eval_data: data from which choose example
        :param n_example: example number
        :param random: whether to infer on a random example or not
        :return:
        """

        # Eval mode
        self.model.eval()

        # Choose random example if needed
        if random:
            rand = randint(0, len(eval_data)-1)
        else:
            rand = n_example
        print(f"Using example {rand}. ")

        # Pull example and save it
        x, y = eval_data[rand][0], eval_data[rand][1]
        save_image(x, "example/target.png")

        # Preprocess input
        x, y = self.model.flatten(x), F.one_hot(torch.tensor([y]), num_classes=10).type(torch.FloatTensor)

        with torch.no_grad():
            img_pred, label_pred = self.model(x.to(self.device),
                                              torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).type(torch.FloatTensor).to(
                                                  self.device))
            print(f'Classic --> Target: {torch.argmax(y)} || Output: {torch.argmax(label_pred)}')
            img_pred = self.model.unflatten(img_pred)
            save_image(img_pred, "example/output.png")

        target = torch.argmax(y)
        prediction = torch.argmax(label_pred)

        # Preprocess prediction
        x = self.model.flatten(img_pred)

        with torch.no_grad():
            img_pred, label_pred = self.model(x.to(self.device),
                                              torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).type(torch.FloatTensor).to(
                                                  self.device))
            print(f'Recursive --> Target: {torch.argmax(y)} || Output: {torch.argmax(label_pred)}')
            img_pred = self.model.unflatten(img_pred)
            save_image(img_pred, "example/output_rec.png")
            print(f'--> Saved images in example/ . ')

        return target, prediction
