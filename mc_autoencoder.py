import torch
from torch import nn


class MultiChannelAutoEncoder(nn.Module):
    """Multi channel autoencoder for MNIST. """

    def __init__(self, img_size: int = 784, label_size: int = 10, img_chan_size: int = 512,
                 label_chan_size: int = 10, global_chan_size: int = 300, num_classes: int = 10):
        """
        Init model and layers.

        :param img_size: size of input image
        :param label_size: size of label one-hot vector
        :param img_chan_size: img encoding size
        :param label_chan_size: label encoding size
        :param global_chan_size: global encoding size
        """

        super(MultiChannelAutoEncoder, self).__init__()

        # Network properties
        self.num_classes = num_classes
        self.img_size = img_size
        self.label_size = label_size
        self.img_chan_size = img_chan_size
        self.label_chan_size = label_chan_size
        self.global_chan_size = global_chan_size

        # Flatten, activation and unflatten layers
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.unflatten = nn.Unflatten(0, torch.Size([1, 28, 28]))

        # Channel encoding layers
        self.img_encoder, self.label_encoder = nn.Linear(img_size, img_chan_size), nn.Linear(label_size,
                                                                                             label_chan_size)

        # Global encoding layer
        self.total_encoder = nn.Linear(img_chan_size + label_chan_size, global_chan_size)
        self.total_decoder = nn.Linear(global_chan_size, img_chan_size + label_chan_size)

        # Channel decoding layers
        self.img_decoder, self.label_decoder = nn.Linear(img_chan_size, img_size), nn.Linear(label_chan_size,
                                                                                             label_size)

    def forward(self, img, label):
        """
        Forward pass through network.

        :param img: input image
        :param label: input label
        :return: img and label logits
        """

        # Pass and activate image and label through respective encoding layers
        img, label = self.img_encoder(img), self.label_encoder(label)
        img, label = self.sigmoid(img), self.sigmoid(label)

        # Merge image and label encoding
        encoding = torch.cat((img[0], label[0]), dim=0)

        # Pass and activate channel encodings through global encoding layer
        encoding = self.total_encoder(encoding)
        encoding = self.sigmoid(encoding)

        # Pass and activate global encoding layer through respective decoding layers
        decoding = self.total_decoder(encoding)
        decoding = self.sigmoid(decoding)

        # Split the global encoding into image and label encodings
        split = torch.split(decoding, self.img_chan_size)

        # Pass and activate final output
        img, label = self.img_decoder(split[0]), self.label_decoder(split[1])
        # img, label = self.sigmoid(img), self.sigmoid(label)

        img_logits, label_logits = img, label

        return img_logits, label_logits
