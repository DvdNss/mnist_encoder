import numpy as np
import torch.nn
from torchvision.transforms import ToTensor

from model import Model

# Load data & dataloader
train_data, test_data, train_dataloader, test_dataloader = Model.load_mnist(transform=ToTensor(), batch_size=1)

# Load model
model = Model(device='cuda', img_chan_size=100, global_chan_size=50)
print(model.model)

# Loss & Optim
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)

epoch = 10
mask_prob = 30

# Training model
for _ in range(0, epoch):
    model.train(dataloader=train_dataloader, loss=loss, optimizer=optimizer, mask_prob=mask_prob, log_iter=60000)
    model.eval(dataloader=test_dataloader, loss=loss)
    model.eval(dataloader=test_dataloader, loss=loss, mask=True)
    mask_prob += 30
    print(f'Mask probability is now {mask_prob}%. ')

# Save model
model.save('model/model.pt')
