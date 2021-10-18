from torchvision.transforms import ToTensor

from model import Model

# Load data & dataloader
train_data, test_data, train_dataloader, test_dataloader = Model.load_mnist(transform=ToTensor(), batch_size=1)

# Load model
model = Model(load_model='model/model.pt', img_chan_size=100, global_chan_size=50)
print(f'Trainable parameters: {sum(p.numel() for p in model.model.parameters())}. ')

# Quick inference
model.infer(eval_data=test_data, random=True)
