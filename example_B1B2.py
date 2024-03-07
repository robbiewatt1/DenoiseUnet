import torch
from src.Network import Unet
import numpy as np
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model
model = Unet()
model.load_state_dict(torch.load('./Models/B1B2Net.pt'))
model.to(device)
model.eval()

# Load the noisy data (256 x 256 slice of image with counts clipped to 0-100)
images = np.load('example_images.npy')

# Normalise the data
images = (images - 16.6915) / 14.0096

# Preprocess the data
images = torch.tensor(images[:, None]).float().to(device)
with torch.no_grad():
    clean_images = model(images)

# plot the clean and noisy images
sample_index = np.random.randint(0, 20, 3)
fig, ax = plt.subplots(2, 3)
for i, index in enumerate(sample_index):
    ax[0, i].pcolormesh(images[index, 0].cpu().numpy().T)
    ax[0, i].axis('off')
    ax[1, i].pcolormesh(clean_images[index, 0].cpu().numpy().T)
    ax[1, i].axis('off')
plt.show()
