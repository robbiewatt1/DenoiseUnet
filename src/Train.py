import torch
from Network import Unet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Dataset import DenoiseDataset, DatasetTest
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_epoch(model, train_loader, loss_fn, optimizer, writer, epoch, device):
    """
    Train the model for one epoch
    :param model: Model to be trained
    :param train_loader: Training data loader
    :param loss_fn: Loss function
    :param optimizer: Optimiser
    :param writer: Tensorboard writer
    :param epoch: Epoch number
    :param device: Device to train on
    """

    model.train()
    losses = []
    with tqdm(total=len(train_loader), dynamic_ncols=True) as tq:
        tq.set_description(
            f"Train :: Epoch: {epoch} ")

        for i, (image_in, image_out) in enumerate(train_loader):

            # Move to device
            image_in = image_in.to(device)
            image_out = image_out.to(device)

            # Forward pass
            image_pred = model(image_in)
            loss = loss_fn(image_pred, image_out)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            tq.set_postfix(loss=loss.item())
            tq.update(1)

            # Write to tensorboard
            writer.add_scalar("Loss/train", loss.item(),
                              len(train_loader) * epoch + i)

        mean_loss = sum(losses) / len(losses)
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss

def test_epoch(model, train_loader, test_dataset, writer, epoch,
               device, n_images=4):
    """
    Test the model for one epoch
    :param model: Model to be tested
    :param train_loader: Training data loader
    :param data_set: Testing dataset
    :param writer: Tensorboard writer
    :param epoch: Epoch number
    :param device: Device to test on
    """

    # Test the model on simulations
    model.eval()
    with torch.no_grad():
        for i, (image_in, image_out) in enumerate(train_loader):
            # Move to device
            image_in = image_in.to(device)
            image_out = image_out.to(device)

            # Forward pass
            image_pred = model(image_in)
            image_in = train_loader.dataset.reverse_transform(image_in)

            fig, ax = plt.subplots(nrows=3, ncols=n_images, figsize=(20, 15))

            for i in range(n_images):
                ax[0, i].pcolormesh(image_in[i, 0].cpu().numpy().T)
                ax[0, i].axis('off')
                ax[1, i].pcolormesh(image_pred[i, 0].cpu().numpy().T)
                ax[1, i].axis('off')
                ax[2, i].pcolormesh(image_out[i, 0].cpu().numpy().T)
                ax[2, i].axis('off')

            writer.add_figure('example_figures', fig, epoch)
            break

    # Test the model on real data
    with torch.no_grad():
        plot_index = [6, 7, 8, 9]
        batch_index = np.random.randint(0, 5)
        images, images_mean = test_dataset[batch_index]

        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        for i, index in enumerate(plot_index):
            sample = images[2].flatten(0, 1).to(device)[:, None]

            sample = model(sample)
            sample = sample.reshape((10, 40, 256, 256))

            ax[0, i].pcolormesh(sample[index, 0].cpu().numpy().T)
            ax[0, i].axis('off')
            ax[1, i].pcolormesh(images_mean[2, index].cpu().numpy().T)
            ax[1, i].axis('off')
        writer.add_figure('real_data', fig, epoch)


def main():

    # Set the dvice and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet()
    model.to(device)

    # Set the dataset
    dataset = DenoiseDataset("./Data/Train_samples.h5", device=device)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True)

    # Set the test dataset
    dataset_test = DatasetTest("./Data/Test_290124.h5")

    # Optimiser and loss
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Define tensorboard writer
    writer = SummaryWriter()

    epochs = 1000
    epoch_loss = []
    for epoch in range(epochs):
        # Train the model
        loss = train_epoch(model, train_loader, loss_fn, optim, writer, epoch,
                           device)
        epoch_loss.append(loss)

        # Test the model
        if (epoch + 1) % 5 == 0:
            test_epoch(model, train_loader, dataset_test, writer, epoch, device)

        # Save the model
        if epoch_loss[-1] == min(epoch_loss) or epoch % 10 == 0:
            torch.save(model.state_dict(), f"./ModelsB1B2/{epoch}.pt")

if __name__ == '__main__':
    main()