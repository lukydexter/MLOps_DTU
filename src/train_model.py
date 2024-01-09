import click
import torch
from models.model import MyNeuralNet
import matplotlib.pyplot as plt

import numpy
from torch import nn, optim
import torch.nn.functional as F


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyNeuralNet()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    # train_set, _ = mnist()
    directory = "data/processed"
    train_set = torch.load(directory + "/normalized_train_set.pt")

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    epochs = 5

    epoch_losses = []  # List to store average loss per epoch
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            images = images.to(torch.float32)
            # TODO: Training pass
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(trainloader)
        epoch_losses.append(average_loss)  # Store average
        print(f"Training loss: {average_loss}")

    torch.save(model, "models/checkpoint.pth")

    # Save the training curve figure to a specific location
    plt.plot(range(epochs), epoch_losses)
    plt.xlabel("Training Step")
    plt.ylabel("Training Loss")
    plt.title("Training Curve")
    plt.savefig("reports/figures/training_curve.png")
# lr= 1e-3
# train(lr)
cli.add_command(train)


if __name__ == "__main__":
   cli()
