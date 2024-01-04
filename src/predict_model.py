import click
import torch
from models.model import MyNeuralNet
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.argument("model_checkpoint")
@click.argument("test_data")
def predict(
    model_checkpoint: torch.nn.Module,  # dataloader: torch.utils.data.DataLoader
    test_data: any,
) -> any:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    print("model: " + model_checkpoint)
    print("data: " + test_data)
    # TODO: Implement evaluation logic here

    # model = MyNeuralNet()
    model = torch.load(model_checkpoint)

    test_dataset_path = test_data
    test_set = torch.load(test_dataset_path)

    testloader = torch.utils.data.DataLoader(test_set, batch_size=64)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(images.shape[0], -1)
            images = images.to(torch.float32)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

    return None  # torch.cat([model(batch) for batch in testloader], 0)


cli.add_command(predict)


if __name__ == "__main__":
    cli()
