import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # Get the data and process it
    # pass

    directory = "data/raw"

    train = []
    train_targets = []  # List to store the targets

    # Load the .pt file from the directory
    for i in range(0, 6):
        i = str(i)
        file_path = directory + "/train_images_" + i + ".pt"
        target_path = directory + "/train_target_" + i + ".pt"  # Path to target file
        loaded_data = torch.load(file_path)
        loaded_targets = torch.load(target_path)  # Load the target file
        train.append(loaded_data)
        train_targets.append(loaded_targets)  # Append the loaded targets
        print("i" + str(i))
        print("i shape" + str(loaded_data.shape))

    train = torch.cat(train)
    train_targets = torch.cat(train_targets)  # Concatenate the targets
    print("Train size:", train.size())
    print("Targets size:", train_targets.size())

    test = torch.load(directory + "/test_images.pt")
    print("Test size:", test.size())
    test_targets = torch.load(directory + "/test_target.pt")
    print("Test targets size:", test_targets.size())

    train_set = TensorDataset(train, train_targets)
    test_set = TensorDataset(test, test_targets)

    destination_dir = "data/processed"

    scaler = StandardScaler()

    # Reshape train and test data to 2-dimensional arrays
    train_2d = train.view(train.size(0), -1)
    test_2d = test.view(test.size(0), -1)

    normalized_train = scaler.fit_transform(train_2d)
    normalized_test = scaler.transform(test_2d)

    normalized_train_set = TensorDataset(torch.tensor(normalized_train), train_targets)
    normalized_test_set = TensorDataset(torch.tensor(normalized_test), test_targets)

    torch.save(normalized_train_set, destination_dir + "/normalized_train_set.pt")
    torch.save(normalized_test_set, destination_dir + "/normalized_test_set.pt")
