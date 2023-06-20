import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """A dataset for time series data."""

    def __init__(self, dataframe, target, features, sequence_length=5):
        """Initializes the dataset.

        Args:
            dataframe (pandas.DataFrame): A dataframe containing both the features and target.
            target (str): The name of the target column.
            features (list[str]): A list of feature column names.
            sequence_length (int): The length of the sequence to use for training.
        """
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        """Returns the length of the dataset."""
        return self.X.shape[0]

    def __getitem__(self, i):
        """Returns a tuple of (X, y) where X is a 2D tensor of shape (seq_len, num_features),
        and y is a 1D tensor of shape (1,).
        """
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


def train_lstm_model(data_loader, model, loss_function, optimizer):
    """Trains the LSTM model (model.ShallowRegressionLSTM).

    Args:
        data_loader (torch.utils.data.DataLoader): A data loader for the training data.
        model (Union[model.ShallowRegressionLSTM, torch.nn.Module]): The model to train.
        loss_function (torch.nn.modules.loss._Loss): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use.
    """
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print("Train loss: {}".format(avg_loss))


def test_lstm_model(data_loader, model, loss_function):
    """Tests the LSTM model (model.ShallowRegressionLSTM).

    Args:
        data_loader (torch.utils.data.DataLoader): A data loader for the test data.
        model (Union[model.ShallowRegressionLSTM, torch.nn.Module]): The model to test.
        loss_function (torch.nn.modules.loss._Loss): The loss function to use.
    """
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print("Test loss: {}".format(avg_loss))


def lstm_predict(data_loader, model):
    """Makes predictions using the LSTM model (model.ShallowRegressionLSTM).

    Args:
        data_loader (torch.utils.data.DataLoader): A data loader for the test data.
        model (Union[model.ShallowRegressionLSTM, torch.nn.Module]): The model to use for prediction.
    """
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output
