import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()

        states = data["hidden_states"]
        y = data["decision"]
        assert states.shape[0] == y.shape[0]  # as y.shape[0] = dataset size
        self.states = states
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.states[index], self.y[index]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, tobert_model_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, tobert_model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, tobert_model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, tobert_model_path):
        """
        Saves model when validation loss decreases and update the minimum validation loss value.
        :param val_loss: validation loss
        :param model: model to be saved
        :param tobert_model_path: path where the model will be saved.
        """
        # if self.verbose:
        # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), tobert_model_path)
        self.val_loss_min = val_loss


def load_states_data(filepath):

    papers = np.load(filepath, allow_pickle=True)
    states_df = pd.DataFrame({'paper_id': papers[:, 0], 'decision': papers[:, 1], 'hidden_states': papers[:, 2]})

    return states_df


def prepare_data_for_tobert(data, max_chunk_length, seed):
    """
    Filter and split the data for training, validation and testing.
    :param data: data frame containing the hidden states.
    :param max_chunk_length: maximum number of chunks possible in a paper
    :param seed: for reproducibility
    :return: myDataset objects of train, dev and test sets
    """

    # remove papers having more than the maximum chunk length allowed.
    data["no_of_chunks"] = [len(data["hidden_states"][i]) for i in range(len(data))]
    filtered_papers = data["no_of_chunks"] <= max_chunk_length
    data = data[filtered_papers]

    # remove papers having 0 chunks
    non_zero_papers = data["no_of_chunks"] > 0
    data = data[non_zero_papers]

    data = data.drop(["no_of_chunks"], axis=1)
    data = data.drop(["paper_id"], axis=1)

    dev_percent = 0.1
    test_percent = 0.2

    train_data, dev_test = train_test_split(data,
                                            test_size=test_percent + dev_percent,
                                            stratify=data['decision'], random_state=seed)
    dev, test = train_test_split(dev_test,
                                 test_size=test_percent / (test_percent + dev_percent),
                                 stratify=dev_test['decision'], random_state=seed)

    train_data.reset_index(drop=True, inplace=True)
    dev.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return MyDataset(train_data), MyDataset(dev), MyDataset(test)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
