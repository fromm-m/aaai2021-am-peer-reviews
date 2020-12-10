import numpy as np
import pandas as pd
import random
import time
import os

from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import trange, tqdm

from BertClassification_for_tobert import predict_papers_using_bert
from tobert_utils import load_states_data, EarlyStopping, prepare_data_for_tobert,\
    epoch_time


class ToBERT_Encoder(nn.Module):

    def __init__(self,
                 hid_dim,
                 n_layers,
                 n_heads,
                 max_chunk_length,
                 dim_feedforward,
                 dropout,
                 approach):

        super(ToBERT_Encoder, self).__init__()

        self.approach = approach
        self.hid_dim = hid_dim
        self.max_chunk_length = max_chunk_length
        self.weights = None

        self.encoder_layer = nn.TransformerEncoderLayer(
            hid_dim, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.linear = nn.Linear(in_features=hid_dim, out_features=1, bias=True)

    def forward(self, seq):

        out = self.transformer_encoder(seq)
        # out = (max_chunk_length, batch_len, hid_dim)
        if self.approach == "mean":

            out_mean = torch.stack([torch.mean(out, 0)])
            # out_mean = (1, batch_len, hid_dim)
            out_mean = out_mean.reshape(out_mean.shape[1], out_mean.shape[2])
            # out_mean = (batch_len, hid_dim)
            logit_mean = self.linear(out_mean)
            # logit_mean = (batch_len, 1)
            logit_mean = logit_mean.double()

            return logit_mean

        elif self.approach == "max":

            out_max = torch.stack([torch.max(out, 0)[0]])
            # out_max = (1, batch_len, hid_dim)
            out_max = out_max.reshape(out_max.shape[1], out_max.shape[2])
            # out_max = (batch_len, hid_dim)
            logit_max = self.linear(out_max)
            # logit_max = (batch_len, 1)
            logit_max = logit_max.double()

            return logit_max

        else:
            cls_concat = nn.Linear(in_features=self.hid_dim * self.max_chunk_length,
                                   out_features=1, bias=True)
            out_concat = out.reshape(out.shape[1], out.shape[0] * out.shape[2])
            # out_concat = (batch_len, max_chunk_len * hid_dim)
            logit_concat = cls_concat(out_concat)
            # logit_concat = (batch_len, 1)
            logit_concat = logit_concat.double()

            return logit_concat

    def loss_function(self, logits, gt):

        gt = gt.double()
        # gt = (batch_length, 1)
        loss = nn.functional.binary_cross_entropy_with_logits(
            input=logits, target=gt, pos_weight=self.weights)

        return loss

    def set_class_weights(self, traindata):

        decisions = traindata[:][1]
        class_distrib = decisions.value_counts()
        self.weights = torch.tensor(round(max(class_distrib) / class_distrib[1], 2), dtype=torch.double)


def train(model, trainloader, max_chunk_length, optimizer, clip):

    model.train()
    epoch_loss = 0

    for step, batch in enumerate(
                    tqdm(trainloader, desc="Iteration")):

        train_states, y = batch
        train_states = torch.stack(train_states)
        # train_states = (chunks, batch length, hid_dim)

        states_padded = torch.zeros((max_chunk_length, train_states.shape[1], train_states.shape[2]))
        states_padded[:train_states.shape[0], :, :] = train_states
        # states_padded = (max_chunk:10, batch length, hid_dim:768)

        decision = y.reshape(y.shape[0], 1)
        # decision = (batch length, 1)

        optimizer.zero_grad()

        logits = model.forward(states_padded)
        loss = model.loss_function(logits, decision)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(trainloader)


def evaluate(model, evalloader, max_chunk_length):

    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for step, batch in enumerate(
                tqdm(evalloader, desc="Iteration")):

            eval_states, y = batch
            eval_states = torch.stack(eval_states)
            # eval_states = (chunks, batch_len, hid_dim)

            states_padded = torch.zeros((max_chunk_length, eval_states.shape[1], eval_states.shape[2]))
            states_padded[:eval_states.shape[0], :, :] = eval_states
            # states_padded = (max_chunk:10, batch_len, hid_dim:768)

            decision = y.reshape(y.shape[0], 1)
            # decision = (batch_len, 1)

            logits = model.forward(states_padded)
            loss = model.loss_function(logits, decision)
            epoch_loss += loss.item()

    return epoch_loss / len(evalloader)


def test(model, testloader, max_chunk_length):

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for step, batch in enumerate(
                tqdm(testloader, desc="Iteration")):

            eval_states, y = batch
            eval_states = torch.stack(eval_states)
            # eval_states = (chunks, batch_len, hid_dim: 768)

            states_padded = torch.zeros((max_chunk_length, eval_states.shape[1], eval_states.shape[2]))
            states_padded[:eval_states.shape[0], :, :] = eval_states
            # states_padded = (max_chunk:10, batch_len, hid_dim)

            decision = y.reshape(y.shape[0], 1)
            # decision = (batch_len, 1)

            logits = model.forward(states_padded)
            prediction = (logits > 0).int()

            prediction = prediction.detach().cpu().numpy()
            decision = decision.to('cpu').numpy()

            y_pred.extend(prediction)
            y_true.extend(decision)

    return y_true, y_pred


def main():
    # If there's a GPU available...
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_paths = ["../6_arg_extraction/extracted_arg/s_20.npy",
                  "../6_arg_extraction/extracted_arg/s_30.npy",
                  "../6_arg_extraction/extracted_arg/s_40.npy",
                  "../6_arg_extraction/extracted_arg/s_50.npy",
                  "../6_arg_extraction/extracted_arg/s_100.npy"]

    ft_bert_model_paths = ["../paper_models/s_20",
                           "../paper_models/s_30",
                           "../paper_models/s_40",
                           "../paper_models/s_50",
                           "../paper_models/s_100"]

    # make a folder to store hidden states, if it doesn't exist yet.
    if not os.path.exists("tobert_hidden_states"):
        os.makedirs("tobert_hidden_states")

    hid_states_paths = ["tobert_hidden_states/hid_states_s_20.npy",
                        "tobert_hidden_states/hid_states_s_30.npy",
                        "tobert_hidden_states/hid_states_s_40.npy",
                        "tobert_hidden_states/hid_states_s_50.npy",
                        "tobert_hidden_states/hid_states_s_100.npy"]

    tobert_models = ["tobert_models/s_20",
                     "tobert_models/s_30",
                     "tobert_models/s_40",
                     "tobert_models/s_50",
                     "tobert_models/s_100"]

    model_id = ["s_20", "s_30", "s_40", "s_50", "s_100"]

    # path to save all the evaluations:
    eval_path = "eval_results_s.csv"

    seeds = [42]

    for seed in seeds:

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True

        for x in range(len(data_paths)):

            data_path = data_paths[x]
            ft_bert_model_path = ft_bert_model_paths[x]
            hid_states_path = hid_states_paths[x]
            tobert_model_dir = tobert_models[x]

            if not os.path.exists(tobert_model_dir):
                os.makedirs(tobert_model_dir)

            max_chunk_length = 10

            ### Prepare Data ###
            # for the first time, hidden_states don't exist.
            if not os.path.exists(hid_states_path):
                data_for_tobert = predict_papers_using_bert(ft_bert_model_path, data_path,
                                                            device, hid_states_path)
            else:
                data_for_tobert = load_states_data(hid_states_path)

            train_data, valid_data, test_data = prepare_data_for_tobert(
                data_for_tobert, max_chunk_length, seed)
            trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
            validloader = torch.utils.data.DataLoader(valid_data, batch_size=128, shuffle=False)
            testloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

            ### Build Model ###
            hidden_dim = 768
            enc_layers = [2]
            enc_heads = 8
            dim_ffs = [512]
            dropouts = [0.1]
            approaches = ["mean"]

            hp_grid = [(i, j, k, m) for i in enc_layers for j in dim_ffs for k in dropouts for m in approaches]

            for (i, j, k, l) in hp_grid:

                # define model path
                tobert_model_name = model_id[x] + '_' + l + "_seed" + str(seed) + "_lay" + str(i) + "_ff" + str(j)\
                                    + "_drop" + str(k) + "_tobert.pt"
                tobert_model_path = os.path.join(tobert_model_dir, tobert_model_name)

                # initialize model
                tobert_model = ToBERT_Encoder(
                    hidden_dim, i, enc_heads, max_chunk_length, j, k, l)
                tobert_model.set_class_weights(train_data)

                ### Start Training ###
                patience = 3
                early_stopping = EarlyStopping(patience=patience, verbose=True)
                optimizer = optim.SGD(tobert_model.parameters(), lr=0.00005, momentum=0.9)
                clip = 1
                epochs = 100
                for epoch in trange(epochs, desc="Epoch"):
                    start_time = time.time()
                    _ = train(tobert_model, trainloader, max_chunk_length, optimizer, clip)
                    valid_loss = evaluate(tobert_model, validloader, max_chunk_length)
                    end_time = time.time()
                    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                    print("Epoch: {} | Time: {}m {}s".format(epoch + 1, epoch_mins, epoch_secs))

                    early_stopping(valid_loss, tobert_model, tobert_model_path)
                    if early_stopping.early_stop:
                        print("Early Stopping")
                        break

                ### Start Evaluation ###
                tobert_model.load_state_dict(torch.load(tobert_model_path))
                y_true, y_pred = test(tobert_model, testloader, max_chunk_length)
                f1_macro, f1_micro = [round(precision_recall_fscore_support(y_true, y_pred, average='macro')[2], 4),
                                      round(precision_recall_fscore_support(y_true, y_pred, average='micro')[2], 4)]

                eval_results = pd.DataFrame({"model_id": model_id[x], "seed": seed, "enc_layer": i,
                                             "ff_dim": j, "dropout": k, "approach": l, "f1_macro": f1_macro,
                                             "f1_micro": f1_micro}, index=[0])
                if not os.path.exists(eval_path):
                    eval_results.to_csv(eval_path, index=False)
                else:
                    eval_results.to_csv(eval_path, index=False, mode='a', header=False)


if __name__ == "__main__":
    main()
