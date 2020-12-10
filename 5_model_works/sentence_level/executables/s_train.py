from __future__ import absolute_import, division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer)
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, classification_report

import numpy as np
from transformers import AdamW
from scipy.special import softmax
from transformers import get_linear_schedule_with_warmup
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import log_loss
import csv
import random
import sklearn
import argparse


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

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


class DataPrecessForSingleSentence(object):

    def __init__(self, bert_tokenizer, max_workers=10):
        self.bert_tokenizer = bert_tokenizer
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def get_input(self, dataset, max_seq_len):
        sentences = dataset.iloc[:, 0].tolist()
        labels = dataset.iloc[:, 1].astype('int64', copy=True).tolist()
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments, labels

    def trunate_and_pad(self, seq, max_seq_len):
        if len(seq) > (max_seq_len - 2):
            seq = seq[0:(max_seq_len - 2)]
        seq = ['[CLS]'] + seq + ['[SEP]']
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        padding = [0] * (max_seq_len - len(seq))
        seq_mask = [1] * len(seq) + padding
        seq_segment = [0] * len(seq) + padding
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment


def gen_dataloader(tokenizer, data_file, max_seq_len, batch_size):

    processor = DataPrecessForSingleSentence(bert_tokenizer=tokenizer)
    data = pd.read_csv(data_file, sep='\t', usecols=['text', 'position'], dtype={'text': str, 'position': str})
    data.loc[data['position'] == " NEG", 'position'] = 0
    data.loc[data['position'] == " NA", 'position'] = 1
    data.loc[data['position'] == " POS", 'position'] = 0
    seqs, seq_masks, seq_segments, labels = processor.get_input(
        dataset=data, max_seq_len=max_seq_len)
    seqs = torch.tensor(seqs, dtype=torch.long)
    seq_masks = torch.tensor(seq_masks, dtype=torch.long)
    seq_segments = torch.tensor(seq_segments, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    data = TensorDataset(seqs, seq_masks, seq_segments, labels)
    dataloader = DataLoader(dataset=data, batch_size=batch_size)
    return dataloader


def test(model, test_dataloader):
    true_labels = []
    pred_labels = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_segments, b_labels = batch
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            pred_labels.append(logits.detach().cpu().numpy())
            true_labels.append(b_labels.to('cpu').numpy())
    prediction = np.concatenate(pred_labels, axis=0)
    prediction = np.argmax(prediction, axis=1).flatten()
    truth = np.concatenate(true_labels, )
    mi_f1_score = sklearn.metrics.f1_score(truth, np.array(prediction), average='micro')
    ma_f1_score = sklearn.metrics.f1_score(truth, np.array(prediction), average='macro')
    w_f1_score = sklearn.metrics.f1_score(truth, np.array(prediction), average='weighted')
    test_accuracy = (sum(np.array(prediction) == np.array(truth)) / float(len(truth)))
    print("micro_f1_score:", mi_f1_score)
    print("macro_f1_score:", ma_f1_score)
    print("weighted_f1_score:", w_f1_score)
    print("test_accuracy:", test_accuracy)
    return mi_f1_score, ma_f1_score, w_f1_score, test_accuracy


def train(arg, device):
    VOCAB = 'vocab.txt'
    if arg.finetuned:
        tokenizer = BertTokenizer.from_pretrained(os.path.join(arg.model, VOCAB), do_lower_case=False,
                                                  num_labels=arg.num_labels)
    else:
        tokenizer = BertTokenizer.from_pretrained(arg.model, do_lower_case=False, num_labels=arg.num_labels)
    model = BertForSequenceClassification.from_pretrained(arg.model, num_labels=arg.num_labels)
    model.to(device)
    lr = arg.learning_rate
    num_total_steps = 1000
    num_warmup_steps = 100

    train_dataloader = gen_dataloader(tokenizer=tokenizer, data_file=arg.traindata_file,
                                      max_seq_len=arg.max_seq_length, batch_size=arg.batch_size)
    validation_dataloader = gen_dataloader(tokenizer=tokenizer, data_file=arg.devdata_file,
                                           max_seq_len=arg.max_seq_length, batch_size=arg.batch_size)
    test_dataloader = gen_dataloader(tokenizer=tokenizer, data_file=arg.testdata_file,
                                     max_seq_len=arg.max_seq_length, batch_size=arg.batch_size)

    ### In PyTorch-Transformers, optimizer and schedules are splitted and instantiated like this:
    optimizer = AdamW(model.parameters(), lr=lr,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,  # Default value in run_glue.py
                                                num_training_steps=num_total_steps)

    epochs = arg.epochs
    patience = arg.patience
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # Store our loss and accuracy for plotting
    valid_losses = []
    train_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    # trange is a tqdm wrapper around the normal python range
    for _ in tqdm(range(epochs)):

        # Training, Set our model to training mode (as opposed to evaluation mode)
        model.train()
        # total_step = len(train_dataloader)

        # Train the data for one epoch
        for i, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_seq_segments, b_labels = batch
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            train_losses.append(loss.item())
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # test model on validation set
        model.eval()
        with torch.no_grad():

            for i, batch in enumerate(validation_dataloader):
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_segments, b_labels = batch
                # Forward pass
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break
        valid_losses = []
        train_losses = []

    # save model

    if arg.save_model:
        model.save_pretrained(arg.outdir_model)
        tokenizer.save_pretrained(arg.outdir_model)
        print("saving model")

    # test model on test set
    test(model, test_dataloader)


def predict(arg, device):
    VOCAB = 'vocab.txt'
    if arg.finetuned:
        tokenizer = BertTokenizer.from_pretrained(os.path.join(arg.model, VOCAB), do_lower_case=False,
                                                  num_labels=arg.num_labels)
    else:
        tokenizer = BertTokenizer.from_pretrained(arg.model, do_lower_case=False, num_labels=arg.num_labels)

    test_dataloader = gen_dataloader(tokenizer=tokenizer, data_file=arg.testdata_file, max_seq_len=arg.max_seq_length,
                                     batch_size=arg.batch_size)

    model = BertForSequenceClassification.from_pretrained(arg.outdir_model, output_hidden_states=True,
                                                          num_labels=arg.num_labels)
    model.to(device)

    # test model on test set
    path_pred = arg.prediction_file
    path_repre = arg.representation_file
    f = open(path_pred, 'w+', newline='')
    writer = csv.writer(f)
    writer.writerow(['label'])

    true_labels = []
    pred_labels = []
    CLS_representation = np.array([])
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_segments, b_labels = batch
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            # outputs: tuple(logits, hidden_state)
            # hidden_state: (batch_size, sequence_length, hidden_size)
            logits = outputs[0]
            hidden_state = outputs[1]
            output_of_the_embeddings = hidden_state[0]
            output_of_each_layer = hidden_state[1:]
            CLS_repre = output_of_the_embeddings[:, 0, :]

            if len(CLS_representation) == 0:
                CLS_representation = CLS_repre.cpu().numpy()
            else:
                CLS_representation = np.concatenate((CLS_representation, CLS_repre.cpu().numpy()), axis=0)

            pred_labels.append(logits.cpu().numpy())
            true_labels.append(b_labels.to('cpu').numpy())
    prediction = np.concatenate(pred_labels, axis=0)
    np.save(path_repre, CLS_representation)
    prediction = softmax(prediction, axis=1)
    for i in range(0, len(prediction)):
        writer.writerow([prediction[i][0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_length', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--traindata_file', type=str)
    parser.add_argument('--devdata_file', type=str)
    parser.add_argument('--testdata_file', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--num_labels', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--save_model', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--outdir_model', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--finetuned', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--task', type=str)
    parser.add_argument('--prediction_file', type=str)
    parser.add_argument('--representation_file', type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print("Using Device:", device)
    print("GPUs:", n_gpu)

    if args.task == 'train':
        train(args, device)
    else:
        predict(args, device)
