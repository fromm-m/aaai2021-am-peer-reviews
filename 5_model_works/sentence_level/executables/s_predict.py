from __future__ import absolute_import, division, print_function
import os
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


class DataPrecessForSingleSentence(object):

    def __init__(self, bert_tokenizer, max_workers=10):
        self.bert_tokenizer = bert_tokenizer
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def get_input(self, dataset, max_seq_len=100):
        sentences = dataset.iloc[:, 0].tolist()
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments

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


def gen_dataloader(tokenizer, data_file):
    processor = DataPrecessForSingleSentence(bert_tokenizer=tokenizer)
    data = pd.read_csv(data_file, usecols=['sentence'], dtype={'sentence': str})
    seqs, seq_masks, seq_segments = processor.get_input(
        dataset=data, max_seq_len=100)
    seqs = torch.tensor(seqs, dtype=torch.long)
    seq_masks = torch.tensor(seq_masks, dtype=torch.long)
    seq_segments = torch.tensor(seq_segments, dtype=torch.long)
    data = TensorDataset(seqs, seq_masks, seq_segments)
    dataloader = DataLoader(dataset=data, batch_size=16)
    return dataloader


def predict(model_dir, test_file, prediction_file):

    print("loading model...")
    model = BertForSequenceClassification.from_pretrained(model_dir,
                                                          output_attentions=True,
                                                          output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(model_dir,
                                              do_lower_case=False)
    model.to(device)

    test_dataloader = gen_dataloader(tokenizer, test_file)

    pred_labels = []
    CLS_representation = np.array([])

    model.eval()
    print("Starting Prediction")

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_segments = batch
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=b_segments, attention_mask=b_input_mask)
            # outputs tuple(logits, hidden_state)
            # hidden_state: (batch_size, sequence_length, hidden_size)
            logits = outputs[0]
            hidden_state = outputs[1]
            output_of_last_layer = hidden_state[-1]
            CLS_repre = output_of_last_layer[:, 0, :]

            if len(CLS_representation) == 0:
                CLS_representation = CLS_repre.cpu().numpy()
            else:
                CLS_representation = np.concatenate((CLS_representation,
                                                     CLS_repre.cpu().numpy()), axis=0)

            pred_labels.append(logits.cpu().numpy())
            print("Batch {} completed".format(i))

    prediction = np.concatenate(pred_labels, axis=0)
    np.save("representations.npy", CLS_representation)
    prediction = softmax(prediction, axis=1)

    # save the predicted confidence values in a tsv file.
    data = pd.read_csv(test_file)
    data["confidence"] = prediction
    data.to_csv(prediction_file, sep="\t", index=False)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()

    model_dir = "model/"
    data_path = "data/all_reviews_by_sentences.csv"
    prediction_path = "data/reviews_with_predictions.tsv"
    predict(model_dir, data_path, prediction_path)
