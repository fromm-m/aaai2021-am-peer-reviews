from __future__ import absolute_import, division, print_function

import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import (BertForTokenClassification, BertTokenizer)
from spacy.lang.en import English
import pandas as pd
import re
import numpy as np

nlp = English()
word_tokenize = nlp.Defaults.create_tokenizer(nlp)
recog_labels = ["non", "arg", "[CLS]", "[SEP]"]


class BertTLC(BertForTokenClassification):
    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                valid_ids=None,
                labels=None,
                attention_mask_label=None):
        """
        :param input_ids: Indices of input sequence tokens in the vocabulary
        :param token_type_ids: Segment token indices to indicate first and second portions of the inputs
        :param attention_mask: Mask to avoid performing attention on padding token indices
        :param labels: Labels for computing the masked language modeling loss
        :param valid_ids: Selects tokens having a valid label
        :param attention_mask_label: Keeps active parts of the loss
        :returns:
            outputs: List containing loss, prediction scores, hidden states and attentions

        """
        outputs = self.bert(input_ids,
                            token_type_ids,
                            attention_mask,
                            head_mask=None)
        sequence_output = outputs[0]

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,
                                   max_len,
                                   feat_dim,
                                   dtype=torch.float32,
                                   device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]

        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        outputs = (logits, ) + outputs[2:]

        return outputs


class Predictor:
    def __init__(self, model_dir, label_list, max_seq_length):
        self.model, self.tokenizer = self.load_model(model_dir)
        self.label_map = {label: i for i, label in enumerate(label_list, 1)}
        self.max_seq_length = max_seq_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def load_model(model_dir):
        print("loading model...")
        model = BertTLC.from_pretrained(model_dir,
                                        output_attentions=True,
                                        output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained(model_dir,
                                                  do_lower_case=False)
        return model, tokenizer

    def tokenize(self, text):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        for word in words:
            token = self.tokenizer.tokenize(word.text)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, text, topic=None):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        topic_position = float("inf")
        token_length = 0
        topic_tokens, topic_valid_pos = [], []

        if topic:
            topic_position = len(tokens)
            topic_tokens, topic_valid_pos = self.tokenize(topic)

        if len(tokens) >= 512 - 1 - token_length:
            tokens = tokens[0:(self.max_seq_length - 2 - token_length)]
            valid_positions = valid_positions[0:(self.max_seq_length - 2 -
                                                 token_length)]

        # insert "[CLS]"
        tokens.insert(0, "[CLS]")
        valid_positions.insert(0, 1)
        # insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(1)
        tokens.extend(topic_tokens)
        valid_positions.extend(topic_valid_pos)
        segment_ids = []

        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids, input_mask, segment_ids, valid_positions, topic_position

    def predict(self, text_id, text, topic=None):
        # print("predicting...", text_id)
        input_ids, input_mask, segment_ids, valid_ids, topic_position = self.preprocess(
            text, topic)
        input_ids = torch.tensor([input_ids],
                                 dtype=torch.long,
                                 device=self.device)
        input_mask = torch.tensor([input_mask],
                                  dtype=torch.long,
                                  device=self.device)
        segment_ids = torch.tensor([segment_ids],
                                   dtype=torch.long,
                                   device=self.device)
        valid_ids = torch.tensor([valid_ids],
                                 dtype=torch.long,
                                 device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, segment_ids, input_mask, valid_ids)

        logits, hidden_states, attentions = outputs
        logits = F.softmax(logits, dim=2)
        hidden_state = hidden_states[-1][0][0]
        attention = attentions[-1][0]

        logits = logits.detach().cpu().numpy()
        hidden_state = hidden_state.detach().cpu().numpy()
        attention = attention.detach().cpu().numpy()

        logits_confidence = [
            values[self.label_map['arg']].item() for values in logits[0]
        ]
        logits = []
        attention_weights = []
        pos = 0

        for index, mask in enumerate(valid_ids[0]):
            if index == 0:
                attention_weights.append(attention[:, index, index].tolist())
                continue
            if mask == 1 and index < topic_position:
                logits.append(logits_confidence[index - pos])
                attention_weights.append(attention[:, index, index].tolist())
            else:
                pos += 1
        logits.pop()

        words = word_tokenize(text)

        if len(logits) != len(words):
            print(text_id, len(words), len(logits), len(logits_confidence))
        output_confidence = {
            "sentence_id": text_id,
            "sentence": words,
            "confidence": logits
        }
        output_hidden = {
            "sentence_id": text_id,
            "representation": hidden_state
        }
        output_attention = [{
            "sentence_id": text_id,
            "attention": attention_weights
        }]

        return output_confidence, output_hidden, output_attention

    @staticmethod
    def save_confidence(scores, out_dir):
        lines = []
        for score in scores:
            match_id = re.match(r'([a-z0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)',
                                score['sentence_id'])
            conference, paper_id, review_id, sentence_id = match_id.groups()
            lines.append(
                dict(conference=conference,
                     paper_id=paper_id,
                     review_id=review_id,
                     sentence_id=sentence_id,
                     sentence=score['sentence'],
                     confidence=score['confidence']))
        data = pd.DataFrame(lines,
                            columns=[
                                'conference', 'paper_id', 'review_id',
                                'sentence_id', 'sentence', 'confidence'
                            ])
        data.to_csv(out_dir, index=False)

    @staticmethod
    def save_hidden(hidden_states, out_dir):
        np.save(out_dir, hidden_states)

    @staticmethod
    def save_attention(attentions, out_dir):
        with open(out_dir, 'w') as outfile:
            json.dump(attentions, outfile)


def read_file(file):
    data = pd.read_csv(file)
    sen_ids, sens = data['sentence_id'], data['sentence']
    sentences = zip(sen_ids, sens)
    return sentences


def run_predict(data_dir, conference, model_dir, seq_length, label_list,
                output_dir):
    """
    Implement predictions.
    :param data_dir: Directory of raw review file.
    :param conference: Conference name.
    :param model_dir: Directory of saved model.
    :param seq_length: Max. sequence length of model.
    :param label_list: List of labels.
    :param output_dir: Directory of output file.

    :return: 3 files containing confidence of each token,
    CLS vector of each sentence and attention weights of each token, separately.
    """
    sentences = read_file(data_dir + conference + '.csv')
    predictor = Predictor(model_dir, label_list, seq_length)
    scores, hidden_states, attentions = [], [], []

    for sentence in tqdm(sentences, desc="sentence"):
        out1, out2, out3 = predictor.predict(sentence[0], sentence[1])
        scores.append(out1)
        hidden_states.append(out2)
        attentions.append(out3)
    predictor.save_confidence(scores,
                              output_dir + conference + '_confidence.csv')
    predictor.save_hidden(np.array(hidden_states),
                          output_dir + conference + '_rep.npy')
    predictor.save_attention(attentions,
                             output_dir + conference + '_attention.json')


if __name__ == "__main__":
    # Implementation Example
    model_path = '../models/recog_notopic/bert-large-cased/rev100_bl_recog_notopic16'
    run_predict('../data/raw_reviews', 'graph20', model_path, 85, recog_labels,
                '../predictions/')
