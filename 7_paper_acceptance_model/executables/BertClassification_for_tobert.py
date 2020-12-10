import numpy as np
import pandas as pd

import torch
from transformers import BertTokenizer, BertForSequenceClassification


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def load_reviews_data(filepath):
    """
    :param filepath: path leading to the extracted arguments
    :return: data containing the fields: paper_id, decision and reviews
    """
    papers = np.load(filepath, allow_pickle=True)
    paper_ids = []
    decisions = []
    reviews = []
    for paper in papers:
        paper_ids.append(paper['paper_id'])
        decisions.append(paper['decision'])
        reviews.append(paper['review_text'])
    df = pd.DataFrame(np.column_stack([paper_ids, decisions, reviews]), columns=['paper_id', 'decision', 'reviews'])
    return df


def preprocess_data_for_bert(review, tokenizer, max_seq_len=512):
    """
    defined for running over one review at a time
    :param review: review text
    :param tokenizer: tokenizes the review
    :param max_seq_len: length of each chunk
    :return: features
    """
    seq_len_adjusted = max_seq_len - 2
    tokens = tokenizer.tokenize(review)
    chunks = [tokens[x:x + seq_len_adjusted] for x in range(0, len(tokens), seq_len_adjusted)]
    features = []
    for chunk in chunks:
        chunk.insert(0, '[CLS]')
        chunk.append('[SEP]')
        input_ids = tokenizer.convert_tokens_to_ids(chunk)
        input_mask = [1] * len(chunk)
        segment_ids = [0] * len(chunk)
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids))
    return features


def predict_review_using_bert(features, model, device):
    """
    defined for running over one review at a time
    :param features:
    :param model:
    :param device:
    :return:
    """
    hidden_states_all = []
    for f in features:
        # update text_id
        input_ids = torch.tensor([f.input_ids], dtype=torch.long, device=device)
        input_mask = torch.tensor([f.input_mask], dtype=torch.long, device=device)
        segment_ids = torch.tensor([f.segment_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask)

        _, hidden_states, _ = outputs
        hidden_state = hidden_states[-1][0][0]

        hidden_state = hidden_state.detach().cpu().numpy()
        hidden_states_all.append(hidden_state)

    return hidden_states_all


def predict_papers_using_bert(model_dir, data_path, device, save_path):
    """
    defined for running over the entire data
    :param model_dir: path to the fine_tuned bert model
    :param data_path: path to the arguments data
    :param device: device on which the file is run
    :param save_path: path for saving the hidden states
    :return: data containing the hidden states by paper and the decision
    """
    data = load_reviews_data(data_path)
    model = BertForSequenceClassification.from_pretrained(model_dir,
                                                          output_attentions=True,
                                                          output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False)

    model.to(device)

    hidden_states_total = []
    for i in range(len(data)):
        hidden_states_by_paper = []
        for review in data["reviews"][i]:
            features = preprocess_data_for_bert(review, tokenizer, 512)
            hidden_states = predict_review_using_bert(features, model, device=device)
            hidden_states_by_paper.extend(hidden_states)
        hidden_states_total.append(hidden_states_by_paper)

    data["hidden_states"] = hidden_states_total
    data = data.drop(['reviews'], axis=1)

    # save the hidden states as a npy file.
    numpy_list = data.to_numpy()
    np.save(save_path, numpy_list)

    return data
