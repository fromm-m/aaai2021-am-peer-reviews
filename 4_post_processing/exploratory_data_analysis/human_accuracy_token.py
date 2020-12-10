from spacy.lang.en import English
import pandas as pd
import numpy as np
import os
import contextlib
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from statistics import mean
from new_preparation import get_all_lengths, get_segments_in_one_sentence
from human_accuracy_sentence import get_only_reviews_from
from annotation.alpha import Annotation

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def split_review_sentence(sentence_id, sen, seg):
    sum_len = 0
    """
    Split 1 sentence in review into [segment, tag]
    """
    sen_splitting = {}
    seg_tag = []
    sen = sen.strip()
    spans, tags = eval(seg)
    splitting = list(zip(spans.split(';'),
                         tags.split(';')))[:-1]
    prev_start = 0
    prev_extend = 0
    first = True
    for elem in splitting:
        span, tag = eval(elem[0]), elem[1]
        if tag == 'POS':
            tag = 'pro'
        elif tag == 'NEG':
            tag = 'con'
        elif tag == 'NA':
            tag = 'non'
        else:
            tag = 'non'
            print(sentence_id, sen, seg)
        start, extend = int(span[0]), int(span[1])
        sum_len += extend
        if start > 0 and first is True:
            seg_tag.append([sen[0: start], 'non'])
        first = False
        if prev_start + prev_extend < start:
            seg_tag.append([sen[prev_start + prev_extend: start], 'non'])
        prev_start = start
        prev_extend = extend
        seg_tag.append([sen[start: start + extend], tag])
    if len(sen) > start + extend:
        seg_tag.append([sen[start + extend: len(sen)], 'non'])
    sen_splitting[sentence_id] = ['paper quality', seg_tag]
    return sen_splitting


# modified from .... file
def create_sen_one_line(sentences):
    for _, sens in sentences.items():
        sentence_text = ''
        labels = []
        complete_sentence = ''
        for seg, tag in sens[1]:
            complete_sentence = complete_sentence + seg
            seg = seg.strip()
            if seg:
                sentence_text = ' '.join([sentence_text, seg])
                tokens = tokenizer(seg)
                n_of_tokens = len(tokens)
                for t in range(len(tokens)):
                    if str(tokens[t]) == ' ':
                        n_of_tokens -= 1
                    if len(tokens[t]) == 1:
                        n_of_label = len(labels) - 1
                labels.extend([tag] * n_of_tokens)
        complete_sentence_tokens = tokenizer(complete_sentence)
        if len(complete_sentence_tokens) < len(labels):
            labels.pop(n_of_label)
    return labels


def get_individual_positions(annotator, file_for_majority, sentence_ids):
    filename = "../../3_annotation_study/data/sampling/samples_withalpha.csv"
    sample_data = pd.read_csv(filename, sep=',').to_numpy()
    all_positions = []
    a_lens = []
    for review in sample_data:
        if annotator in review[5]:
            folder = "../../3_annotation_study/data/cleaned_annotation"
            current_filename = "/" + review[0] + "_anno.csv"
            path = folder + current_filename
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                annotation = Annotation.read_from_file(path)
            all_lengths = get_all_lengths()
            length = all_lengths[review[0]]
            annotation.fill_gap(length)
            annotators_position = np.empty(length, dtype=object)
            for u in annotation.units[annotator]:
                for i in range(u.begin, u.end):
                    annotators_position[i] = u.tag
            for line in file_for_majority:
                if review[0] in line[0]:
                    sent_id = line[0]
                    row = file_for_majority[
                        file_for_majority[:, 0] == sent_id][0]
                    sentence_text = line[1][1:]
                    sentence_text = row[1][1:]
                    annotation_text = review[1]
                    sentence_begin = annotation_text.find(sentence_text)
                    sentence_end = sentence_begin + len(sentence_text)
                    sentence = np.array(
                        [sentence_begin, sentence_end, sentence_text])
                    positions_in_sentence = get_segments_in_one_sentence(
                        sentence, annotators_position)
                    sent = split_review_sentence(
                        sent_id, sentence_text, positions_in_sentence)
                    annotator_positions_sentence = create_sen_one_line(sent)
                    a_lens.append(len(annotator_positions_sentence))
                    annotator_positions_sentence = [
                        'non' if p == 'ERROR IN DATA'
                        else p for p in annotator_positions_sentence]
                    all_positions.extend(annotator_positions_sentence)
    return all_positions, a_lens


def only_argument_detection(positions):
    arguments_detected = []
    for pos in positions:
        if pos == 'pro':
            arguments_detected.append('ARG')
        elif pos == 'con':
            arguments_detected.append('ARG')
        elif pos == 'non':
            arguments_detected.append('NON_ARG')
        else:
            arguments_detected.append(pos)
    return arguments_detected


def only_pos_neg_in_ground_truth(y_true, y_pred):
    new_y_true = y_true.copy()
    new_y_pred = y_pred.copy()
    i = 0
    while i < len(new_y_true):
        if 'non' in new_y_true[i]:
            new_y_true.pop(i)
            new_y_pred.pop(i)
            i = i - 1
        i = i + 1
    return new_y_true, new_y_pred


def only_pos_neg_in_prediction(y_true, y_pred):
    new_y_true = y_true.copy()
    new_y_pred = y_pred.copy()
    i = 0
    while i < len(new_y_pred):
        if 'non' in new_y_pred[i]:
            new_y_true.pop(i)
            new_y_pred.pop(i)
            i = i - 1
        i = i + 1
    return new_y_true, new_y_pred


if __name__ == "__main__":
    annotators = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
    filename = "../data/text_sentences_majority.csv"
    file_for_majority = pd.read_csv(filename, sep='\t', header=None).to_numpy()

    both_accuracies = []
    arg_detection_accuracies = []
    na_removed_accuracies = []
    only_pos_neg_accuracies = []

    both_micro_f1 = []
    arg_detection_micro_f1 = []
    only_pos_neg_micro_f1 = []

    both_macro_f1 = []
    arg_detection_macro_f1 = []
    only_pos_neg_macro_f1 = []

    both_weighted_f1 = []
    arg_detection_weighted_f1 = []
    only_pos_neg_weighted_f1 = []

    for annotator in annotators:
        ground_truth_positions = []
        g_lens = []
        both_y_true, sentence_ids = get_only_reviews_from(
            annotator, file_for_majority)
        for line in range(len(sentence_ids)):
            sent_id = sentence_ids[line]
            row = file_for_majority[file_for_majority[:, 0] == sent_id][0]
            sentence_text = row[1][1:]
            spans_and_tags = row[2]
            sent = split_review_sentence(
                sent_id, sentence_text, spans_and_tags)
            ground_truth_sentence = create_sen_one_line(sent)
            g_lens.append(len(ground_truth_sentence))
            ground_truth_positions.extend(ground_truth_sentence)
        annotator_positions, a_lens = get_individual_positions(
            annotator, file_for_majority, sentence_ids)

        for i in range(len(g_lens)):
            if g_lens[i] != a_lens[i]:
                print('The number of tokens is different.')
                print('The problematic sentence: ')
                print(sentence_ids[i])

        both_y_true = ground_truth_positions
        both_y_pred = annotator_positions
        both_acc = accuracy_score(both_y_true, both_y_pred)
        both_accuracies.append(both_acc)
        both_micro_f1.append(
            f1_score(both_y_true, both_y_pred, average='micro'))
        both_macro_f1.append(
            f1_score(both_y_true, both_y_pred, average='macro'))
        both_weighted_f1.append(
            f1_score(both_y_true, both_y_pred, average='weighted'))

        arg_detection_y_true = only_argument_detection(both_y_true)
        arg_detection_y_pred = only_argument_detection(both_y_pred)
        arg_detection_acc = accuracy_score(
            arg_detection_y_true, arg_detection_y_pred)
        arg_detection_accuracies.append(arg_detection_acc)
        arg_detection_micro_f1.append(f1_score(
            arg_detection_y_true, arg_detection_y_pred, average='micro'))
        arg_detection_macro_f1.append(f1_score(
            arg_detection_y_true, arg_detection_y_pred, average='macro'))
        arg_detection_weighted_f1.append(f1_score(
            arg_detection_y_true, arg_detection_y_pred, average='weighted'))

        na_removed_y_true, na_removed_y_pred = only_pos_neg_in_ground_truth(
            both_y_true, both_y_pred)
        na_removed_acc = accuracy_score(na_removed_y_true, na_removed_y_pred)
        na_removed_accuracies.append(na_removed_acc)

        only_pos_neg_y_true, only_pos_neg_y_pred = only_pos_neg_in_prediction(
            na_removed_y_true, na_removed_y_pred)
        only_pos_neg_acc = accuracy_score(
            only_pos_neg_y_true, only_pos_neg_y_pred)
        only_pos_neg_accuracies.append(only_pos_neg_acc)
        only_pos_neg_micro_f1.append(f1_score(
            only_pos_neg_y_true, only_pos_neg_y_pred, average='micro'))
        only_pos_neg_macro_f1.append(f1_score(
            only_pos_neg_y_true, only_pos_neg_y_pred, average='macro'))
        only_pos_neg_weighted_f1.append(f1_score(
            only_pos_neg_y_true, only_pos_neg_y_pred, average='weighted'))

# Accuracies for Pos-Neg-NonArg
print('Positive, Negative, Non-Argument accuracies: ')
print(both_accuracies)
print('Positive, Negative, Non-Argument mean accuracy: ')
print(mean(both_accuracies))
print('micro F1 scores:')
print(both_micro_f1)
print('mean micro F1 score:')
print(mean(both_micro_f1))
print('macro F1 scores:')
print(both_macro_f1)
print('mean macro F1 score:')
print(mean(both_macro_f1))
print('weighted F1 scores:')
print(both_weighted_f1)
print('mean weighted F1 score:')
print(mean(both_weighted_f1))

# Arg-Non-Arg accuracies
print('Argument, Non-Argument accuracies: ')
print(arg_detection_accuracies)
print('Argument, Non-Argument mean accuracy: ')
print(mean(arg_detection_accuracies))
print('micro F1 scores:')
print(arg_detection_micro_f1)
print('mean micro F1 score:')
print(mean(arg_detection_micro_f1))
print('macro F1 scores:')
print(arg_detection_macro_f1)
print('mean macro F1 score:')
print(mean(arg_detection_macro_f1))
print('weighted F1 scores:')
print(arg_detection_weighted_f1)
print('mean weighted F1 score:')
print(mean(arg_detection_weighted_f1))

print('Non-Arguments removed in ground truth and prediction: accuracies: ')
print(only_pos_neg_accuracies)
print('Non-Arguments removed in ground truth and prediction: mean accuracy: ')
print(mean(only_pos_neg_accuracies))
print('micro F1 scores:')
print(only_pos_neg_micro_f1)
print('mean micro F1 score:')
print(mean(only_pos_neg_micro_f1))
print('macro F1 scores:')
print(only_pos_neg_macro_f1)
print('mean macro F1 score:')
print(mean(only_pos_neg_macro_f1))
print('weighted F1 scores:')
print(only_pos_neg_weighted_f1)
print('mean weighted F1 score:')
print(mean(only_pos_neg_weighted_f1))
