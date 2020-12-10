import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from annotation.alpha import Annotation
from new_preparation import get_all_lengths, get_segments_in_one_sentence
from calculate_sentence_position import get_sentence_position
from statistics import mean
import numpy as np
import os
import contextlib


def get_individual_positions(annotator, file_for_majority):
    filename = "../../3_annotation_study/data/sampling/samples_withalpha.csv"
    sample_data = pd.read_csv(filename, sep=',').to_numpy()
    all_positions = []
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
                    sentence_text = line[1][1:]
                    annotation_text = review[1]
                    sentence_begin = annotation_text.find(sentence_text)
                    sentence_end = sentence_begin + len(sentence_text)
                    sentence = np.array(
                        [sentence_begin, sentence_end, sentence_text])
                    positions_in_sentence = get_segments_in_one_sentence(
                                            sentence, annotators_position)
                    position = get_sentence_position(positions_in_sentence)
                    if position == 'ERROR IN DATA':
                        position = 'NA'
                    all_positions.append(position)
    return all_positions


def get_only_reviews_from(annotator, file_for_majority):
    filename = "../../3_annotation_study/data/sampling/samples_withalpha.csv"
    sample_data = pd.read_csv(filename, sep=',').to_numpy()
    this_annotators_reviews = []
    for row in sample_data:
        if annotator in row[5]:
            this_annotators_reviews.append(row[0])
    y_true_reduced = []
    sentence_ids = []
    for line in file_for_majority:
        for review_id in this_annotators_reviews:
            if review_id in line[0]:
                y_true_reduced.append(line[2][1:])
                sentence_ids.append(line[0])
    return y_true_reduced, sentence_ids


def only_argument_detection(positions):
    arguments_detected = []
    for pos in positions:
        if pos == 'POS':
            arguments_detected.append('ARG')
        elif pos == 'NEG':
            arguments_detected.append('ARG')
        elif pos == 'NA':
            arguments_detected.append('NON_ARG')
        else:
            arguments_detected.append(pos)
    return arguments_detected


def only_pos_neg_in_ground_truth(y_true, y_pred):
    new_y_true = y_true.copy()
    new_y_pred = y_pred.copy()
    i = 0
    while i < len(new_y_true):
        if 'NA' in new_y_true[i]:
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
        if 'NA' in new_y_pred[i]:
            new_y_true.pop(i)
            new_y_pred.pop(i)
            i = i - 1
        i = i + 1
    return new_y_true, new_y_pred


def print_statistics(annotators, file_for_majority):
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
        both_y_true, sentence_ids = get_only_reviews_from(
                                    annotator, file_for_majority)
        # print(both_y_true)
        annotator_positions = get_individual_positions(
                                    annotator, file_for_majority)
        both_y_pred = annotator_positions
        both_acc = accuracy_score(both_y_true, both_y_pred)
        both_accuracies.append(both_acc)
        both_micro_f1.append(f1_score(
                             both_y_true, both_y_pred, average='micro'))
        both_macro_f1.append(f1_score(
                             both_y_true, both_y_pred, average='macro'))
        both_weighted_f1.append(f1_score(
                             both_y_true, both_y_pred, average='weighted'))

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

    # Pos-Neg accuracies
    # print('Non-Arguments removed in ground truth: accuracies: ')
    # print(na_removed_accuracies)
    # print('Non-Arguments removed in ground truth: mean accuracy: ')
    # print(mean(na_removed_accuracies))

    print('Non-Arguments removed in ground truth and prediction: accuracies: ')
    print(only_pos_neg_accuracies)
    print('mean accuracy: ')
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


if __name__ == "__main__":
    filename = "../data/sentences_just_one_position.csv"
    file_for_majority = pd.read_csv(filename, sep='\t', header=None).to_numpy()
    annotators = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
    print_statistics(annotators, file_for_majority)
