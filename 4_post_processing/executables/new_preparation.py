import numpy as np
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import nltk
# nltk.download('all')
from annotation.alpha import Annotation
# import os

filename = "../../3_annotation_study/data/sampling/samples_withalpha.csv"
sample_data = pd.read_csv(filename, sep=',')
folder = "../../3_annotation_study/data/cleaned_annotation"


def get_all_lengths():
    lengths = {}
    for r in range(len(sample_data["review_id"])):
        lengths[sample_data["review_id"][r]] = len(sample_data["review"][r])
    return(lengths)


def get_decisions(length, raters, annotation):
    current_decisions = np.empty((length, len(raters)), dtype=object)
    rater_nr = 0
    for r in raters:
        column = np.empty(length, dtype=object)
        for u in annotation.units[r]:
            for i in range(u.begin, u.end):
                column[i] = u.tag
        current_decisions[:, rater_nr] = column
        rater_nr = rater_nr + 1
    return current_decisions


def get_majority(row):
    majority = len(row)/2
    (unique, counts) = np.unique(row, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    for f in frequencies:
        if float(f[1]) > majority:
            return f[0]
    return None


def get_majorities(current_decisions, length):
    majority = np.empty(length, dtype=object)
    row_nr = 0
    for row in current_decisions[:, :]:
        majority[row_nr] = get_majority(row)
        row_nr = row_nr + 1
    return majority


def get_next_change(offset, tag_per_offset):
    current_annotation = tag_per_offset[offset]
    next_offset = offset
    while current_annotation == tag_per_offset[next_offset]:
        next_offset = next_offset + 1
        if next_offset >= len(tag_per_offset):
            return None
    return next_offset


def get_segments(tag_per_offset, text):
    tag_list = np.empty((0, 4), dtype=str)
    offset_begin = 0
    offset_end = 0
    while True:
        tag = tag_per_offset[offset_begin]
        if get_next_change(offset_begin, tag_per_offset) is not None:
            offset_end = get_next_change(offset_begin, tag_per_offset)
            sentence = text[offset_begin:offset_end]
            # if tag == None: #Print Segments without agreement
            #    print(sentence)
            data = np.array([[sentence, offset_begin, offset_end, tag]])
            tag_list = np.concatenate((tag_list, data), axis=0)
        else:
            offset_end = len(tag_per_offset)-1
            sentence = text[offset_begin:offset_end]
            # if tag == None: #Print Segments without agreement
            #    print(sentence)
            data = np.array([[sentence, offset_begin, offset_end, tag]])
            tag_list = np.concatenate((tag_list, data), axis=0)
            break
        offset_begin = offset_end
    return tag_list


def get_sentences(text):
    punkt_param = PunktParameters()
    punkt_param.abbrev_types = set(['e.g', 'Fig', 'fig', 'et al', 'no', 'vol',
                                    'p', 'pp', 'Eq', 'et al. (', 'et', 'Ph.D',
                                    'Jan', 'i.e', 'w.r.t', 'vs', 'etc', 'ex',
                                    'sec', 'Ph. D', 'et. al', 'al'])
    tokenizer = PunktSentenceTokenizer(punkt_param)
    sentence_span = np.asarray(list(tokenizer.span_tokenize(text)))
    sentence_text = np.asarray(tokenizer.tokenize(text))
    both = np.concatenate((sentence_span, sentence_text[:, None]), axis=1)
    return both


def remove_leading_characters(sentence):
    sentence_begin = int(sentence[0])
    sentence_text = sentence[2]
    for character in sentence_text:
        if character.isalnum():
            break
        else:
            sentence_begin = sentence_begin + 1
        sentence_text = sentence_text[1:]
    sentence[0] = sentence_begin
    sentence[2] = sentence_text
    return sentence


def get_segments_in_one_sentence(sentence, majority):
    sentence = remove_leading_characters(sentence)
    segment_string = ""
    segment_string_part_one = ""
    segment_string_part_two = ""
    offset_sentence_begin = int(sentence[0])
    offset_sentence_end = int(sentence[1])
    segment_list = get_segments(majority[
        offset_sentence_begin:offset_sentence_end+1], sentence[2])
    for segment in segment_list:
        segment_start = int(segment[1])
        segment_length = int(segment[2]) - int(segment[1])
        position = segment[3]
        if segment_length > 1:
            segment_string_part_one = segment_string_part_one + \
                                      "(" + str(segment_start) + \
                                      "," + str(segment_length) + ");"
            segment_string_part_two = segment_string_part_two + \
                str(position) + ";"
            segment_string = "('" + segment_string_part_one + \
                "', '" + segment_string_part_two + "')"
    return segment_string


def get_segments_in_sentences(majority, sentences, sentence_ids):
    all_segments = np.empty(len(sentences), dtype=object)
    count = 0
    for s in sentences:
        all_segments[count] = get_segments_in_one_sentence(s, majority)
        count = count + 1
    sentence_and_segments = np.concatenate(
        (sentences[:, 2:], all_segments[:, None]), axis=1)
    sentence_id_added = np.concatenate(
        (sentence_ids[:, None], sentence_and_segments), axis=1)
    return sentence_id_added


def n_of_tokens(txt):
    tokenList = nltk.word_tokenize(txt)
    for token in tokenList:
        if not token.isalnum():
            tokenList.remove(token)
    return len(tokenList)


def remove_too_short_segments(segments):
    row_count = 0
    for row in segments[:, :]:
        if (n_of_tokens(row[1]) < 3):
            segments = np.delete(segments, row_count, 0)
            row_count = row_count - 1
            # otherwise wrong line gets deleted when several lines are deleted
        row_count = row_count + 1
    return segments


def get_sentence_ids(review_name, number_of_sentences):
    array_of_ids = np.empty(number_of_sentences, dtype=object)
    for i in range(number_of_sentences):
        array_of_ids[i] = review_name + "_" + str(i)
    return array_of_ids


def run_algorithm_on_one_review(
        path, current_filename, current_anno, current_review):
    annotation = Annotation.read_from_file(path)
    all_lengths = get_all_lengths()
    length = all_lengths[current_review]
    annotation.fill_gap(length)
    raters = annotation.raters
    current_decisions = get_decisions(length, raters, annotation)
    majority = get_majorities(current_decisions, length)
    text = sample_data.loc[
        sample_data['review_id'] == current_review].review.values[0]
    sentences = get_sentences(text)
    sentence_ids = get_sentence_ids(current_review, len(sentences[:, 0]))
    segments = get_segments_in_sentences(majority, sentences, sentence_ids)
    segments = remove_too_short_segments(segments)
    # folder_name = "../data/text_sentences_majority/"
    # if not os.path.isdir(folder_name):
    #     os.mkdir(folder_name)
    # output_filename = folder_name + current_review + ".csv"
    one_file_for_all = "../data/text_sentences_majority.csv"
    # np.savetxt(output_filename,
    #   segments, fmt="%1s", delimiter='\t', newline="\n")
    opened_file = open(one_file_for_all, 'a')
    np.savetxt(opened_file, segments, fmt="%1s", delimiter='\t ', newline="\n")
    opened_file.close()


def iterate_over_all_reviews():
    for review in sample_data["review_id"]:
        current_filename = "/" + review + "_anno.csv"
        path = folder + current_filename
        current_anno = pd.read_csv(path, sep=',')
        run_algorithm_on_one_review(
            path, current_filename, current_anno, review)


if __name__ == "__main__":
    iterate_over_all_reviews()
