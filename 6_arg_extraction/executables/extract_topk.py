import pandas as pd
# from spacy.lang.en import English
import re
import numpy as np

# TODO: Remove reviews before calculating thresholds

# # tokenizer
# nlp = English()
# tokenizer = nlp.Defaults.create_tokenizer(nlp)

# ignored reviews at token level since len(tokens) != len(scores)
# assii = ['\x02', '\x0f', '\x1b']
# assii_sen = ['iclr19_828_3_29', 'iclr19_910_2_8', 'iclr20_79_1_27', 'iclr20_2005_1_21', 'iclr20_2005_1_22']
# long_sen = ['idl19_44_1_41', 'midl20_124_4_2', 'iclr20_223_3_5', 'iclr20_457_1_7',
#             'iclr20_1440_1_16', 'iclr20_1746_2_9', 'iclr20_1939_1_24']


def t_topk(file, k, out_dir):
    """
    Extract top k arguments at sentence level.
    :param file: Input csv file each sentence per line.
    :param k: For each paper, k percentage of tokens will be extracted.
    :param out_dir: Directory of extracted arguments in .npy.
    """

    df = pd.read_csv(file)
    grouped_paper = df.groupby(
        ['conference', 'paper_id', 'decision'], sort=False)

    # Initialize paper list
    extracted_papers = []
    # Group paper
    for name, group in grouped_paper:
        # Information of paper will be stored in dictionary
        paper = {'paper_id': str(name[0]) + '_' + str(name[1]), 'decision': name[2]}
        # Calculate threshold if taking top k
        scores_all = []
        confidence_all = group['confidence']
        for conf in confidence_all:
            scores_all.extend(eval(conf))
        scores_all.sort(reverse=True)
        threshold = scores_all[int(len(scores_all) * k) - 1]
        # Group review
        grouped_review = group.groupby(['review_id', 'rating'])
        review_ids = []
        review_ratings = []
        extracted_reviews = []
        for review_name, review in grouped_review:
            extracted_sentences = []
            sentences, confidence = review['sentence'], review['confidence']
            for sen, conf in zip(sentences, confidence):
                tokens = tokenizer(sen)
                conf = eval(conf)
                if len(tokens) == len(conf):  # Ignore sentence exceeding 512 or containing bytes
                    sentence = []
                    for i in range(len(tokens)):
                        if conf[i] >= threshold:
                            sentence.append(tokens[i].text)
                    if sentence:  # Ignore empty sentence
                        extracted_sentences.extend(sentence)
            if extracted_sentences:  # Ignore empty review
                extracted_reviews.append(' '.join(extracted_sentences))
                review_ids.append(review_name[0])
                review_ratings.append(review_name[1])

        paper['review_ids'] = review_ids
        paper['ratings'] = review_ratings
        paper['review_text'] = extracted_reviews
        extracted_papers.append(paper)
    np.save(out_dir, extracted_papers)


def get_labeled_review(file):
    """
    Gets labeled review_ids.
    :param file: File containing labeled reviews, e.g. sentences_just_one_position.csv.
    :return: Labeled review ids set.
    """

    labeled_reviews = pd.read_csv(file, sep='\t', header=None)
    review_id_set = set()
    for _, row in labeled_reviews.iterrows():
        sen_id, sen = row[0], row[1]
        match_id = re.match(r'([a-z0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)', sen_id)
        conference_id, paper_id, review_id = match_id.group(1), match_id.group(2), match_id.group(3)
        review_id_set.add('_'.join([conference_id, paper_id, review_id]))
    return review_id_set


def topk(labeled_file, file, k, out_dir):
    """
    Extracts top k arguments at sentence level.
    :param labeled_file: File containing labeled review ids.
    :param file: Input csv file each sentence per line.
    :param k: For each paper, k percentage of sentences will be extracted.
    :param out_dir: Directory of extracted arguments in .npy.
    """

    # Gets labeled review ids
    labeled_id = get_labeled_review(labeled_file)

    df = pd.read_csv(file, sep="\t")
    # Group paper
    grouped_paper = df.groupby(
        ['conference', 'paper_id', 'decision'], sort=False)
    extracted_papers = []
    for name, group in grouped_paper:
        paper = {'paper_id': str(name[0]) + '_' + str(name[1]), 'decision': name[2]}
        # Calculate threshold
        scores = group['confidence'].tolist()
        scores.sort(reverse=True)
        threshold = scores[int(len(scores) * k) - 1]
        # Group review
        grouped_review = group.groupby(['review_id', 'rating'])
        review_ids = []
        review_ratings = []
        extracted_reviews = []
        for review_name, review in grouped_review:
            if paper['paper_id'] + '_' + str(review_name[0]) not in labeled_id:
                extracted_sentences = review['sentence'][review['confidence'] >= threshold].tolist()
                if extracted_sentences:  # Ignore empty review
                    extracted_sentences = ' '.join(extracted_sentences)
                    extracted_reviews.append(extracted_sentences)
                    review_ids.append(review_name[0])
                    review_ratings.append(review_name[1])
        paper['review_ids'] = review_ids
        paper['ratings'] = review_ratings
        paper['review_text'] = extracted_reviews
        extracted_papers.append(paper)

    np.save(out_dir, extracted_papers)


if __name__ == "__main__":

    path_label = '../../4_post_processing/data/sentences_just_one_position.csv'

    # Example sentence level
    topk(path_label,
         's_reviews_with_predictions.tsv', 1.0,
         'extracted_arg/s_100.npy')

    # Example token level
    # t_topk(path + 't_prediction/t_confidence.csv', 0.2,
    #        path + 'extracted_arg/t_reviews_20.npy')




