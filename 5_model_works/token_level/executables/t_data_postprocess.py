import pandas as pd
import re
import csv
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

seeds = [1, 5, 10, 16, 24, 42, 57, 68, 79, 93]
tasks = [
    'classify_notopic', 'classify_topic', 'recog_notopic', 'recog_topic',
    'stance_notopic', 'stance_topic'
]
scores = ['mi_f1', 'ma_f1', 'w_f1', 'accuracy']


# ---------------Methods regarding model evaluation------------------------------
def merge_review_scores(input_dir, arch, task, model_name, filename,
                        output_dir, mode):
    """
    Merge all scores of one architecture training on different sizes of reviews.
    """
    rows = []
    for size in range(20, 120, 20):
        for seed in seeds:
            file = ''.join([
                input_dir, task, '/', arch, '/rev',
                str(size), model_name, task,
                str(seed), filename
            ])
            rows.append([arch, str(size)])
            with open(file, 'r') as f:
                lines = f.readlines()
                line = lines[-1].strip().split('\t')
                rows[-1].extend(line)
    out_file = ''.join([output_dir, task, '.csv'])
    with open(out_file, mode) as d:
        writer = csv.writer(d, delimiter=',')
        for row in rows:
            writer.writerow(row)


def merge_aurc_scores(input_dir, file_name, score_name, output_dir, mode):
    """
    Merge scores of aurc models.
    """
    rows = []
    for task in tasks:
        for seed in seeds:
            file = ''.join(
                [input_dir, task, '/aurc/aurc_', task,
                 str(seed), file_name])
            current = [task, score_name]
            with open(file, 'r') as f:
                lines = f.readlines()
                line = lines[-1].strip().split('\t')
                current.extend(line)
                rows.append(current)
    with open(output_dir, mode) as d:
        writer = csv.writer(d, delimiter=',')
        for row in rows:
            writer.writerow(row)


def average_scores(file):
    """
    Calculate average scores per model and training data.
    """
    df = pd.read_csv(
        file,
        header=None,
        names=['model', 'train_size', 'mi_f1', 'ma_f1', 'w_f1', 'accuracy'])
    print(df.groupby(['model', 'train_size']).mean().round(3))


def draw_graph(input_dir, task, score):
    """
    Draw training set size - score curve.
    """
    file = ''.join([input_dir, task, '.csv'])
    data = pd.read_csv(file,
                       header=None,
                       names=[
                           'architecture', 'train_size', scores[0], scores[1],
                           scores[2], scores[3]
                       ])
    sns.lineplot(x='train_size',
                 y=score,
                 hue='architecture',
                 markers=True,
                 data=data)
    plt.title(' '.join([task, score]))
    plt.xlabel("training set (%)")
    plt.ylim(0, 1)
    plt.savefig(''.join([input_dir, task, '_', score]))


# ----------------Methods regarding predictions----------------------------
def read_labeled_reviews(file):
    """
    Read labeled reviews.
    """
    reviews = pd.read_csv(file, sep='\t', header=None)
    return reviews.iloc[:, 0].values.tolist()


def construct_reviews(file_labeled, file_unlabeled, out_dir):
    """
    Construct reviews of each conference.
    """
    labeled_reviews = read_labeled_reviews(file_labeled)
    data = pd.read_csv(file_unlabeled)
    current_id = ''

    # Get first unlabeled sentence id
    for i in range(len(data)):
        first_id = data.iloc[1, i]
        if first_id in labeled_reviews:
            continue
        else:
            first_id_match = re.match(r'([a-z0-9]+)_(.*)', first_id)
            current_id = first_id_match.group(1)
            break
    print(current_id)
    # Initialize sentences
    sentences = []
    review_id_set = set()

    # Iterate through all review sentences
    count = 0
    for _, row in data.iterrows():
        sen_id, sen = row[['id', 'sentence']]
        if sen_id not in labeled_reviews:
            if count % 10000 == 0:
                print("parse to sentence", count, sen_id, current_id)
            # Matching
            match_id = re.match(r'([a-z0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)',
                                sen_id)
            conference_id, paper_id, review_id = match_id.group(
                1), match_id.group(2), match_id.group(3)
            # print(conference_id, paper_id, review_id)
            review_id_set.add('_'.join([conference_id, paper_id, review_id]))
            if current_id == conference_id:
                sentences.append(dict(sentence_id=sen_id, sentence=sen))
            else:
                print('end of conference', current_id)
                review = pd.DataFrame(sentences,
                                      columns=['sentence_id', 'sentence'])
                review.to_csv(''.join([out_dir, current_id, '.csv']),
                              index=False)
                current_id = conference_id
                sentences = [dict(sentence_id=sen_id, sentence=sen)]
            count += 1
    if sentences:
        print('saving last conference:', current_id)
        review = pd.DataFrame(sentences, columns=['sentence_id', 'sentence'])
        review.to_csv(''.join([out_dir, current_id, '.csv']), index=False)

    # Save review ids
    review_id_dict = pd.DataFrame([{'review_id': list(review_id_set)}])
    review_id_dict.to_csv(''.join([out_dir, 'review_id.csv']), index=False)


def split_conference(file, out_dir, size, last):
    """
    Split reviews in one conference in case file size too large for the model.
    """

    data = pd.read_csv(file)
    first_id = data.iloc[1, 0]
    first_id_match = re.match(r'([a-z0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)',
                              first_id)
    current_id = int(first_id_match.group(2))
    print(current_id)

    sentences = []
    count = 1

    for _, row in data.iterrows():
        sen_id, sen = row[['sentence_id', 'sentence']]
        match_id = re.match(r'([a-z0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)', sen_id)
        conference_id, paper_id, review_id = match_id.group(1), int(
            match_id.group(2)), match_id.group(3)
        if current_id == paper_id or paper_id % size != 1 or paper_id > last + 1:
            sentences.append(dict(sentence_id=sen_id, sentence=sen))
        elif current_id != paper_id and paper_id % size == 1 and paper_id <= last + 1:
            print('end of paper', current_id)
            split = pd.DataFrame(sentences,
                                 columns=['sentence_id', 'sentence'])
            split.to_csv(''.join([out_dir, str(count), '.csv']), index=False)
            current_id = paper_id
            sentences = [dict(sentence_id=sen_id, sentence=sen)]
            count += 1
    if sentences:
        split = pd.DataFrame(sentences, columns=['sentence_id', 'sentence'])
        split.to_csv(''.join([out_dir, str(count), '.csv']), index=False)


def merge_confidence(input_dir, file_names, output_dir):
    """
    Merge confidence of all sentences together.
    """
    df = pd.DataFrame()
    for file in file_names:
        print("merging", file)
        df = df.append(pd.read_csv(input_dir + file + '_confidence.csv'),
                       ignore_index=True)
    print('merging file...')
    df.to_csv(output_dir, index=False)


def merge_hidden(input_dir, file_names, output_dir):
    """
    Merge all hidden states of sentences togethter.
    """
    df = {}
    for file in file_names:
        print('merging', file)
        current = np.load(input_dir + file + '_rep.npy', allow_pickle=True)
        for elem in current:
            df[elem['sentence_id']] = elem['representation']
    print('saving hidden')
    np.save(output_dir, df)


def calculate_sen_score(file, out_dir):
    """
    Calculate average score and argument percentage.
    """
    df = pd.read_csv(file)
    t_score = df['confidence']
    avg_score = []
    arg_percentage = []
    for confidences in tqdm(t_score, desc="line"):
        confidences = eval(confidences)
        avg_score.append(sum(confidences) / len(confidences))
        args = [arg for arg in confidences if arg > 0.5]
        arg_percentage.append(len(args) / len(confidences))
    df['avg_score'] = avg_score
    df['arg_percentage'] = arg_percentage
    df.to_csv(out_dir, index=False)


def add_rating_and_decision(t_confidence, s_confidence, out_dir):
    mapping = {}
    df = pd.read_csv(s_confidence)
    df = df.groupby(
        ['conference', 'paper_id', 'review_id', 'rating', 'decision'],
        sort=False)
    group_names = df.groups.keys()
    for name in group_names:
        review = '_'.join([name[0], str(name[1]), str(name[2])])
        mapping[review] = [name[3], name[4]]
    print(list(mapping.items())[:10])
    dd = pd.read_csv(t_confidence)
    ratings = []
    decisions = []
    for _, row in dd.iterrows():
        review = '_'.join(
            [row['conference'],
             str(row['paper_id']),
             str(row['review_id'])])
        ratings.append(mapping[review][0])
        decisions.append(mapping[review][1])
    dd['rating'] = ratings
    dd['decision'] = decisions
    dd.to_csv(out_dir, index=False)


if __name__ == "__main__":
    # Example Implementation

    # Methods regarding model evaluation
    path_model = '../models/'

    for task in tasks:
        merge_review_scores(path_model, 'bert-large-cased', task, '_bl_',
                            '/eval_scores.txt', '../results/', 'a')
    merge_aurc_scores(path_model, '/reviews_test/eval_scores.txt', 'review',
                      '../results/', 'a')

    average_scores('../results/recog_topic.csv')
    draw_graph('../results/', tasks[0], scores[0])

    # Methods regarding predictions
    construct_reviews('../Data/raw_data/reviews_segmentLevel.csv',
                      '../Data/raw_data/all_review_confidence.csv',
                      '../Data/raw_reviews/')
    split_conference('../Data/raw_reviews/iclr20.csv',
                     '../Data/raw_reviews/iclr20_', 100, 1200)
    iclr19_names = []
    iclr20_names = []
    for num in range(1, 15):
        iclr19_names.append('iclr19_' + str(num))
    for num in range(1, 14):
        iclr20_names.append('iclr20_' + str(num))
    all_names = ['graph20', 'midl19', 'midl20', 'neuroai19']
    all_names.extend(iclr19_names)
    all_names.extend(iclr20_names)

    merge_confidence('../predictions/', all_names,
                     '../predictions/all_confidence.csv')
    merge_hidden('../predictions/', all_names, '../predictions/all_rep.npy')

    path = '../../pbds-group-5-project/6_arg_extraction/'
    calculate_sen_score(path + 't_confidence.csv',
                        path + 't_confidence_new.csv')
    add_rating_and_decision(path + 't_prediction/t_confidence.csv',
                            path + 'prediction/confidence.csv',
                            path + 't_prediction/t_confidence_new.csv')
