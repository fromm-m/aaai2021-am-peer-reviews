from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from collections import defaultdict

train_set = 0.7
dev_set = 0.1
test_set = 0.2


def split_train_dev_test(file, out_dir):
    """
    Split data to train/dev/test set according to ratio.
    """

    data = np.load(file, allow_pickle=True)
    print("data read")
    paper_ids = []
    decisions = []
    for paper in data:
        paper_ids.append(paper['paper_id'])
        decisions.append(paper['decision'])
    df = pd.DataFrame({'paper_id': paper_ids, 'decision': decisions})
    train_pre, dev_test = train_test_split(df,
                                           test_size=test_set + dev_set,
                                           stratify=df['decision'])
    dev_pre, test_pre = train_test_split(dev_test,
                                         test_size=test_set / (test_set + dev_set),
                                         stratify=dev_test['decision'])
    train_id, dev_id, test_id = train_pre['paper_id'], dev_pre['paper_id'], test_pre['paper_id']

    print("data splited")
    train = defaultdict(list)
    dev = defaultdict(list)
    test = defaultdict(list)
    for paper in data:
        if paper['decision'] == 0:
            review_decisions = ['rejected'] * len(paper['review_ids'])
        else:
            review_decisions = ['accepted'] * len(paper['review_ids'])
        if paper['paper_id'] in train_id.tolist():
            train['review_text'].extend(paper['review_text'])
            train['decision'].extend(review_decisions)
        elif paper['paper_id'] in dev_id.tolist():
            dev['review_text'].extend(paper['review_text'])
            dev['decision'].extend(review_decisions)
        else:
            test['review_text'].extend(paper['review_text'])
            test['decision'].extend(review_decisions)

    train = pd.DataFrame(train)
    dev = pd.DataFrame(dev)
    test = pd.DataFrame(test)
    print("sets created")
    train.to_csv(out_dir + 'train.tsv', sep='\t', index=False)
    dev.to_csv(out_dir + 'dev.tsv', sep='\t', index=False)
    test.to_csv(out_dir + 'test.tsv', sep='\t', index=False)


if __name__ == "__main__":
    path_source = '../6_arg_extraction/extracted_arg/s_100.npy'
    path_target = 'data/s_100/'

    if not os.path.exists(path_target):
        os.makedirs(path_target)

    split_train_dev_test(path_source, path_target)
    data1 = pd.read_csv(path_target + 'train.tsv', sep='\t')
    print(path_target + 'train.tsv')
    print(len(data1))
    data2 = pd.read_csv(path_target + 'dev.tsv', sep='\t')
    print(len(data2))
    data3 = pd.read_csv(path_target + 'test.tsv', sep='\t')
    print(len(data3))