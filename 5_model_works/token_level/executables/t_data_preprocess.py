import pandas as pd
import csv
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil


nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

model_addeds = ['classify', 'recog', 'stance']

train_set = 0.7
dev_set = 0.1
test_set = 0.2


def get_splitting_info(data):
    """
    Get cross-domain splitting train/dev/test. Meant for AURC data
    """
    split_info = pd.read_csv(data, sep='\t')
    hash_to_split = {}
    for _, row in split_info.iterrows():
        hash_to_split[row['sentence_hash']] = row['Cross-Domain']
    return hash_to_split


def split_sentence(length, sentence, spans, tags):
    """
    Split a single sentence. Meant for AURC data
    """
    seg_tag = []  # [[seg1, tag1], [seg2, tag2], ...]
    splitting = list(zip(spans.split(';'),
                         tags.split(';')))[:-1]  # [((start1, extend1), tag1)]

    if len(splitting) > 1:
        splitting.sort(key=lambda x: eval(x[0])[0])  # Sort w.r.t start1
        for i in range(len(splitting) - 1):
            split1 = splitting[i]
            start1, extend1 = eval(split1[0])
            split2 = splitting[i + 1]
            start2, extend2 = eval(split2[0])
            # start1 + extend1 may not exceed start2
            if int(start1) + int(extend1) > int(start2):
                print('Error! Overlapping occurs', split1, split2)
            # Gap between seg1 and seg2
            elif int(start1) + int(extend1) < int(start2):
                splitting.append((str(
                    (int(start1) + int(extend1),
                     int(start2) - int(extend1) - int(start1))), 'non'))
        splitting.sort(key=lambda x: eval(x[0])[0])

    # Check gap before first seg and gap after last seg
    fst_elem, last_elem = eval(splitting[0][0]), eval(splitting[-1][0])
    fst, fst_extend = int(fst_elem[0]), int(fst_elem[1])
    last, last_extend = int(last_elem[0]), int(last_elem[1])

    # Head & Tail padding
    if fst != 0:
        splitting.insert(0, (str((0, fst)), 'non'))
    if last + last_extend < length:
        splitting.append((str(
            (last + last_extend, length - last - last_extend)), 'non'))

    for elem in splitting:
        span, tag = eval(elem[0]), elem[1]
        start, extend = int(span[0]), int(span[1])
        seg_tag.append([sentence[start:start + extend], tag])

    return seg_tag


def split_aurc(file):
    """
    Split sentences in AURC into [segment, label]. Meant for AURC
    """
    data = pd.read_csv(file, sep='\t')
    sen_splitting = {
    }  # mapping sentence_hash to list of [topic, [segment, tag]]
    for _, row in data.iterrows():
        topic, sen_hash, length, sen, seg = row[[
            'topic', 'sentence_hash', 'length_in_text', 'sentence',
            'merged_segments'
        ]]
        arg, spans, tags = eval(seg)
        if arg == 'false':  # Argument
            sen_splitting[sen_hash] = [
                topic, split_sentence(length, sen, spans, tags)
            ]
        else:  # Non-Argument
            sen_splitting[sen_hash] = [topic, [[sen, 'non']]]

    return sen_splitting


def split_review(file, topic):
    """
    Split sentences in review into [segment, tag]
    """

    data = pd.read_csv(file, sep='\t', header=None)
    sen_splitting = {
    }  # mapping sentence_hash to list of [topic, [segment, tag]]
    for _, row in data.iterrows():
        seg_tag = []
        sen_hash, sen, seg = row[[0, 1, 2]]
        sen = sen.strip()
        spans, tags = eval(seg)
        splitting = list(zip(spans.split(';'), tags.split(';')))[:-1]
        for elem in splitting:
            span, tag = eval(elem[0]), elem[1]
            if tag == 'POS':
                tag = 'pro'
            elif tag == 'NEG':
                tag = 'con'
            elif tag == 'NA':
                tag = 'non'
            else:
                print(sen_hash, sen, seg)
            start, extend = int(span[0]), int(span[1])
            seg_tag.append([sen[start:start + extend], tag])
        sen_splitting[sen_hash] = [topic, seg_tag]

    return sen_splitting


def create_sen_per_line_data(sentences, out_dir):
    """
    Create data file consisting of sentence id, sentence, list of labels per line. Meant for AMSR.
    :param sentences: Dictionary mapping sentence id to segments in sentence.
    :param out_dir: Path of outpus file.
    """

    lines = []
    for k, sens in sentences.items():

        topic = sens[0]
        sentence_text = ''
        labels = []
        for seg, tag in sens[1]:
            seg = seg.strip()
            if seg:
                sentence_text = ' '.join([sentence_text, seg])
                tokens = tokenizer(seg)
                labels.extend([tag] * len(tokens))
        lines.append({
            "sentence_id": k,
            'sentence': sentence_text.strip(),
            'labels': labels,
            'topic': topic
        })

    df = pd.DataFrame(lines,
                      columns=['sentence_id', 'sentence', 'labels', 'topic'])
    df.to_csv(out_dir, index=False, sep='\t')


def create_token_per_line_data(file, out_dir):
    """
    Create file containing sentence_id, token, label per line. Meant for AMSR
    Sentence_id, EOF, EOF at the end of each sentence.
    :param file: Sentence per line file.
    :param out_dir: Path of output file.
    """
    df = pd.read_csv(file, sep='\t')
    lines = []
    for _, row in df.iterrows():
        sen_id, sen, labels = row[['sentence_id', 'sentence', 'labels']]
        tokens = tokenizer(sen)
        for i in range(len(tokens)):
            if len(tokens) != len(eval(labels)):
                print('error', sen_id, sen, labels)
            else:
                token, label = tokens[i], eval(labels)[i]
                if token.text.strip():
                    lines.append({
                        "sentence_id": sen_id,
                        'token': token,
                        'label': label
                    })
        lines.append({"sentence_id": sen_id, 'token': 'EOF', 'label': 'EOF'})
    df = pd.DataFrame(lines, columns=['sentence_id', 'token', 'label'])
    df.to_csv(out_dir, index=False, sep='\t')


def create_evaluate_data(file, task, out_dir, if_id=False):
    """
    Create data for evaluating models. Meant for AURC/AMSR models
    """
    df = pd.read_csv(file, sep='\t')
    lines = []
    for _, row in df.iterrows():
        sen_id, sen, labels, topic = row[[
            'sentence_id', 'sentence', 'labels', 'topic'
        ]]
        labels_new = []
        if task not in models:
            print('Error: no such task available')
        if task == models[0]:
            labels_new = labels
        if task == models[1]:
            for label in eval(labels):
                new = label if label == 'non' else 'arg'
                labels_new.append(new)
        if task == models[2]:
            if len(set(eval(labels))) == 1 and 'non' in eval(labels):
                continue
            labels_new = labels
        if labels_new and if_id:
            lines.append([sen_id, sen, labels_new, topic])
        elif labels_new and not if_id:
            lines.append([sen, labels_new, topic])

    with open(out_dir, 'w') as f:
        if if_id:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['sentence_id', 'sentence', 'labels', 'topic'])
        else:
            writer = csv.writer(f, delimiter=' ')
        for line in lines:
            writer.writerow(line)


def split_train_dev_test_topic(file_data, file_split, task, paths):
    """
    Split data into train, dev, test set according to topic. Meant for AURC
    :param file_data: Data file.
    :param file_split: File containing splitting information.
    :param task: Classify/recog/stance.
    :param paths: Paths of output files.
    """
    hash_to_split = get_splitting_info(file_split)
    train, dev, test = [], [], []
    df = pd.read_csv(file_data, sep='\t')
    for _, row in df.iterrows():
        sen_id, sen, labels, topic = row[[
            'sentence_id', 'sentence', 'labels', 'topic'
        ]]
        labels_new = []
        if task not in models:
            print('Error: no such task')
        if task == models[0]:
            labels_new = labels
        if task == models[1]:
            for label in eval(labels):
                new = label if label == 'non' else 'arg'
                labels_new.append(new)
        if task == models[2]:
            if len(set(eval(labels))) == 1 and 'non' in eval(labels):
                continue
            labels_new = labels
        line = [sen, labels_new, topic]

        domain = hash_to_split[sen_id]

        if domain == 'Train':
            train.append(line)
        elif domain == 'Dev':
            dev.append(line)
        elif domain == 'Test':
            test.append(line)
    total = [train, dev, test]
    for i in range(3):
        rows, path = total[i], paths[i]
        with open(path, 'w', encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=' ')
            for row in rows:
                writer.writerow(row)


def split_train_dev_test_ratio(file_token, file_sen, task, folder, paths):
    """
    Split data to train/dev/test set according to ratio.
    """
    sen_label = pd.read_csv(file_sen, sep='\t', header=None)
    sen_label.columns = ['sen_id', 'sen', 'label']
    if task == models[2]:
        # sen_label = sen_label.drop(sen_label[sen_label['label'] == 'NA'].index)
        sen_label = sen_label[sen_label['label'].str.strip().isin(
            ['POS', 'NEG'])]

    print(len(sen_label))
    train_pre, dev_test = train_test_split(sen_label,
                                           test_size=test_set + dev_set,
                                           stratify=sen_label['label'])
    dev_pre, test_pre = train_test_split(dev_test,
                                         test_size=test_set /
                                         (test_set + dev_set),
                                         stratify=dev_test['label'])
    train_id, dev_id, test_id = train_pre['sen_id'], dev_pre[
        'sen_id'], test_pre['sen_id']

    data = pd.read_csv(file_token, sep='\t')

    train, dev, test = [], [], []
    for _, row in data.iterrows():
        sen_id, sen, labels, topic = row[[
            'sentence_id', 'sentence', 'labels', 'topic'
        ]]
        labels_new = []
        if task == models[0]:
            labels_new = labels
        if task == models[1]:
            for label in eval(labels):
                new = label if label == 'non' else 'arg'
                labels_new.append(new)
        if task == models[2]:
            if len(set(eval(labels))) == 1 and 'non' in eval(labels):
                continue
            labels_new = labels

        line = [sen, labels_new, topic]
        if sen_id in train_id.tolist():
            train.append(line)
        elif sen_id in dev_id.tolist():
            dev.append(line)
        elif sen_id in test_id.tolist():
            test.append(line)

    total = [train, dev, test]
    for i in range(3):
        rows, path = total[i], paths[i]
        path = folder + path
        with open(path, 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            for row in rows:
                writer.writerow(row)


def check_max_length_review(file, length):
    """
    Check the number of sentences longer than the given length.
    """
    df = pd.read_csv(file, sep='\t')
    max_length = 0
    count = 0
    for _, row in df.iterrows():
        sen_id, labels = row['sentence_id'], eval(row['labels'])
        current_length = len(labels)
        if current_length > length:
            count += 1
            print(sen_id, current_length)
        elif current_length > max_length:
            max_length = current_length
    print(max_length, count)


def split_rev_training(parent_folder, paths):
    """
    Split review training set into 0.2/0.4/0.6/0.8/1.
    """
    train_path = paths[0]
    dev_path = paths[1]
    test_path = paths[2]

    df = pd.read_csv(os.path.join(parent_folder, train_path), header=None, sep=' ')

    for i in np.arange(0.2, 1, 0.2):

        folder_name = "train" + str(int(i*100))
        new_folder_path = os.path.join(parent_folder, folder_name)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        new_train_path = os.path.join(new_folder_path, train_path)
        current = df.sample(frac=i).reset_index(drop=True)
        current.to_csv(new_train_path,
                       sep=' ',
                       index=False,
                       header=False)

        new_dev_path = os.path.join(new_folder_path, dev_path)
        new_test_path = os.path.join(new_folder_path, test_path)
        shutil.copy(os.path.join(parent_folder, dev_path), new_dev_path)
        shutil.copy(os.path.join(parent_folder, test_path), new_test_path)


if __name__ == "__main__":

    raw_data_folder = 'data/raw_data/'
    preprocessed_folder = 'data/preprocessed/'
    aurc_folder = 'data/aurc/'
    amsr_folder = 'data/amsr/'

    if not os.path.exists(raw_data_folder):
        os.makedirs(raw_data_folder)
    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)
    if not os.path.exists(aurc_folder):
        os.makedirs(aurc_folder)
    if not os.path.exists(amsr_folder):
        os.makedirs(amsr_folder)

    # copy "sentence_just_one_position.csv" and "text_sentence_majority.csv"
    # from /4_post_processing/data and save to the raw_data_folder.
    # download "AURC_DATA.tsv" and "AURC_DOMAIN_SPLITS.tsv from https://github.com/trtm/AURC/tree/master/data
    # and save to raw_data_folder.

    tasks = ["classify", "recog", "stance"]

    for task in tasks:

        #### Prepare AURC Data ####
        aurc_splitting = split_aurc(raw_data_folder + 'AURC_DATA.tsv')
        create_sen_per_line_data(aurc_splitting,
                                 preprocessed_folder + 'aurc_sen_per_line.tsv')

        # create the file names for AURC data
        file_names_aurc = [aurc_folder + task + '_sen_' + name + '.txt'
                           for name in ['train', 'dev', 'test']]
        split_train_dev_test_topic(preprocessed_folder + 'aurc_sen_per_line.tsv',
                                   raw_data_folder + 'AURC_DOMAIN_SPLITS.tsv',
                                   task, file_names_aurc)

        #### Prepare AMSR data ####
        review_splitting = split_review(raw_data_folder + 'text_sentences_majority.csv',
                                        'paper quality')
        create_sen_per_line_data(review_splitting,
                                 preprocessed_folder + 'review_sen_per_line.tsv')

        # Prepare AMSR for aurc model evaluation.
        task_eval_path = os.path.join(aurc_folder, "amsr_" + task + "_eval")
        if not os.path.exists(task_eval_path):
            os.makedirs(task_eval_path)
        create_evaluate_data(preprocessed_folder + 'review_sen_per_line.tsv',
                             task, aurc_folder + task_eval_path + task + '_sen_test.txt')

        # check to see how many sentences are longer than 85 tokens
        check_max_length_review(preprocessed_folder + 'review_sen_per_line.tsv', 85)

        # creates the file names for AMSR data
        file_names_rev = [task + '_sen_' + name + '.txt'
                          for name in ['train', 'dev', 'test']]

        # Prepare the datasets for AMSR
        split_train_dev_test_ratio(preprocessed_folder + 'review_sen_per_line.tsv',
                                   raw_data_folder + 'sentences_just_one_position.csv',
                                   task, amsr_folder, file_names_rev)

        # Prepare data folders for different training sets
        split_rev_training(amsr_folder, file_names_rev)
