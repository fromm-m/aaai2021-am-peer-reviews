import os
import pandas as pd
import datetime
import numpy as np
import logging

timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
filename = "".join(["log_", timestamp, ".log"])
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: Use max_seq_length instead of hard coding 512, remember to add hyperparameter in p_modelprocessing.py

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, tokenizer):
    """
    Loads a data file into a list of `InputBatch`s

    :param label_list:
        List of unique labels in the dataset.
    :param examples:
        List of InputExamples to transform
    :param tokenizer:
        Tokenizer to tokenize sentences + topic

    :returns:
        Tokenized Input Features from passed data examples.
    """

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if isinstance(example.text_a, float):
            print(example.guid, example.text_a)
        tokens = tokenizer.tokenize(example.text_a)
        label = example.label
        token_chunks = []

        while len(tokens) > 510:
            token_chunks.append(tokens[:510])
            tokens = tokens[:510]

        if not token_chunks:
            token_chunks = [tokens]

        for chunk in token_chunks:
            chunk.insert(0, '[CLS]')
            chunk.append('[SEP]')
            input_ids = tokenizer.convert_tokens_to_ids(chunk)
            input_mask = [1] * len(chunk)
            segment_ids = [0] * len(chunk)

            while len(input_ids) < 512:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == 512
            assert len(input_mask) == 512
            assert len(segment_ids) == 512

            features.append(InputFeatures(input_ids=input_ids,
                                          input_mask=input_mask,
                                          segment_ids=segment_ids,
                                          label_id=label_map[label]))
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x)
                                                        for x in input_ids]))
                logger.info("input_mask: %s" %
                            " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" %
                            " ".join([str(x) for x in segment_ids]))

    return features


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'train.tsv')), 'train')

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'dev.tsv')), 'dev')

    def get_test_examples(self, data_dir):
        """Gets a collection of 'InputExamples' for the test set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    @staticmethod
    def get_labels():
        """Gets the list of labels for this data set."""
        return ['accepted', 'rejected']

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        return pd.read_csv(input_file, sep='\t')

    @staticmethod
    def _create_examples(lines, set_type):
        """
        Create examples for the training, dev and test sets.
        :param lines:
            list of sentences and labels
        :param set_type:
            train, dev or test.
        :return:
            list of InputExamples
        """
        examples = []
        for i in range(len(lines)):
            guid = "%s-%s" % (set_type, i)
            text_a = lines['review_text'][i]
            text_b = None
            label = lines['decision'][i]
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             label=label))
        return examples
