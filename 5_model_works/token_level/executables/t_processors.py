import os
import pandas as pd
import datetime
from spacy.lang.en import English
import logging

timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
filename = "".join(["log_", timestamp, ".log"])
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


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
                 label_id,
                 valid_ids=None,
                 label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """
    Loads a data file into a list of `InputBatch`s

    :param label_list:
        List of unique labels in the dataset.
    :param examples:
        List of InputExamples to transform
    :param tokenizer:
        Tokenizer to tokenize sentences + topic
    :param max_seq_length:
        The max token length

    :returns:
        Tokenized Input Features from passed data examples.
    """

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    nlp = English()
    tokenizer1 = nlp.Defaults.create_tokenizer(nlp)

    features = []
    for (ex_index, example) in enumerate(examples):
        text_list = tokenizer1(example.text_a)
        label_list = example.label
        if len(text_list) != len(label_list):
            continue
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(text_list):
            token = tokenizer.tokenize(word.text)
            tokens.extend(token)
            label_1 = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)

        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

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
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features


def convert_topic_examples_to_features(examples, label_list, max_seq_length,
                                       tokenizer):
    """
    Loads a data file into a list of `InputBatch`s

    :param label_list:
        List of unique labels in the dataset.
    :param examples:
        List of InputExamples, that includes topic information, to transform
    :param tokenizer:
        Tokenizer to tokenize sentences + topic
    :param max_seq_length:
        The max token length

    :returns:
        Tokenized Input Features from passed data examples.
    """

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    nlp = English()
    tokenizer1 = nlp.Defaults.create_tokenizer(nlp)

    features = []
    for (ex_index, example) in enumerate(examples):
        text_list = tokenizer1(example.text_a)
        topic_list = tokenizer1(example.text_b)
        label_list = example.label
        if len(text_list) != len(label_list):
            continue
        tokens = []
        labels = []
        valid = []
        label_mask = []

        # define the tokens and labels for the text
        for i, word in enumerate(text_list):
            token = tokenizer.tokenize(word.text)
            tokens.extend(token)
            label_1 = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)

        # create new topic related list
        topic_tokens = []
        topic_labels = []
        topic_valid = []
        topic_label_mask = []

        # append the SEP token used to separate sentence and topic info.
        topic_tokens.append("[SEP]")
        topic_labels.append("[SEP]")
        topic_valid.append(1)
        topic_label_mask.append(1)

        # include topic info here.
        for i, word in enumerate(topic_list):
            topic_token = tokenizer.tokenize(word.text)
            topic_tokens.extend(topic_token)
            label_1 = "non"
            for m in range(len(topic_token)):
                if m == 0:
                    topic_labels.append(label_1)
                    topic_valid.append(1)
                    topic_label_mask.append(1)
                else:
                    topic_valid.append(0)

        # check whether truncation is needed with the adjusted length.
        adj_seq_length = max_seq_length - len(topic_tokens)
        if len(tokens) >= adj_seq_length - 1:
            tokens = tokens[0:(adj_seq_length - 1)]
            labels = labels[0:(adj_seq_length - 1)]
            valid = valid[0:(adj_seq_length - 1)]
            label_mask = label_mask[0:(adj_seq_length - 1)]

        # concatenate the text info and topic info
        tokens.extend(topic_tokens)
        labels.extend(topic_labels)
        valid.extend(topic_valid)
        label_mask.extend(topic_label_mask)

        # define the input_ids and label_ids
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)

        # define the padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)

        # label_ids and label_mask might need further padding
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)

        # ensure that the length match the max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        # show first 5 examples when training
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
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))

    return features


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir, use_topic):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, use_topic):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        return pd.read_csv(input_file,
                           delimiter=' ',
                           names=["sentence", "labels", "topic"])

    @staticmethod
    def _create_examples(lines, set_type, use_topic):
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
            text_a = lines["sentence"][i]
            text_b = None if not use_topic else lines["topic"][i]
            label = eval(lines["labels"][i])
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             label=label))
        return examples


class RecogProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""
    def get_train_examples(self, data_dir, use_topic):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "recog_sen_train.txt")),
            "train", use_topic)

    def get_dev_examples(self, data_dir, use_topic):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "recog_sen_dev.txt")), "dev",
            use_topic)

    def get_test_examples(self, data_dir, use_topic):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "recog_sen_test.txt")),
            "test", use_topic)

    def get_labels(self):
        return ["non", "arg", "[CLS]", "[SEP]"]


class ClassifyProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""
    def get_train_examples(self, data_dir, use_topic):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "classify_sen_train.txt")),
            "train", use_topic)

    def get_dev_examples(self, data_dir, use_topic):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "classify_sen_dev.txt")),
            "dev", use_topic)

    def get_test_examples(self, data_dir, use_topic):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "classify_sen_test.txt")),
            "test", use_topic)

    def get_labels(self):
        return ["non", "pro", "con", "[CLS]", "[SEP]"]


class StanceProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""
    def get_train_examples(self, data_dir, use_topic):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "stance_sen_train.txt")),
            "train", use_topic)

    def get_dev_examples(self, data_dir, use_topic):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "stance_sen_dev.txt")),
            "dev", use_topic)

    def get_test_examples(self, data_dir, use_topic):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "stance_sen_test.txt")),
            "test", use_topic)

    def get_labels(self):
        return ["non", "pro", "con", "[CLS]", "[SEP]"]


# available processor classes (values) with a unique task name (key)
t_processors = {
    "classify": ClassifyProcessor,
    "recog": RecogProcessor,
    "stance": StanceProcessor
}
