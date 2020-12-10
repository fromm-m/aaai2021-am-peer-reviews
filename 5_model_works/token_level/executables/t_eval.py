import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
import csv
import logging
import datetime

# Logging Setup
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
filename = "".join(["log_", timestamp, ".log"])
logging.basicConfig(
    filename=filename,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

seeds = [1, 5, 10, 16, 24, 42, 57, 68, 79, 93]
tasks = [
    'classify_notopic', 'classify_topic', 'recog_notopic', 'recog_topic',
    'stance_notopic', 'stance_topic'
]
scores = ['mi_f1', 'ma_f1', 'w_f1', 'accuracy']


def predict_test_data(model, device, label_list, eval_dataloader, task):
    """
    Predicts test data for a passed model and test data loader.

    :param model:
        The trained model to use for prediction.
    :param device:
        The device to test on.
    :param label_list:
        The list of labels.
    :param eval_dataloader:
        The data loader containing the test data rows.

    :return:
        True labels of test data, Predictions of the model on the test data.
    """

    model.eval()
    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(
            eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids,
                            segment_ids,
                            input_mask,
                            valid_ids=valid_ids,
                            attention_mask_label=l_mask)
            logits = outputs[0]

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_ids = input_ids.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    break
                else:
                    if logits[i][j] not in label_map:
                        print(logits[i][j], label_ids[i][j], input_ids[i][j])
                    if label_map[logits[i][j]] == '[SEP]':
                        print(input_ids[i][j], label_ids[i][j], logits[i][j])
                    if task != 'stance' or \
                            (task == 'stance' and label_map[label_ids[i][j]] != 'non'):
                        y_true.append(label_map[label_ids[i][j]])
                        y_pred.append(label_map[logits[i][j]])

    logger.info('length of y_true: {}'.format(len(y_true)))
    logger.info('length of y_pred: {}'.format(len(y_pred)))

    return y_true, y_pred


def calculate_accuracy(true_labels, predictions):
    logger.info("Calculating accuracy...")
    return accuracy_score(true_labels, predictions, normalize=True)


def calculate_f1(true_labels, predictions):
    logger.info("Calculating f1...")
    f1 = [
        f1_score(true_labels, predictions, average='micro'),
        f1_score(true_labels, predictions, average='macro'),
        f1_score(true_labels, predictions, average='weighted')
    ]
    return f1


def write_scores(f1, accuracy, output_eval_scores):
    logger.info("Writing scores...")
    lines = [['micro', 'macro', 'weighted', 'accuracy']]
    f1.append(accuracy)
    lines.append(f1)
    with open(output_eval_scores, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for line in lines:
            writer.writerow(line)


def write_report(true_labels, predictions, output_eval_report):
    report = classification_report(true_labels, predictions, digits=4)
    with open(output_eval_report, "w") as writer:
        logger.info("***** Eval results *****")
        logger.info("\n%s", report)
        writer.write(report)
