from __future__ import absolute_import, division, print_function

import argparse
import json
import logging.config
import os
import random
import datetime
import numpy as np
import torch
from collections import defaultdict
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from p_eval import (predict_test_data, calculate_f1, calculate_accuracy,
                    write_scores, write_report)
from p_processors import (DataProcessor, convert_examples_to_features)

timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
filename = "".join(["log_", timestamp, ".log"])
logging.basicConfig(
    filename=os.path.join("logs/", filename),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


# python src/p_modelprocessing.py --data_dir=data/k_50/ --models=bert-base-cased --output_dir=../../paper_models/k_50 --do_train --do_eval --seed 1 --num_train_epochs 10

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, verbose, delta):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.report_decrease(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info('EarlyStopping counter: {} out of {}'.format(
                self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.report_decrease(val_loss)
            self.counter = 0

    def report_decrease(self, val_loss):
        """ reports when validation loss decrease."""
        if self.verbose:
            logger.info(
                'Validation loss decreased ({:.6f} --> {:.6f}).'.format(
                    self.val_loss_min, val_loss))
        self.val_loss_min = val_loss


def data_preparer(examples, label_list, tokenizer):
    """
    Prepare the data
    :param examples: Input Examples
    :param label_list: List containing the unique labels
    :param tokenizer: Which tokenizer to use for tokenization
    :return:
        Data
    """

    features = convert_examples_to_features(examples, label_list, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features],
                                   dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features],
                                 dtype=torch.long)

    feature_data = TensorDataset(all_input_ids, all_input_mask,
                                 all_segment_ids, all_label_ids)

    return feature_data


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .npy files (or other data files) for the task."
    )

    parser.add_argument(
        "--models",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
             "bert-base-multilingual-cased, bert-base-chinese, or a user-defined model directory."
    )

    # Other parameters
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=5,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=5,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=50.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--patience",
        default=3.0,
        type=float,
        help="How long to wait after last time validation loss improved.")
    parser.add_argument(
        "--delta",
        default=0.001,
        type=float,
        help="Minimum change in the monitored quantity to qualify as an improvement."
    )
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                                        and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    print("device:", device)
    print("n_gpus:", n_gpu)

    # Add meta information about the model and device to be used.
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))
    logger.info(
        "Data Directory: {}, Model Directory: {}, Useed: {}"
            .format(args.data_dir, args.output_dir, args.seed))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
                .format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # set the seed value to the provided one, default is 42
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    # Set up output model directory.
    if args.do_train and not args.output_dir:
        raise ValueError(
            "Please mention an empty output directory to save the fine-tuned model."
        )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Call processor based on the task

    processor = DataProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    # Prepare model
    if os.path.exists(args.models) and os.listdir(args.models):

        # TODO: check Loading fine-finetuned model

        logger.info(
            "Loading the fine-tuned model from the directory ({})".format(
                args.models))
        model = BertForSequenceClassification.from_pretrained(args.models)
        tokenizer = BertTokenizer.from_pretrained(
            args.models, do_lower_case=args.do_lower_case)
    else:
        logger.info("Loading {} model".format(args.models))
        model = BertForSequenceClassification.from_pretrained(args.models,
                                                              num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.models,
                                                  do_lower_case=args.do_lower_case,
                                                  num_labels=num_labels)
    model.to(device)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size /
            args.gradient_accumulation_steps) * args.num_train_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
            args.weight_decay
    }, {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                     num_warmup_steps=warmup_steps,
                                    num_training_steps=num_train_optimization_steps)
    early_stopping = EarlyStopping(patience=args.patience,
                                   verbose=True,
                                   delta=args.delta)

    # multi-gpu training
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    global_step = 0

    # Run Training if requested
    if args.do_train:
        logger.info("***** Setting Up Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        # prepare the training dataloader
        train_data = data_preparer(train_examples, label_list, tokenizer)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size)

        # prepare the Validation dataloader
        logger.info("***** Setting up Validation *****")
        val_examples = processor.get_dev_examples(args.data_dir)
        logger.info("  Num examples = %d", len(val_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        val_data = data_preparer(val_examples, label_list, tokenizer)
        # Run validation for full validation dataset
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data,
                                    sampler=val_sampler,
                                    batch_size=args.eval_batch_size)

        # Start Train Cycle
        model.train()
        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(
                    tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=label_ids)
                loss = outputs[0]
                if n_gpu > 1:
                    # mean() to average on multi-gpu.
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    # Update learning rate schedule
                    # noinspection PyArgumentList
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

            logger.info("Training loss after Epoch {} = {:06.3f}".format(
                ep + 1, tr_loss))

            # Validation Cycle
            model.eval()
            val_loss, val_accuracy = 0, 0
            for input_ids, input_mask, segment_ids, label_ids in \
                    tqdm(val_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=label_ids)
                    valid_loss = outputs[0]

            val_loss += valid_loss
            logger.info("Validation loss after Epoch {} = {:06.3f}".format(
                ep + 1, val_loss))

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                logger.info("Early Stopping at Epoch %d", ep + 1)
                break

        # Save a trained model and the associated configuration
        # Only save the model it-self
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        model_config = {
            "Model": args.models,
            "do_lower": args.do_lower_case,
            "num_labels": len(label_list) + 1,
            "label_map": label_map
        }
        json.dump(
            model_config,
            open(os.path.join(args.output_dir, "model_config.json"), "w"))

    # Run Evaluation if requested.
    model.to(device)
    if args.do_eval:
        logger.info("***** Setting Up Evaluation *****")
        eval_examples = processor.get_test_examples(args.data_dir)
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_data = data_preparer(eval_examples, label_list, tokenizer)

        # Prepare Evaluation Dataloader
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data,
                                     sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        y_true, y_pred = predict_test_data(model, device, eval_dataloader)
        name_report = 'eval_report.txt'
        name_scores = 'eval_scores.txt'
        if args.output_dir:
            path_report = os.path.join(args.output_dir, name_report)
            path_scores = os.path.join(args.output_dir, name_scores)
        else:
            # Will overwrite existing eval files, if any.
            path_report = os.path.join(args.models, name_report)
            path_scores = os.path.join(args.models, name_scores)

        f1, accuracy = calculate_f1(y_true, y_pred), calculate_accuracy(
            y_true, y_pred)
        logger.info("f1: {} accuracy: {}".format(f1, accuracy))
        write_scores(f1, accuracy, path_scores)
        write_report(y_true, y_pred, path_report)


if __name__ == "__main__":
    main()
