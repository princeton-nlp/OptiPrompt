import json
import argparse
import os
import random
import sys
import logging
from tqdm import tqdm
import torch

from models import Prober
from utils import load_vocab, load_data, batchify, save_model, evaluate, get_relation_meta

from transformers import AdamW, get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def init_template(args, model):
    relation = get_relation_meta(args)
    return relation['template']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='the huggingface model name')
    parser.add_argument('--model_dir', type=str, default=None, help='the model directory (if not using --model_name)')
    parser.add_argument('--output_dir', type=str, default='output', help='the output directory to store trained model and prediction results')
    parser.add_argument('--common_vocab_filename', type=str, default='data/common_vocab_cased.txt', help='common vocabulary of models (used to filter triples)')
    parser.add_argument('--relation_profile', type=str, default='data/relations.jsonl', help='meta infomation of 41 relations, containing the pre-defined templates')

    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=8)

    parser.add_argument('--seed', type=int, default=6)

    parser.add_argument('--relation', type=str, required=True, help='which relation is considered in this run')
    parser.add_argument('--random_init', type=str, default='none', choices=['none', 'embedding', 'all'], help='none: use pre-trained model; embedding: random initialize the embedding layer of the model; all: random initialize the whole model')

    parser.add_argument('--output_predictions', action='store_true', help='whether to output top-k predictions')
    parser.add_argument('--k', type=int, default=5, help='how many predictions will be outputted')

    args = parser.parse_args()

    logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))

    logger.info(args)
    n_gpu = torch.cuda.device_count()
    logger.info('# GPUs: %d'%n_gpu)
    if n_gpu == 0:
        logger.warning('No GPU found! exit!')

    logger.info('Model: %s'%args.model_name)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)

    model = Prober(args, random_init=args.random_init)

    if args.common_vocab_filename is not None:
        vocab_subset = load_vocab(args.common_vocab_filename)
        logger.info('Common vocab: %s, size: %d'%(args.common_vocab_filename, len(vocab_subset)))
        filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
    else:
        filter_indices = None
        index_list = None

    if n_gpu > 1:
        model.mlm_model = torch.nn.DataParallel(model.mlm_model)
    
    template = init_template(args, model)
    logger.info('Template: %s'%template)

    eval_samples = load_data(args.test_data, template, vocab_subset=vocab_subset, mask_token=model.MASK)
    eval_samples_batches, eval_sentences_batches = batchify(eval_samples, args.eval_batch_size * n_gpu)
    evaluate(model, eval_samples_batches, eval_sentences_batches, filter_indices, index_list, output_topk=args.output_dir if args.output_predictions else None)
