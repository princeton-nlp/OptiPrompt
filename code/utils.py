import json
import os
from tqdm import tqdm
import sys
import logging

logger = logging.getLogger(__name__)

def load_vocab(vocab_filename):
    with open(vocab_filename, "r") as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def parse_template(template, subject_label, object_label='[MASK]'):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return [template]

def convert_tokens_to_string(tokens):
    out_string = " ".join(tokens).replace(" ##", "").strip()
    return out_string

def get_relation_meta(args):
    relations = load_file(args.relation_profile)
    for relation in relations:
        if relation['relation'] == args.relation:
            return relation
    raise ValueError('Relation info %s not found in file %s'%(args.relation, args.relation_profile))

def batchify(data, batch_size):
    list_samples_batches = []
    list_sentences_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    
    c = 0
    for sample in data:
        input_sentences = sample['input_sentences']
        current_samples_batch.append(sample)
        current_sentences_batches.append(input_sentences)
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            current_samples_batch = []
            current_sentences_batches = []
            c = 0

    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)

    return list_samples_batches, list_sentences_batches


def save_model(model, args):
    logger.info('Saving model...')
    model_to_save = model.mlm_model
    model_to_save.save_pretrained(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)

def output_result(result, eval_loss):
    logger.info('* Evaluation result *')
    cor = 0
    tot = 0
    macro = 0.0
    loss = 0.0
    for rel in result:
        cor_, tot_, avg_, loss_ = result[rel]
        cor += cor_
        tot += tot_
        macro += avg_
        loss_ /= tot_
        loss += loss_
        logger.info('%s\t%.5f\t%d\t%d\t%.5f' % (rel, avg_, cor_, tot_, loss_))
    macro = cor / tot if tot > 0 else 0.0
    micro = macro / len(result) if len(result) > 0 else 0.0
    logger.info('Macro avg: %.5f' % macro)
    logger.info('Micro avg: %.5f, Eval_loss: %.5f, Eval_loss (common vocab): %.5f' %(micro, eval_loss / tot, loss / len(result) if len(result) > 0 else 0.0))
    sys.stdout.flush()
    return micro, macro

def evaluate(model, samples_batches, sentences_batches, filter_indices=None, index_list=None, output_topk=None):
    vocab_to_common_vocab = None
    if index_list is not None:
        vocab_to_common_vocab = {}
        for cid, idx in enumerate(index_list):
            vocab_to_common_vocab[idx] = cid

    cor_all = 0
    tot_all = 0
    result = {}
    list_of_predictions = {}
    eval_loss = 0.0
    common_eval_loss = 0.0
    for i in tqdm(range(len(samples_batches))):
        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]

        log_probs, cor_b, tot_b, pred_b, topk_preds, loss, common_vocab_loss = model.run_batch(sentences_b, samples_b, training=False, filter_indices=filter_indices, index_list=index_list, vocab_to_common_vocab=vocab_to_common_vocab)
        cor_all += cor_b
        tot_all += tot_b

        for pred, sample, topk, vocab_loss in zip(pred_b, samples_b, topk_preds, common_vocab_loss):
            rel = sample['predicate_id']
            if rel not in result:
                result[rel] = (0, 0, 0, 0.0)
                list_of_predictions[rel] = []
            cor, tot, _, rel_tot_loss = result[rel]
            tot += 1
            cor += pred
            rel_tot_loss += vocab_loss
            result[rel] = (cor, tot, cor / tot if tot > 0 else 0.0, rel_tot_loss)
            list_of_predictions[rel].append({
                'uuid': sample['uuid'],
                'relation': sample['predicate_id'],
                'sub_label': sample['sub_label'],
                'obj_label': sample['obj_label'],
                'masked_sentences': sample['input_sentences'],
                'topk': topk,
            })
        
        eval_loss += loss.item() * tot_b
    
    if output_topk is not None:
        logger.info('Output top-k prediction to %s..'%output_topk)
        for rel in list_of_predictions:
            with open(os.path.join(output_topk, '%s_predictions.jsonl'%rel), 'w') as f:
                f.write('\n'.join([json.dumps(x) for x in list_of_predictions[rel]]))

    micro, macro = output_result(result, eval_loss)
    return micro, result

def gen_feature_sample(data_sample, template, mask_token='[MASK]'):
    feature_sample = {}
    feature_sample['predicate_id'] = data_sample['predicate_id']
    feature_sample['sub_label'] = data_sample['sub_label']
    feature_sample['obj_label'] = data_sample['obj_label']
    feature_sample['uuid'] = data_sample['uuid'] if 'uuid' in data_sample else ''
    masked_sentence = parse_template(template.strip(), feature_sample['sub_label'].strip(), mask_token)
    feature_sample['input_sentences'] = [masked_sentence[0]]
    return feature_sample

def load_data(data_path, template, vocab_subset=None, mask_token='[MASK]'):
    all_samples = []

    distinct_facts = set()
    raw_samples = load_file(data_path)
    for data_sample in raw_samples:
        # follow the LAMA setting, only keep distinct (sub, obj) pairs
        if (data_sample['sub_label'], data_sample['obj_label']) in distinct_facts:
            continue
        if (data_sample['obj_label'] not in vocab_subset):
            continue
        distinct_facts.add((data_sample['sub_label'], data_sample['obj_label']))

        feature_sample = gen_feature_sample(data_sample, template, mask_token)
        all_samples.append(feature_sample)

    return all_samples

