import os
import json
import logging
import random

from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertConfig
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
from transformers import AutoConfig

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class Prober():

    def __init__(self, args, random_init='none'):
        assert(random_init in ['none', 'all', 'embedding'])

        super().__init__()

        self._model_device = 'cpu'

        model_name = args.model_name
        vocab_name = model_name

        if args.model_dir is not None:
            # load bert model from file
            model_name = str(args.model_dir) + "/"
            vocab_name = model_name
            logger.info("loading BERT model from {}".format(model_name))

        # Load pre-trained model tokenizer (vocabulary)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(args.seed)

        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, AlbertConfig):
            self.model_type = 'albert'
            self.tokenizer = AlbertTokenizer.from_pretrained(vocab_name)
            self.mlm_model = AlbertForMaskedLM.from_pretrained(model_name)
            if random_init == 'all':
                logger.info('Random initialize model...')
                self.mlm_model = AlbertForMaskedLM(self.mlm_model.config)
            self.base_model = self.mlm_model.albert
        elif isinstance(config, RobertaConfig):
            self.model_type = 'roberta'
            self.tokenizer = RobertaTokenizer.from_pretrained(vocab_name)
            self.mlm_model = RobertaForMaskedLM.from_pretrained(model_name)
            if random_init == 'all':
                logger.info('Random initialize model...')
                self.mlm_model = RobertaForMaskedLM(self.mlm_model.config)
            self.base_model = self.mlm_model.roberta
        elif isinstance(config, BertConfig):
            self.model_type = 'bert'
            self.tokenizer = BertTokenizer.from_pretrained(vocab_name)
            self.mlm_model = BertForMaskedLM.from_pretrained(model_name)
            if random_init == 'all':
                logger.info('Random initialize model...')
                self.mlm_model = BertForMaskedLM(self.mlm_model.config)
            self.base_model = self.mlm_model.bert
        else:
            raise ValueError('Model %s not supported yet!'%(model_name))

        self.mlm_model.eval()

        if random_init == 'embedding':
            logger.info('Random initialize embedding layer...')
            self.mlm_model._init_weights(self.base_model.embeddings.word_embeddings)

        # original vocab
        self.map_indices = None
        self.vocab = list(self.tokenizer.get_vocab().keys())
        logger.info('Vocab size: %d'%len(self.vocab))
        self._init_inverse_vocab()

        self.MASK = self.tokenizer.mask_token
        self.EOS = self.tokenizer.eos_token
        self.CLS = self.tokenizer.cls_token
        self.SEP = self.tokenizer.sep_token
        self.UNK = self.tokenizer.unk_token
        # print(self.MASK, self.EOS, self.CLS, self.SEP, self.UNK)

        self.pad_id = self.inverse_vocab[self.tokenizer.pad_token]
        self.unk_index = self.inverse_vocab[self.tokenizer.unk_token]

        # used to output top-k predictions
        self.k = args.k

    def _cuda(self):
        self.mlm_model.cuda()

    def try_cuda(self):
        """Move model to GPU if one is available."""
        if torch.cuda.is_available():
            if self._model_device != 'cuda':
                logger.info('Moving model to CUDA')
                self._cuda()
                self._model_device = 'cuda'
        else:
            logger.info('No CUDA found')

    def init_indices_for_filter_logprobs(self, vocab_subset, logger=None):
        index_list = []
        new_vocab_subset = []
        for word in vocab_subset:
            tokens = self.tokenizer.tokenize(' '+word)
            if (len(tokens) == 1) and (tokens[0] != self.UNK):
                index_list.append(self.tokenizer.convert_tokens_to_ids(tokens)[0])
                new_vocab_subset.append(word)
            else:
                msg = "word {} from vocab_subset not in model vocabulary!".format(word)
                if logger is not None:
                    logger.warning(msg)
                else:
                    logger.info("WARNING: {}".format(msg))

        indices = torch.as_tensor(index_list)
        return indices, index_list

    def _init_inverse_vocab(self):
        self.inverse_vocab = {w: i for i, w in enumerate(self.vocab)}

    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)

        return indexed_string

    def _get_input_tensors_batch_train(self, sentences_list, samples_list):
        tokens_tensors_list = []
        segments_tensors_list = []
        masked_indices_list = []
        tokenized_text_list = []
        mlm_labels_tensor_list = []
        mlm_label_ids = []

        max_tokens = 0
        for (sentences, samples) in zip(sentences_list, samples_list):
            tokens_tensor, segments_tensor, masked_indices, tokenized_text, mlm_labels_tensor, mlm_label_id = self.__get_input_tensors(sentences, mlm_label=samples['obj_label'])
            tokens_tensors_list.append(tokens_tensor)
            segments_tensors_list.append(segments_tensor)
            masked_indices_list.append(masked_indices)
            tokenized_text_list.append(tokenized_text)
            mlm_labels_tensor_list.append(mlm_labels_tensor)
            mlm_label_ids.append(mlm_label_id)
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]

        # apply padding and concatenate tensors
        # use [PAD] for tokens and 0 for segments
        final_tokens_tensor = None
        final_segments_tensor = None
        final_attention_mask = None
        final_mlm_labels_tensor = None
        for tokens_tensor, segments_tensor, mlm_labels_tensor in zip(tokens_tensors_list, segments_tensors_list, mlm_labels_tensor_list):
            dim_tensor = tokens_tensor.shape[1]
            pad_lenght = max_tokens - dim_tensor
            attention_tensor = torch.full([1,dim_tensor], 1, dtype= torch.long)
            if pad_lenght>0:
                pad_1 = torch.full([1,pad_lenght], self.pad_id, dtype= torch.long)
                pad_2 = torch.full([1,pad_lenght], 0, dtype= torch.long)
                attention_pad = torch.full([1,pad_lenght], 0, dtype= torch.long)
                pad_3 = torch.full([1,pad_lenght], -100, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor,pad_1), dim=1)
                segments_tensor = torch.cat((segments_tensor,pad_2), dim=1)
                attention_tensor = torch.cat((attention_tensor,attention_pad), dim=1)
                mlm_labels_tensor = torch.cat((mlm_labels_tensor, pad_3), dim=1)
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_segments_tensor = segments_tensor
                final_attention_mask = attention_tensor
                final_mlm_labels_tensor = mlm_labels_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0)
                final_segments_tensor = torch.cat((final_segments_tensor,segments_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask,attention_tensor), dim=0)
                final_mlm_labels_tensor = torch.cat((final_mlm_labels_tensor,mlm_labels_tensor), dim=0)

        return final_tokens_tensor, final_segments_tensor, final_attention_mask, masked_indices_list, tokenized_text_list, final_mlm_labels_tensor, mlm_label_ids

    def __get_input_tensors_batch(self, sentences_list):
        tokens_tensors_list = []
        segments_tensors_list = []
        masked_indices_list = []
        tokenized_text_list = []
        max_tokens = 0
        for sentences in sentences_list:
            tokens_tensor, segments_tensor, masked_indices, tokenized_text = self.__get_input_tensors(sentences)
            tokens_tensors_list.append(tokens_tensor)
            segments_tensors_list.append(segments_tensor)
            masked_indices_list.append(masked_indices)
            tokenized_text_list.append(tokenized_text)
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
        # logger.info("MAX_TOKENS: {}".format(max_tokens))
        # apply padding and concatenate tensors
        # use [PAD] for tokens and 0 for segments
        final_tokens_tensor = None
        final_segments_tensor = None
        final_attention_mask = None
        for tokens_tensor, segments_tensor in zip(tokens_tensors_list, segments_tensors_list):
            dim_tensor = tokens_tensor.shape[1]
            pad_lenght = max_tokens - dim_tensor
            attention_tensor = torch.full([1,dim_tensor], 1, dtype= torch.long)
            if pad_lenght>0:
                pad_1 = torch.full([1,pad_lenght], self.pad_id, dtype= torch.long)
                pad_2 = torch.full([1,pad_lenght], 0, dtype= torch.long)
                attention_pad = torch.full([1,pad_lenght], 0, dtype= torch.long)
                tokens_tensor = torch.cat((tokens_tensor,pad_1), dim=1)
                segments_tensor = torch.cat((segments_tensor,pad_2), dim=1)
                attention_tensor = torch.cat((attention_tensor,attention_pad), dim=1)
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_segments_tensor = segments_tensor
                final_attention_mask = attention_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0)
                final_segments_tensor = torch.cat((final_segments_tensor,segments_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask,attention_tensor), dim=0)
        # logger.info(final_tokens_tensor)
        # logger.info(final_segments_tensor)
        # logger.info(final_attention_mask)
        # logger.info(final_tokens_tensor.shape)
        # logger.info(final_segments_tensor.shape)
        # logger.info(final_attention_mask.shape)
        return final_tokens_tensor, final_segments_tensor, final_attention_mask, masked_indices_list, tokenized_text_list

    def __get_input_tensors(self, sentences, mlm_label=None):

        if len(sentences) > 2:
            logger.info(sentences)
            raise ValueError("BERT accepts maximum two sentences in input for each data point")

        first_tokenized_sentence = [self.tokenizer.tokenize(token) if ((not token.startswith('[unused')) and (token != self.MASK)) else [token] for token in sentences[0].split()]
        first_tokenized_sentence = [item for sublist in first_tokenized_sentence for item in sublist]
        if self.model_type == 'roberta':
            first_tokenized_sentence = self.tokenizer.tokenize(sentences[0])
        first_segment_id = np.zeros(len(first_tokenized_sentence), dtype=int).tolist()

        # add [SEP] token at the end
        first_tokenized_sentence.append(self.SEP)
        first_segment_id.append(0)

        if len(sentences)>1 :
            second_tokenized_sentece = [self.tokenizer.tokenize(token) if not token.startswith('[unused') else [token] for token in sentences[1].split()]
            second_tokenized_sentece = [item for sublist in second_tokenized_sentece for item in sublist]
            if self.model_type == 'roberta':
                second_tokenized_sentece = self.tokenizer.tokenize(sentences[1])
            second_segment_id = np.full(len(second_tokenized_sentece),1, dtype=int).tolist()

            # add [SEP] token at the end
            second_tokenized_sentece.append(self.SEP)
            second_segment_id.append(1)

            tokenized_text = first_tokenized_sentence + second_tokenized_sentece
            segments_ids = first_segment_id + second_segment_id
        else:
            tokenized_text = first_tokenized_sentence
            segments_ids = first_segment_id

        # add [CLS] token at the beginning
        tokenized_text.insert(0,self.CLS)
        segments_ids.insert(0,0)

        # look for masked indices
        masked_indices = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == self.MASK:
                masked_indices.append(i)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        if mlm_label is None:
            return tokens_tensor, segments_tensors, masked_indices, tokenized_text

        # Handle mlm_label
        mlm_labels = np.full(len(tokenized_text), -100, dtype=int).tolist()
        tmp_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '+mlm_label))
        assert(len(tmp_ids) == 1)
        mlm_labels[masked_indices[-1]] = tmp_ids[0]
        mlm_labels_tensor = torch.tensor([mlm_labels])

        return tokens_tensor, segments_tensors, masked_indices, tokenized_text, mlm_labels_tensor, tmp_ids[0]

    def __get_token_ids_from_tensor(self, indexed_string):
        token_ids = []
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)
            token_ids = np.asarray(indexed_string)
        else:
            token_ids = indexed_string
        return token_ids

    def get_batch_generation(self, sentences_list, logger= None,
                             try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)

        if logger is not None:
            logger.debug("\n{}\n".format(tokenized_text_list))

        with torch.no_grad():
            logits = self.mlm_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
            )

            log_probs = F.log_softmax(logits, dim=-1).cpu()

        token_ids_list = []
        for indexed_string in tokens_tensor.numpy():
            token_ids_list.append(self.__get_token_ids_from_tensor(indexed_string))

        return log_probs, token_ids_list, masked_indices_list

    def run_batch(self, sentences_list, samples_list, try_cuda=True, training=True, filter_indices=None, index_list=None, vocab_to_common_vocab=None):
        if try_cuda and torch.cuda.device_count() > 0:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list, mlm_labels_tensor, mlm_label_ids = self._get_input_tensors_batch_train(sentences_list, samples_list)

        if training:
            self.mlm_model.train()
            loss = self.mlm_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
                masked_lm_labels=mlm_labels_tensor.to(self._model_device),
            )
            loss = loss[0]
        else:
            self.mlm_model.eval()
            with torch.no_grad():
                loss, logits = self.mlm_model(
                    input_ids=tokens_tensor.to(self._model_device),
                    token_type_ids=segments_tensor.to(self._model_device),
                    attention_mask=attention_mask_tensor.to(self._model_device),
                    masked_lm_labels=mlm_labels_tensor.to(self._model_device),
                )
            log_probs = F.log_softmax(logits, dim=-1).cpu()

        if training:
            return loss
        else:
            # During testing, return accuracy and top-k predictions
            tot = log_probs.shape[0]
            cor = 0
            preds = []
            topk = []
            common_vocab_loss = []

            for i in range(log_probs.shape[0]):
                masked_index = masked_indices_list[i][0]
                log_prob = log_probs[i][masked_index]
                mlm_label = mlm_label_ids[i]
                if filter_indices is not None:
                    log_prob = log_prob.index_select(dim=0, index=filter_indices)
                    pred_common_vocab = torch.argmax(log_prob)
                    pred = index_list[pred_common_vocab]

                    # get top-k predictions
                    topk_preds = []
                    topk_log_prob, topk_ids = torch.topk(log_prob, self.k)
                    for log_prob_i, idx in zip(topk_log_prob, topk_ids):
                        ori_idx = index_list[idx]
                        token = self.vocab[ori_idx]
                        topk_preds.append({'token': token, 'log_prob': log_prob_i.item()})
                    topk.append(topk_preds)

                    # compute entropy on common vocab
                    common_logits = logits[i][masked_index].cpu().index_select(dim=0, index=filter_indices)
                    common_log_prob = -F.log_softmax(common_logits, dim=-1)
                    common_label_id = vocab_to_common_vocab[mlm_label]
                    common_vocab_loss.append(common_log_prob[common_label_id].item())
                else:
                    pred = torch.argmax(log_prob)
                    topk.append([])
                if pred == mlm_labels_tensor[i][masked_index]:
                    cor += 1
                    preds.append(1)
                else:
                    preds.append(0)
                            
            return log_probs, cor, tot, preds, topk, loss, common_vocab_loss 
