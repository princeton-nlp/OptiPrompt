This folder contains the vobacularies we used in our experiments. 
For the main results, we use `common_vocab_cased.txt`; for the experiments comparing BERT, RoBERTa, and ALBERT, we create a new vocabulary `common_vocab_cased_be_ro_al.txt`.

Vocabs:
* `common_vocab_cased.txt`: the common vocab released in [LAMA](https://github.com/facebookresearch/LAMA). It is the intersection of vocabularies of five models (Transformer-XL, BERT, ELMo, GPT, and RoBERTa). We use this file in our main experiments for fair comparision.
* `common_vocab_cased_be_ro_al.txt`: the common vocab we used to compared BERT, RoBERTa, and ALBERT (see Appendix C). It is the intersection of vocabularies of BERT, RoBERTa, and ALBERT.
