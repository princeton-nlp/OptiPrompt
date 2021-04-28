#!/bin/bash

OUTPUTS_DIR=finetune-outputs
MODEL=bert-base-cased
RAND=none

for REL in P1001 P101 P103 P106 P108 P127 P1303 P131 P136 P1376 P138 P140 P1412 P159 P17 P176 P178 P19 P190 P20 P264 P27 P276 P279 P30 P31 P36 P361 P364 P37 P39 P407 P413 P449 P463 P47 P495 P527 P530 P740 P937; do

    DIR=${OUTPUTS_DIR}/${REL}
    mkdir -p ${DIR}

    python code/run_finetune.py \
        --relation_profile relation_metainfo/LAMA_relations.jsonl \
        --relation ${REL} \
        --common_vocab_filename common_vocabs/common_vocab_cased.txt \
        --model_name ${MODEL} \
        --do_train \
        --train_data data/autoprompt_data/${REL}/train.jsonl \
        --dev_data data/autoprompt_data/${REL}/dev.jsonl \
        --do_eval \
        --test_data data/LAMA-TREx/${REL}.jsonl \
        --output_dir ${DIR} \
        --random_init ${RAND} \
        --output_predictions 

done

python code/accumulate_results.py ${OUTPUTS_DIR}