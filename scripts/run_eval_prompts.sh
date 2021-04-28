#!/bin/bash

method=${1}

if [ "${method}" = "lama" ]; then
    OUTPUTS_DIR=lama-outputs
    prompt_file=LAMA_relations.jsonl
fi

if [ "${method}" = "lpaqa" ]; then
    OUTPUTS_DIR=lpaqa-outputs
    prompt_file=LPAQA_relations.jsonl
fi

if [ "${method}" = "autoprompt" ]; then
    OUTPUTS_DIR=autoprompt-outputs
    prompt_file=AutoPrompt_relations.jsonl
fi

MODEL=bert-base-cased
RAND=none

for REL in P1001 P101 P103 P106 P108 P127 P1303 P131 P136 P1376 P138 P140 P1412 P159 P17 P176 P178 P19 P190 P20 P264 P27 P276 P279 P30 P31 P36 P361 P364 P37 P39 P407 P413 P449 P463 P47 P495 P527 P530 P740 P937; do

    DIR=${OUTPUTS_DIR}/${REL}
    mkdir -p ${DIR}

    python code/run_eval_prompts.py \
        --relation_profile relation_metainfo/${prompt_file} \
        --relation ${REL} \
        --common_vocab_filename common_vocabs/common_vocab_cased.txt \
        --model_name ${MODEL} \
        --test_data data/LAMA-TREx/${REL}.jsonl \
        --output_dir ${DIR} \
        --output_predictions 

done

python code/accumulate_results.py ${OUTPUTS_DIR}