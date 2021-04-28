import json
import argparse
import os
import random
import sys
import logging

from utils import load_file

output_dir = sys.argv[1]
relations = 'P1001 P101 P103 P106 P108 P127 P1303 P131 P136 P1376 P138 P140 P1412 P159 P17 P176 P178 P19 P190 P20 P264 P27 P276 P279 P30 P31 P36 P361 P364 P37 P39 P407 P413 P449 P463 P47 P495 P527 P530 P740 P937'.split()

tot = 0
cor = 0

rel_avg = []

for relation in relations:
    rel_tot = 0
    rel_cor = 0
    samples = load_file(os.path.join(output_dir, '%s/%s_predictions.jsonl'%(relation, relation)))
    for sample in samples:
        if sample['obj_label'] == sample['topk'][0]['token']:
            rel_cor += 1
        rel_tot += 1

    rel_avg.append(rel_cor / rel_tot)
    tot += rel_tot
    cor += rel_cor

    print('%s\t%.2f\t(%d / %d)'%(relation, (rel_cor / rel_tot * 100), rel_cor, rel_tot))

micro = sum(rel_avg) / len(rel_avg) * 100
macro = cor / tot * 100

print('Macro: %.2f\t(%d / %d)'%(macro, cor, tot))
print('Micro: %.2f'%(micro))
