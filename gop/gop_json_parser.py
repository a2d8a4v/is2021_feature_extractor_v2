import os
import argparse
from tqdm import tqdm

import glob
import yaml
import json
import numpy as np

from collections import defaultdict
from gop_web_parser import GOP
from scipy import stats

parser = argparse.ArgumentParser()

parser.add_argument("--json_fn",
                     default="models/chinese-model/model_online/gop_L2_DS_test_0928_capt_hires/json/gop_scores.json",
                     type=str)

parser.add_argument("--anno_fn",
                     default="../corpus/L2_DS_test_0928/data/annotation.txt",
                     type=str)

parser.add_argument("--text_fn",
                     default="data/L2_DS_test_0928/text",
                     type=str)

parser.add_argument("--lexicon_fn",
                     default="models/chinese-model/data/local/dict_hanyu_lg/lexicon.txt",
                     type=str)

parser.add_argument("--result_dir",
                     default="models/chinese-model/model_online/gop_L2_DS_test_0928_capt_hires/json",
                     type=str)
                    
args = parser.parse_args()

json_fn = args.json_fn
anno_fn = args.anno_fn
text_fn = args.text_fn
lexicon_fn = args.lexicon_fn

json_dict = {}
anno_dict = {}
text_dict = {}
lexicon_dict = {}

pred_scores_dict = defaultdict(list)
anno_scores_dict = defaultdict(list)

# Stage 0
# preprocess json
with open(json_fn, "r") as fn:
    json_dict = json.load(fn)

for utt_id in list(json_dict.keys()):
    GOP_info = json_dict[utt_id]["GOP"]
    for syb_info in GOP_info:
        syb = syb_info[0]
        phn_scores = syb_info[1]
        ave_score = int(phn_scores[-1][-1]) 
        pred_scores_dict[utt_id].append(ave_score)

# preprocess anno_fn
with open(anno_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()[0].split(",")
        utt_id = info[0]
        labels = info[1:]
        anno_dict[utt_id] = labels
        # average score
        ave_score = 0
        for i, lb in enumerate(labels):
            if lb == "T":
                ave_score += 1
            # T,T,T 為一組
            if i % 3 == 2:
                anno_scores_dict[utt_id].append(ave_score)
                ave_score = 0

# preprocess text_fn
with open(text_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        utt_id = info[0]
        sybs = info[1:]
        text_dict[utt_id] = sybs

# preprocess lexicon_fn
with open(lexicon_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        utt_id = info[0]
        prons = info[1:]
        lexicon_dict[utt_id] = prons

# Stage 1
# calcluate correlation for the test set
anno_scores_list = []
pred_scores_list = []

utt_ids = list(pred_scores_dict.keys())
print("Length of anno_scores_dict", len(list(anno_scores_dict.keys())))

for utt_id in utt_ids:
    if utt_id not in anno_scores_dict:
        continue
    
    anno_scores = anno_scores_dict[utt_id]
    pred_scores = pred_scores_dict[utt_id]
    assert len(anno_scores) == len(pred_scores)
    
    for i in range(len(anno_scores)):
        anno_score, pred_score = anno_scores[i], pred_scores[i]
        #if anno_score == 3 and pred_score < 8:
        #    continue
        #if anno_score == 2: and (pred_score > 9):
        #    continue
        #if anno_score == 0 and pred_score > 4:
        #    continue
         
        anno_scores_list.append(anno_score)
        pred_scores_list.append(pred_score)

print("Length of anno_scores_dict", len(anno_scores_list))
print(stats.spearmanr(anno_scores_list, pred_scores_list))

anno_scores_np = np.array(anno_scores_list)
pred_scores_np = np.array(pred_scores_list)

#selected_idx = np.where((anno_scores_np == 3) | (anno_scores_np == 0))
#selected_anno_scores_np = 1 * (anno_scores_np[selected_idx] > 0)
#selected_idx = np.where(anno_scores_np != 2)
selected_idx = np.where(anno_scores_np < 10)
selected_anno_scores_np = 1 * (anno_scores_np[selected_idx] > 1)

from sklearn.metrics import classification_report
import pprint
pp = pprint.PrettyPrinter(indent=4)

for th in range(0, 50, 10):
    selected_pred_scores_np = 1 * (pred_scores_np[selected_idx] > th)
    results = classification_report(selected_anno_scores_np, selected_pred_scores_np, output_dict=True)
    print(th)
    pp.pprint(results)
    print("="*10)

# Stage 2
# create results.csv
