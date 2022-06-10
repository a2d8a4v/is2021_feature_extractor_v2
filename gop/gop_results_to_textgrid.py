'''
Single Word
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = 0.66 
tiers? <exists> 
size = 1
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "phones" 
        xmin = 0 
        xmax = 0.66 
        intervals: size = 3 
        intervals [1]:
            xmin = 0 
            xmax = 0.06 
            text = "SIL" 
        intervals [2]:
            xmin = 0.06 
            xmax = 0.24 
            text = "l" 
        intervals [3]:
            xmin = 0.24 
            xmax = 0.66 
            text = "uo4" 
'''

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

parser.add_argument("--lang_dir",
                     default="models/chinese-model/data/lang",
                     type=str)

parser.add_argument("--result_dir",
                     default="models/chinese-model/model_online/gop_L2_DS_test_0928_capt_hires",
                     type=str)
                     
parser.add_argument("--corpus_dir",
                     default="../corpus/L2_DS_test",
                     type=str)                     


parser.add_argument("--utt2dur_fn",
                     default="data/L2_DS_test/utt2dur",
                     type=str)                     
                    
args = parser.parse_args()

json_fn = args.json_fn
lang_dir = args.lang_dir
result_dir = args.result_dir
corpus_dir = args.corpus_dir
phones_txt = os.path.join(lang_dir, "phones.txt")
utt2dur_fn = args.utt2dur_fn

#{"utt_id": {"GOP": [word:[phn, gop_score]], ...}}
gop_json = defaultdict(dict)
#{"utt_id": ["phn": [start_time, duration], ...]}
ctm_info = defaultdict(list)
phn_tbl = {}
phn_inv_tbl = {}
utt2dur_info = {}
sil_tbl = ["SIL", "SIL_B", "SIL_E", "SIL_I", "SIL_S", "SPN", "SPN_B", "SPN_E", "SPN_I", "SPN_S"]

# Stage 0
# preprocess phones.txt
with open(phones_txt, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        phn, phn_id = info
        
        if phn == "<eps>":
            continue
        
        phn_tbl[phn] = phn_id
        phn_inv_tbl[phn_id] = phn


end_time = 1000000000
# preprocess ctm infomation of the result_dir
with open(result_dir + "/phone.ctm") as fn:
    for line in fn.readlines():
        info = line.split()
        utt_id, _, start_time, duration, phn_id = info
        
        if utt_id not in ctm_info:
            end_time = 1000000000
        
        if phn_inv_tbl[phn_id] in sil_tbl:
            continue
        
        start_time = round(float(start_time), 4)
        if start_time > end_time:
            start_time = end_time
        end_time = start_time + float(duration)
        end_time = round(end_time, 4)
        
        ctm_info[utt_id].append([phn_inv_tbl[phn_id], start_time, end_time])

# preprocess GOP json
with open(json_fn, "r") as fn:
    gop_json = json.load(fn)


# preprocess utt2dur
with open(utt2dur_fn, "r") as fn:
    for line in fn.readlines():
        utt, dur = line.split()
        utt2dur_info[utt] = float(dur)
        

corr_word_dict = {}
perc_phn_dict = {}
# create phone dict
for utt_id in list(gop_json.keys()):
    utt_gop_infos = gop_json[utt_id]["GOP"]
    corr_word_dict[utt_id] = {"Words":[], "Time":[]}
    perc_phn_dict[utt_id] = {"Phones":[], "Time":[]}
    
    for word_phn_infos in utt_gop_infos:
        word, phn_infos = word_phn_infos
        assert len(ctm_info[utt_id]) == len(phn_infos) -1
        min_time = ctm_info[utt_id][0][1]
        
        for idx, phn_info in enumerate(phn_infos):
            phn, conf = phn_info
            conf = format(float(conf), ".2f")
            if phn == "average":
                corr_word_dict[utt_id]["Words"].append([word, min_time, max_time, conf])
            else:
                assert ctm_info[utt_id][idx][0] == phn
                # removed position tags (e.g., _I, _S, or _E)
                if "_" in phn:
                    phn = phn.split("_")[0]
                perc_phn_dict[utt_id]["Phones"].append([phn, ctm_info[utt_id][idx][1], ctm_info[utt_id][idx][2]])
                max_time = ctm_info[utt_id][idx][2]
                
    corr_word_dict[utt_id]["Time"] = ctm_info[utt_id][-1][2]
    perc_phn_dict[utt_id]["Time"] = ctm_info[utt_id][-1][2]
        

# Stage 1
# generated textgrid
def write_interval(tg_fn, phn_list, xmax, name, num_items):
    tg_fn.write("\titem ["+ num_items +"]:\n")
    tg_fn.write("\t\tclass = \"IntervalTier\"\n")
    tg_fn.write("\t\tname = \"" + name + "\"\n")
    tg_fn.write("\t\txmin = 0\n")
    tg_fn.write("\t\txmax = " + str(xmax) + "\n")
    tg_fn.write("\t\tintervals: size = " + str(len(phn_list)) + "\n")
    
    for i in range(len(list(phn_list))):
        if len(phn_list[i]) == 4:
            phn, xmin, xmax, conf = phn_list[i]
        else:
            phn, xmin, xmax = phn_list[i]
            
        tg_fn.write("\t\tintervals [" + str(i+1) + "]:\n")
        tg_fn.write("\t\t\txmin = " + str(xmin) + "\n")
        tg_fn.write("\t\t\txmax = " + str(xmax) + "\n")
        
        if len(phn_list[i]) == 4:
            tg_fn.write("\t\t\ttext = \"" + phn + "," + conf + "\"\n")
        else:
            tg_fn.write("\t\t\ttext = \"" + phn + "\"\n")

def add_empty_boundary(phn_list, max_time):
    # [phn, xmin, xmax, conf (option)]
    fst_phn_info = phn_list[0]
    last_phn_info = phn_list[-1]
    
    # xmin > 0
    if fst_phn_info[1] == 0:
        phn_list[0][1] += 0.0001
    if last_phn_info[2] == max_time:
        phn_list[-1][2] -= 0.0001
        
    if fst_phn_info[1] > 0:
        phn_list.insert(0, ["", 0, fst_phn_info[1]])
    # xmax < max_time
    if last_phn_info[2] < max_time:
        phn_list.append(["", last_phn_info[2], max_time])

for fname in list(corr_word_dict.keys()):
    fname_info = fname.split("_")
    _, grade, class_no, _, _ = fname_info
    grade_no = "_".join([grade[1:], class_no])
    dest_dir = os.path.join(corpus_dir, grade[:2], grade_no)
    print(dest_dir)
    with open(dest_dir + "/" + fname + ".textgrid", "w") as tg_fn:
        tg_fn.write("File type = \"ooTextFile\"\n")
        tg_fn.write("Object class = \"TextGrid\"\n")
        tg_fn.write("\n")
        tg_fn.write("xmin = 0\n")
        corr_xmax = corr_word_dict[fname]["Time"]
        perc_xmax = perc_phn_dict[fname]["Time"]
        #if corr_xmax > perc_xmax:
        #    tg_fn.write("xmax = " + str(corr_xmax) + "\n")
        #else:
        #    tg_fn.write("xmax = " + str(perc_xmax) + "\n")
        max_time = utt2dur_info[fname]
        tg_fn.write("xmax = " + str(max_time) + "\n")
        
        phn_list_ipa = []
        for phn_list in perc_phn_dict[fname]["Phones"]:
            phn_list_ipa.append(["", phn_list[1], phn_list[2]])
        
        add_empty_boundary(corr_word_dict[fname]["Words"], max_time)
        add_empty_boundary(perc_phn_dict[fname]["Phones"], max_time)
        add_empty_boundary(phn_list_ipa, max_time)
        
        tg_fn.write("tiers? <exists>\n")
        tg_fn.write("size = 4\n")
        tg_fn.write("item []:\n")
        write_interval(tg_fn, corr_word_dict[fname]["Words"], corr_xmax, "CorrectWord", "1")
        write_interval(tg_fn, perc_phn_dict[fname]["Phones"], perc_xmax, "PerceivedPhone", "2")
        write_interval(tg_fn, phn_list_ipa, perc_xmax, "PerceivedPhoneIPA", "3")
        write_interval(tg_fn, [["", 0, max_time]], max_time, "Notes", "4")
