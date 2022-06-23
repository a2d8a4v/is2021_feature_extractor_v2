import json
import sys


def jsonLoad(scores_json):
    with open(scores_json) as json_file:
        return json.load(json_file)


def jsonSave(path, dict_json):
    with open(path, "w") as f:
        json.dump(dict_json, f, indent=4)


def opentext( file, col_start ):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            f_ = []
            for w in l_[col_start:]:
                f_.append(checkDisfluency(w))
            s[l_[0]] = f_
    return s


## START
file = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/cefr_train_tr/gigaspeech_20220525_prompt/all.json"
data = jsonLoad(file)['utts']

## SPLIT
new = {}
for i, (utt_id, d) in enumerate(data.items()):
    new[utt_id] = d
    if i > 10:
        break

save = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/cefr_train_tr/gigaspeech_20220525_prompt/all.subset_20220606.json"
jsonSave(save, {'utts':new})