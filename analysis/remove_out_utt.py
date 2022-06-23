import json
import sys
import pickle


def jsonLoad(scores_json):
    with open(scores_json) as json_file:
        return json.load(json_file)


def jsonSave(path, dict_json):
    with open(path, "w") as f:
        json.dump(dict_json, f, indent=4)


def pikleOpen( filename ):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p


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
file = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/dump/cefr_train_tr/deltafalse/data.new.dis.json"
data = jsonLoad(file)['utts']

## SPLIT
new = {}
pro = []
for i, (utt_id, d) in enumerate(data.items()):
    _f = data.get(utt_id).get('input')[2].get('save_path_pickle')
    _d = pikleOpen(_f)
    if not _d.get('word_index_list'):
        pro.append(utt_id)
        continue
    new[utt_id] = d
    
print(pro)
print(len(pro))

if pro:
    save = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/dump/cefr_train_tr/deltafalse/data.new.dis.json.20220426"
    jsonSave(save, {'utts':new})