import json
import sys


def jsonLoad(scores_json):
    with open(scores_json) as json_file:
        return json.load(json_file)


def jsonSave(path, dict_json):
    with open(path, "w") as f:
        json.dump(dict_json, f, indent=4)



## START
file = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/dump/cefr_train_tr/deltafalse/data.dis.pk.json"
data = jsonLoad(file)

save = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/dump/cefr_train_tr/deltafalse/data.dis.pk.json.new"
jsonSave(save, {'utts':data})