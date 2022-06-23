import json
import sys
import csv

def jsonLoad(scores_json):
    with open(scores_json, encoding="utf8") as json_file:
        return json.load(json_file)


def jsonSave(path, dict_json):
    with open(path, "w") as f:
        json.dump(dict_json, f, indent=4)


def opendict(file):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            s[l_[0]] = l_[1]
    return s

"""
{ "stt": text, "stt(g2p)": phone_text, "prompt": text_prompt,
                "wav_path": wav_path, "ctm": ctm_info, 
                "feats": {  **f0_info, **energy_info, 
                            **sil_feats_info, **word_feats_info,
                            **phone_feats_info,
                            "pitch": pitch_feats_info,
                            "intensity": intensity_feats_info,
                            "formant": formants_info,
                            "rhythm": rhythm_feats_info,
                            "total_duration": total_duration,
                            "tobi": textgrid_file_path,
                            "response_duration": response_duration}
}
"""

# variables
scores_json = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/dump/cefr_train_cv/deltafalse/data.json"
utt_json_data = jsonLoad(scores_json)['utts']

save = {}
level = {}
for utt_id, data in utt_json_data.items():

    text_prompt = data.get('output')[0].get('text_prompt')
    level[utt_id] = data.get('output')[0].get('text')
    save[utt_id] = text_prompt

print(len(save.keys()))

with open("/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/cerf_train_cv/fix", 'w') as fp:
    for utt_id, text in save.items():
        fp.write("{} {}\n".format(utt_id, text))

with open("/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/cerf_train_cv/level", 'w') as fp:
    for utt_id, text in level.items():
        fp.write("{} {}\n".format(utt_id, text))