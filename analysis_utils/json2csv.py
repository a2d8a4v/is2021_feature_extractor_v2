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


## START
_file = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/dump/cefr_train_cv/deltafalse/data.new.false.dis.wer.json"
level = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/cefr_train_cv/text"
_data = jsonLoad(_file)['utts']
_levels = opendict(level) 
level_map = {
    'a1': 1,
    'a2': 2,
    'b1': 3
}

rhy_columns = [
    'npvi_v',
    'cci_c',
    'varco_V',
    'varco_P',
    'v_num',
    'num_consonants',
    'rpvi',
    'v_std_dev',
    'phones_num',
    'mean_v_dur',
    'varco_C',
    'speech_rate',
    'c_sum_dur_sec',
    'rpvi_c',
    'v_sum_dur_sec',
    'num_nucleus',
    'c_num',
    'npvi_c',
    'mean_c_dur',
    'v_to_c_ratio',
    'p_std_dev',
    'cci',
    'p_sum_variance',
    'mean_p_dur',
    'status',
    'cci_v',
    'c_std_dev',
    'npvi',
    'sum_dur_sec',
    'rpvi_v',
    'percent_v'
]
csv_columns = [
    'energy_mad',
    'energy_max',
    'energy_mean',
    'energy_median',
    'energy_min',
    'energy_number',
    'energy_std',
    'energy_summ',
    'f0_mad',
    'f0_max',
    'f0_mean',
    'f0_median',
    'f0_min',
    'f0_number',
    'f0_nz_mad',
    'f0_nz_max',
    'f0_nz_mean',
    'f0_nz_median',
    'f0_nz_min',
    'f0_nz_number',
    'f0_nz_std',
    'f0_nz_summ',
    'f0_std',
    'f0_summ',
    'long_sil_mad',
    'long_sil_max',
    'long_sil_mean',
    'long_sil_median',
    'long_sil_min',
    'long_sil_number',
    'long_sil_rate1',
    'long_sil_rate2',
    'long_sil_std',
    'long_sil_summ',
    'phone_conf_mad',
    'phone_conf_max',
    'phone_conf_mean',
    'phone_conf_median',
    'phone_conf_min',
    'phone_conf_number',
    'phone_conf_std',
    'phone_conf_summ',
    'phone_count',
    'phone_duration_mad',
    'phone_duration_max',
    'phone_duration_mean',
    'phone_duration_median',
    'phone_duration_min',
    'phone_duration_number',
    'phone_duration_std',
    'phone_duration_summ',
    'phone_freq',
    'phrase_num_disfluency_phrases',
    'response_duration',
    'sil_mad',
    'sil_max',
    'sil_mean',
    'sil_median',
    'sil_min',
    'sil_number',
    'sil_rate1',
    'sil_rate2',
    'sil_std',
    'sil_summ',
    'total_duration',
    'word_conf_mad',
    'word_conf_max',
    'word_conf_mean',
    'word_conf_median',
    'word_conf_min',
    'word_conf_number',
    'word_conf_std',
    'word_conf_summ',
    'word_count',
    'word_distinct'
    'word_duration_mad',
    'word_duration_max',
    'word_duration_mean',
    'word_duration_median',
    'word_duration_min',
    'word_duration_number',
    'word_duration_std',
    'word_duration_summ',
    'word_freq',
    'word_num_disfluency'
    'word_num_disfluency_baseline',
    'word_num_repeat',
]
level_columns = [
    "level"
]

# filter out the data we want to save

dict_data = []
for utt_id, data in _data.items():

    data = data.get('input')[1].get('feats')

    print(data.get('formant')[0])
    print(data.get('formant')[100])
    print(data.get('formant')[200])
    print(len(data.get('formant')))
    print(len(data.get('f0_list')))
    input()

    save = {}
    for col in csv_columns:
        d = data.get(col, 0.0)
        save[col] = d
    for col in rhy_columns:
        d = data.get("rhythm").get(col)
        save[col] = d
    for col in level_columns:
        save[col] = level_map.get(_levels.get(utt_id))
    dict_data.append(save)

csv_file = "apl_features_prompt_train_cv.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns+rhy_columns+level_columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
except IOError:
    print("I/O error")