import json
import sys
import csv
import argparse
from utilities import (
    jsonLoad,
    jsonSave,
    open_utt2value
)


if __name__ == '__main__':


    """
        { "stt": text, "stt(g2p)": phone_text, "prompt": text_prompt,
                        "wav_path": wav_path, "ctm": ctm_info, "phone_ctm": phone_ctm_info,
                        "feats": {  **f0_info, **energy_info, 
                                    **sil_feats_info, **word_feats_info,
                                    **phone_feats_info,
                                    "pitch": pitch_feats_info,
                                    "intensity": intensity_feats_info,
                                    "formant": formants_info,
                                    "rhythm": rhythm_feats_info,
                                    "total_duration": total_duration,
                                    "tobi": tobi_feats_info,
                                    "tg_path": textgrid_file_path,
                                    "response_duration": response_duration}
        }
    """

    ## args
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json_file_path",
                        default="dump/train/deltafalse/data.json",
                        type=str)

    parser.add_argument("--input_scale_file_path",
                        default="data/train/scale",
                        type=str)

    parser.add_argument("--save_csv_file_path",
                        default="./apl_features_gop_s2t_train_cv.csv",
                        type=str)

    args = parser.parse_args()


    ## variables
    input_json_file_path = args.input_json_file_path
    input_scale_file_path = args.input_scale_file_path
    save_csv_file_path = args.save_csv_file_path

    level_map = {
        'a1': 1,
        'a2': 2,
        'b1': 3
    }
    rhy_columns = [
        'percent_v', 'rpvi_v', 'cci_v', 'rpvi_c', 'v_std_dev', 'cci', 'rpvi', 'mean_p_dur', 'npvi', 'v_sum_dur_sec', 'p_sum_variance', 'speech_rate', 'num_nucleus', 'sum_dur_sec', 'c_std_dev', 'varco_V', 'v_num', 'mean_v_dur', 'phones_num', 'varco_C', 'c_num', 'npvi_c', 'p_std_dev', 'cci_c', 'c_sum_dur_sec', 'mean_c_dur', 'num_consonants', 'v_to_c_ratio', 'npvi_v', 'status', 'varco_P'
    ]
    csv_columns = [
        'phone_conf_median', 'long_sil_summ', 'phone_duration_mad', 'f0_mean', 'f0_nz_max', 'sil_rate2', 'phone_conf_max', 'long_sil_max', 'word_duration_mad', 'f0_nz_summ', 'f0_nz_std', 'word_count', 'word_conf_summ', 'phone_duration_std', 'sil_std', 'f0_max', 'phone_conf_summ', 'word_num_disfluency_baseline', 'phone_conf_mean', 'sil_rate1', 'f0_nz_number', 'response_duration', 'long_sil_mad', 'phone_duration_number', 'word_conf_std', 'energy_min', 'word_num_repeat', 'sil_number', 'word_duration_mean', 'f0_summ', 'f0_nz_mean', 'f0_mad', 'phone_conf_std', 'energy_mad', 'long_sil_mean', 'phone_duration_median', 'total_duration', 'energy_std', 'phone_conf_mad', 'long_sil_min', 'energy_mean', 'word_duration_median', 'word_conf_mean', 'f0_number', 'word_conf_max', 'f0_nz_min', 'f0_std', 'word_duration_min', 'word_conf_median', 'sil_median', 'f0_nz_mad', 'energy_summ', 'word_duration_max', 'energy_median', 'word_duration_std', 'f0_median', 'word_freq', 'phone_duration_mean', 'sil_mad', 'word_distinct', 'phone_freq', 'phone_duration_min', 'long_sil_std', 'long_sil_number', 'sil_max', 'f0_nz_median', 'f0_min', 'word_num_disfluency', 'long_sil_rate1', 'phone_duration_max', 'energy_number', 'word_conf_min', 'word_duration_summ', 'phone_conf_min', 'sil_mean', 'long_sil_median', 'phone_conf_number', 'word_conf_mad', 'word_duration_number', 'long_sil_rate2', 'phone_count', 'energy_max', 'phone_duration_summ', 'sil_min', 'sil_summ', 'word_conf_number']
    level_columns = [
        "level"
    ]

    _data = jsonLoad(input_json_file_path)['utts']
    _levels = open_utt2value(input_scale_file_path) 

    # filter out the data we want to save
    dict_data = []
    for utt_id, data in _data.items():

        data = data.get('input')[1].get('feats')

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

    try:
        with open(save_csv_file_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns+rhy_columns+level_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")