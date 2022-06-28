import json
import os
import argparse
from tqdm import tqdm
from utilities import (
    jsonLoad,
    open_utt2value,
    pickleStore
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json_file_path",
                        default="data/train/all.json",
                        type=str)

    parser.add_argument("--tmp_decoding_list_file_path",
                        default="data/train/all.json",
                        type=str)

    args = parser.parse_args()

    # variables
    input_json_file_path = args.input_json_file_path
    tmp_decoding_list_file_path = args.tmp_decoding_list_file_path

    # processing
    utt_pkl_dict = open_utt2value(tmp_decoding_list_file_path)

    # json file without utts key
    json_data = jsonLoad(input_json_file_path)['utts']

    print('processing...')

    for utt_id, utt_info in tqdm(json_data.items()):

        get_pkl_path = utt_pkl_dict[utt_id]
        pickleStore(utt_info, get_pkl_path)
    
    print('Done!')
