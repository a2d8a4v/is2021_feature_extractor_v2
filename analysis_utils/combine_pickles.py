import json
import sys
import argparse

sys.path.insert(0, os.path.abspath('../stt')) # Remember to add this line to avoid "module no exist" error

from utils import (
    pikleOpen,
    pickleStore
)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json_files",
                        default="data/train/all.json",
                        type=str)

    parser.add_argument("--output_dir_path",
                        default="data/train/all.json",
                        type=str)

    args = parser.parse_args()

    # variables
    input_json_files = args.input_json_files
    output_dir_path = args.output_dir_path
    save_json = {}

    for pkl_file_path in input_json_files:
        data_dict = pikleOpen(pkl_file_path)
        for vowel, vowel_info in data_dict.items():
            save_json.setdefault(save_json, []).extend(vowel_info)

    pickleStore(save_json, output_dir_path)

