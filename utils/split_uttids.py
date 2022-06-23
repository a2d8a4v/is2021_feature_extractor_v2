import json
import os
import argparse
from utilities import (
    splitList,
    open_utt2value,
    write_txt_with_uttids,
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--split_number",
                        default=10,
                        required=True,
                        type=int)

    parser.add_argument("--tmp_decoding_list_file_path",
                        default="data/train/all.json",
                        type=str)

    parser.add_argument("--output_dir_path",
                        default="data/train/",
                        type=str)

    args = parser.parse_args()

    # variables
    tmp_decoding_list_file_path = args.tmp_decoding_list_file_path
    output_dir_path = args.output_dir_path
    split_number = args.split_number

    # processing
    utt_dict = open_utt2value(tmp_decoding_list_file_path)
    utt_list = list(utt_dict.keys())

    listTemp = splitList(utt_list, split_number)
    for i, chunk_utt_list in enumerate(listTemp):

        new_file_name = 'utt_pkl' + ".{}".format(i+1) + '.list'
        new_file_path = os.path.join(output_dir_path, new_file_name)

        filterd_utt_dict = {utt_id:utt_dict[utt_id] for utt_id in chunk_utt_list}

        # generate file
        write_txt_with_uttids(filterd_utt_dict, new_file_path)

