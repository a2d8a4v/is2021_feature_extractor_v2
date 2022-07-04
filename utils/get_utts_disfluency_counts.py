import argparse
from utilities import (
    pikleOpen,
    open_utt2value
)

def nullable_string(val):
    if val.lower() == 'none':
        return None
    return val


if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_utts_list_file_path",
                        default="data/train/ctm",
                        type=str)

    parser.add_argument("--input_tmp_decoding_list_file_path",
                        default="data/train/ctm",
                        type=str)

    parser.add_argument("--scale",
                        default="A1",
                        type=str)

    # parser.add_argument("--output_file_path",
    #                     default="-",
    #                     type=str) 

    args = parser.parse_args()

    # variables
    input_tmp_decoding_list_file_path = args.input_tmp_decoding_list_file_path
    input_utts_list_file_path = args.input_utts_list_file_path
    # output_file_path = args.output_file_path

    utts_pkl_file_path_dict = open_utt2value(input_tmp_decoding_list_file_path)

    utts_list = []
    with open(input_utts_list_file_path, 'r') as f:
        lines = f.readlines()
        for utt_id in lines:
            utts_list.append(utt_id)

    # process
    filtered_utts_pkl_file_path_dict = { utt_id:pikleOpen(pkl_file_path) for utt_id, pkl_file_path in utts_pkl_file_path_dict.items() if utt_id in utts_list }

    count = 0
    for utt_id, utt_info in filtered_utts_pkl_file_path_dict.items():
        count += utt_info.get('feats').get('word_num_disfluency')

    print("There are {} disfluency tokens in scale {}!".format(count, args.scale.upper()))
    print("done!")


