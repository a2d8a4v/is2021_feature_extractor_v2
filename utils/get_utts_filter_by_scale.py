import argparse
from utilities import (
    open_utt2value
)

mapping_dict = {
    0: 0,
    'a1': 1,
    'a1+': 2,
    'a2': 3,
    'a2+': 4,
    'b1': 5,
    'b1+': 6,
    'b2': 7
}

def nullable_string(val):
    if val.lower() == 'none':
        return None
    return val


if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_tmp_decoding_list_file_path",
                        default="data/train/ctm",
                        type=str)

    parser.add_argument("--input_scale_file_path",
                        default="-",
                        type=str)                      

    parser.add_argument("--filter_scale",
                        default=None,
                        type=nullable_string) 

    parser.add_argument("--output_file_path",
                        default="-",
                        type=str) 

    args = parser.parse_args()

    # variables
    input_tmp_decoding_list_file_path = args.input_tmp_decoding_list_file_path
    input_scale_file_path = args.input_scale_file_path
    output_file_path = args.output_file_path

    filter_scale = args.filter_scale
    assert filter_scale.lower() in list(mapping_dict.keys()), 'out of domain!'

    utt_scale_dict = open_utt2value(input_scale_file_path)
    utts_list = []

    # process
    for utt_id, scale in utt_scale_dict.items():
        if scale.lower() == filter_scale.lower():
            utts_list.append(utt_id)

    with open(output_file_path, 'w') as f:
        for utt_id in utts_list:
            f.write("{}\n".format(utt_id))

    print("done!")
