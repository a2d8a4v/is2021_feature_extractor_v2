import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "local.apl.v3/utils"))) # Remember to add this line to avoid "module no exist" error
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "local.apl.v3/stt")))

from espnet.utils.cli_utils import strtobool
from utilities import (
    jsonLoad,
    opendict,
    open_text,
    open_utt2value,
    remove_partial_words_call,
    remove_tltschool_interregnum_tokens
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

native_language_labels = ['X', 'italiano', 'japanese']

def nullable_string(val):
    return None if val.lower() == 'none' else val

def argparse_function():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_text_file_path",
                        default='data/trn/text',
                        type=str)

    parser.add_argument("--input_json_file_path",
                    default='CEFR_LABELS_PATH/trn_cefr_scores.txt',
                    type=str)

    parser.add_argument("--output_text_file_path",
                    default='CEFR_LABELS_PATH/trn_cefr_scores.txt',
                    type=str)

    parser.add_argument("--input_spk2momlang_file_path",
                        default='data/trn/momlanguage',
                        type=nullable_string)

    parser.add_argument("--input_cefr_label_file_path",
                        default='CEFR_LABELS_PATH/trn_cefr_scores.txt',
                        type=str)

    parser.add_argument("--input_spk2utt_file_path",
                        default='data/trn/text',
                        type=nullable_string)

    parser.add_argument("--s2t",
                    default=False,
                    type=strtobool)

    parser.add_argument("--get_specific_labels",
                    default=None,
                    type=nullable_string)

    parser.add_argument("--remove_filled_pauses",
                    default=False,
                    type=strtobool)

    parser.add_argument("--remove_partial_words",
                    default=False,
                    type=strtobool)

    parser.add_argument("--combine_same_speakerids",
                    default=False,
                    type=strtobool)

    parser.add_argument("--sort_by",
                    choices=['key', 'value', 'none'],
                    default=None,
                    type=nullable_string)

    args = parser.parse_args()

    return args

def mapping_cefr2num(scale):
    return mapping_dict[scale]

def xstr(s):
    return '' if s is None else str(s)


if __name__ == '__main__':

    # argparse
    args = argparse_function()

    # variables
    s2t = args.s2t
    combine_same_speakerids = args.combine_same_speakerids
    get_specific_labels = args.get_specific_labels
    remove_filled_pauses = args.remove_filled_pauses
    remove_partial_words = args.remove_partial_words
    input_spk2utt_file_path = args.input_spk2utt_file_path
    input_json_file_path = args.input_json_file_path

    if get_specific_labels is not None:
        assert get_specific_labels in mapping_dict.keys(), "get_specific_labels was given out-of-domain label!"

    if not s2t:
        utt_text_dict = open_text(args.input_text_file_path)
    else:
        utt_json_data = jsonLoad(input_json_file_path)['utts']
        utt_text_dict = { utt_id:utt_info.get('stt') for utt_id, utt_info in utt_json_data.items() if xstr(utt_info.get('stt')).strip() }

    utt_cefr_file_path_dict = open_utt2value(args.input_cefr_label_file_path)
    utt_momlang_dict = open_utt2value(args.input_spk2momlang_file_path)

    if remove_filled_pauses:
        utt_text_dict = { utt_id:remove_tltschool_interregnum_tokens(texts) for utt_id, texts in utt_text_dict.items() }

    if remove_partial_words:
        utt_text_dict = { utt_id:remove_partial_words_call(texts) for utt_id, texts in utt_text_dict.items() }

    if combine_same_speakerids:
        assert input_spk2utt_file_path is not None, 'You need to point a specific path for input_spk2utt_file_path'

        spk2utt_dict = opendict(input_spk2utt_file_path)
        new_utt_text_dict = {}
        new_utt_cefr_file_path_dict = {}
        new_utt_momlang_dict = {}
        for spk_id, utts_list in spk2utt_dict.items():
            text_list = []
            for utt_id in utts_list:
                if utt_id in utt_text_dict: # BUG: some recognized result has empty result!
                    text_list.extend(utt_text_dict[utt_id].split())
                new_utt_cefr_file_path_dict.setdefault(spk_id, utt_cefr_file_path_dict[utt_id])
                new_utt_momlang_dict.setdefault(spk_id, utt_momlang_dict[utt_id])
            if text_list: # BUG: some recognized result has empty result!
                new_utt_text_dict[spk_id] = " ".join(text_list)
        utt_text_dict = new_utt_text_dict
        utt_cefr_file_path_dict = new_utt_cefr_file_path_dict
        utt_momlang_dict = new_utt_momlang_dict

    # analysis words
    word_cumulation_dict = {}

    for utt_or_spk_id, text in utt_text_dict.items():
        token_list = text.split()
        for token in token_list:
            if get_specific_labels is not None:
                if get_specific_labels.lower() == utt_cefr_file_path_dict[utt_or_spk_id].lower():
                    word_cumulation_dict.setdefault(token, []).append(utt_or_spk_id)
            else:
                word_cumulation_dict.setdefault(token, []).append(utt_or_spk_id)

    # sort
    word_cumulation_dict = { token:len(utt_or_spk_id_count_list) for token, utt_or_spk_id_count_list in word_cumulation_dict.items() }
    if args.sort_by == 'key':
        word_cumulation_dict = sorted(word_cumulation_dict)
    elif args.sort_by == 'value':
        word_cumulation_dict = dict(sorted(word_cumulation_dict.items(), reverse=True, key=lambda item: item[1]))

    # save
    with open(args.output_text_file_path, 'w') as f:
        for token, token_count in word_cumulation_dict.items():
            f.write("{} {}\n".format(token, token_count))

    print('Output path: {}'.format(args.output_text_file_path))