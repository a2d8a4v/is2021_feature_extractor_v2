import re
import argparse

from tqdm import tqdm
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

def nullable_string(val):
    if val.lower() == 'none':
        return None
    return val

def argparse_function():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_text_file_path",
                        default='data/trn/text',
                        type=str)

    parser.add_argument("--input_spk2utt_file_path",
                        default='data/trn/text',
                        type=nullable_string)

    parser.add_argument("--input_spk2momlang_file_path",
                        default='data/trn/momlanguage',
                        type=nullable_string)

    parser.add_argument("--input_cefr_label_file_path",
                        default='CEFR_LABELS_PATH/trn_cefr_scores.txt',
                        type=str)

    parser.add_argument("--output_text_file_path",
                    default='CEFR_LABELS_PATH/trn_cefr_scores.txt',
                    type=str)

    parser.add_argument("--input_json_file_path",
                    default='CEFR_LABELS_PATH/trn_cefr_scores.txt',
                    type=str)

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

    args = parser.parse_args()

    return args

def mapping_cefr2num(scale):
    return mapping_dict[scale]

def xstr(s):
    return '' if s is None else str(s)

## Data preparation
"""
The experiments in our work were performed on responses collected to Cambridge Assessment's [BULATs examination](https://www.cambridgeenglish.org/exams-and-tests/bulats), which is not publicly available. However, you can provide any TSV file (containing a header) of transcriptions containing the following columns:
- text (required): the transcription of the speech (spaces are assumed to signify tokenization)
- score (required): the numerical score assigned to the speech (by default, a scale between 0 - 6 is used to match CEFR proficiency levels)
- pos (optional): Penn Treebank part of speech tags. These should be space-separated and aligned with a token in text (i.e. there should be an identical number of tokens and POS tags)
- deprel (optional): Universal Dependency relation to head/parent token. These should be space-separated and aligned with a token in text (i.e. there should be an identical number of tokens and Universal Dependency relation labels)
- l1 (optional): native language/L1 of the speaker. Our experiments included L1 speakers of Arabic, Dutch, French, Polish, Thai and Vietnamese.
"""

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

    if get_specific_labels is not None:
        count_cefr_labels = 0

    sst = 0
    max_seq_len = 0
    with open(args.output_text_file_path, 'w') as f:
        f.write("{}\t{}\t{}\t{}\n".format('score', 'sst', 'l1', 'text'))
        for utt_or_spk_id, text in utt_text_dict.items():

            if len(text.split()) > max_seq_len:
                max_seq_len = len(text.split())

            if get_specific_labels is not None:
                if get_specific_labels.lower() == utt_cefr_file_path_dict[utt_or_spk_id].lower():
                    f.write("{}\t{}\t{}\t{}\n".format(
                            mapping_cefr2num(
                                utt_cefr_file_path_dict[utt_or_spk_id]
                            ),
                            sst,
                            utt_momlang_dict[utt_or_spk_id],
                            text
                        )
                    )
                    count_cefr_labels+=1
            else:
                f.write("{}\t{}\t{}\t{}\n".format(
                        mapping_cefr2num(
                            utt_cefr_file_path_dict[utt_or_spk_id]
                        ),
                        sst,
                        utt_momlang_dict[utt_or_spk_id],
                        text
                    )
                )

    if max_seq_len > 0:
        print("Max length from all sequences is {}".format(max_seq_len))

    if get_specific_labels is not None:
        print("{} has {} utterances.".format(
                get_specific_labels,
                count_cefr_labels
            )
        )
