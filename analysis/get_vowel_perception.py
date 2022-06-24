import argparse
import textgrid
import re
import os
import sys
import math
import json
from tqdm import tqdm
import numpy as np
from espnet.utils.cli_utils import strtobool
import matplotlib as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "local.apl.v3/utils"))) # Remember to add this line to avoid "module no exist" error
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "local.apl.v3/stt")))

from g2p_model import G2PModel
from utilities import (
    opendict,
    openlist,
    open_utt2pkl,
    pickleStore,
    getLeft,
    getRight
)
from objects import (
    Which,
    Interval
)

# functions
def make_phn_ctm(word_ctm, w2p_dict):

    phone_ctm_info = []
    
    for word, start_time, duration, conf in word_ctm:

        phones = w2p_dict[word.lower()]
        duration /= len(phones)
        
        for phone in phones:
            phone_ctm_info.append([phone, start_time, duration, conf])
            start_time += duration

    return phone_ctm_info

def process_useless_tokens_phoneme(phone_ctm):
    rtn = []
    for phn_token, start, end, gop_score in phone_ctm:
        rtn.append([str(phn_token.split('_')[0]), start, end, gop_score])
    return rtn

if __name__ == '__main__':

    # parsers
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_file', default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/cefr_train_tr/gigaspeech_20220525_prompt/all.json", type=str)
    parser.add_argument('--input_dict', default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/local/dict/lexicon.txt", type=str)
    parser.add_argument('--lexicon_file_path', default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/local/dict/lexicon.txt", type=str)
    parser.add_argument('--output_file_path', default="./test.pkl", type=str)
    parser.add_argument('--phn_from_data', default=True, type=strtobool)
    args = parser.parse_args()

    # variables
    vowel_formant_dict = {}
    input_file = args.input_file
    input_dict = args.input_dict
    phn_from_data = args.phn_from_data
    output_file_path = args.output_file_path
    lexicon_file_path = args.lexicon_file_path

    # read file
    utt_infos = open_utt2pkl(input_file)

    # initialize model
    g2p_model = G2PModel(lexicon_file_path)
    _w = Which()

    # initialize w2p dictionary
    w2p_dict = opendict(input_dict)

    # get F1 and F2 in each utterance
    for utt_id, utt_info in tqdm(utt_infos.items()):

        # text and word2phone dictionary
        text = utt_info.get('stt')
        w2p_dict = g2p_model.g2p(text, w2p_dict)

        # from F1 to F5, but we only need F1 and F2
        ctm = utt_info.get('ctm')
        if phn_from_data:
            phn_ctm = process_useless_tokens_phoneme(
                utt_info.get('phone_ctm')
            )
        else:
            # evenly divide the word duration
            phn_ctm = make_phn_ctm(ctm, w2p_dict)
        total_duration = utt_info.get('feats').get('total_duration')
        formants = np.array(utt_info.get('feats').get('formant'))

        nLabel = []
        # first, we need to get vowels from phoneme-level ctm
        for phn_id, start, duration, conf in phn_ctm:
            
            phn_id = re.sub("\d+", '', phn_id.split('_')[0])

            # skip silent tokens
            if phn_id.lower() in ['sil','@sil','spn']:
                continue

            # duaration
            end = start + duration

            # is consonant or vowel
            _n = Interval()
            _n.set_start(start)
            _n.set_end(end)
            _n.set_label(phn_id)
            _n.set_type(_w._is(phn_id))
            nLabel.append(_n)

        vowel_dict = {}
        # second, make a dict to count each vowel
        for i in nLabel:
            if i.get_type() == _w.get_v():
                vowel_dict.setdefault(i.get_label(), []).append([i.get_start(), i.get_end()])
        
        # get formant array for each vowel and record that
        for vowel, vowel_info in vowel_dict.items():
            for start, end in vowel_info:
                # TODO: fix the interval issue
                _interval = [
                    getLeft(start, total_duration, formants),
                    getRight(end, total_duration, formants)
                ]
                
                formant_array = formants[_interval]
                formant_array_f1 = formant_array[:,1]
                formant_array_f2 = formant_array[:,2]
                formant_mean_f1  = np.mean(formant_array_f1).item()
                formant_mean_f2  = np.mean(formant_array_f2).item()
                vowel_formant_dict.setdefault(vowel, []).append([formant_mean_f1, formant_mean_f2])

    # save vowel_formant_dict
    pickleStore(vowel_formant_dict, output_file_path)
