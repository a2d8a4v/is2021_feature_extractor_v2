import argparse
import textgrid
import os
import sys
import math
import json
import numpy as np
from espnet.utils.cli_utils import strtobool
import matplotlib as plt

sys.path.insert(0,os.path.abspath(os.path.join(os.getcwd(), "local.apl.v3/stt"))) # Remember to add this line to avoid "module no exist" error

from g2p_model import G2PModel
from utils import (
    opendict,
    pickleStore
)

# functions
def jsonLoad(scores_json):
    with open(scores_json) as json_file:
        return json.load(json_file)

def getLeft(timepoint, total_duration, formants):
    percent = timepoint / total_duration
    len_formant = formants.shape[0]
    time_list = formants[:,0]

    # get the ambiguous frame position
    position = math.ceil(len_formant*percent)

    # check the timestamp of formant is bigger than timepoint
    # we hypothesis two continuous timestamp should have similar frequency
    formant_timestamp = time_list[position].item()
    while True:
        if (formant_timestamp == timepoint) or (formant_timestamp < timepoint and time_list[position+1].item() > timepoint):
            return position
        elif formant_timestamp > timepoint:
            position = position-1
            formant_timestamp = time_list[position].item()
        elif formant_timestamp < timepoint and time_list[position+1].item() < timepoint:
            # if the ambiguous position has deviation
            position = position+1
            formant_timestamp = time_list[position].item()
    return

def getRight(timepoint, total_duration, formants):
    percent = timepoint / total_duration
    len_formant = formants.shape[0]
    time_list = formants[:,0]

    # get the ambiguous frame position
    position = math.floor(len_formant*percent)

    # check the timestamp of formant is bigger than timepoint
    # we hypothesis two continuous timestamp should have similar frequency
    formant_timestamp = time_list[position].item()
    while True:
        if (formant_timestamp == timepoint) or (formant_timestamp < timepoint and time_list[position+1].item() > timepoint):
            return position
        elif formant_timestamp < timepoint:
            position = position+1
            formant_timestamp = time_list[position].item()
        elif formant_timestamp > timepoint and time_list[position+1].item() > timepoint:
            # if the ambiguous position has deviation
            position = position-1
            formant_timestamp = time_list[position].item()
    return

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

# class
class Which(object):

    # @https://hackage.haskell.org/package/hsc3-lang-0.15/docs/src/Sound-SC3-Lang-Data-CMUdict.html
    def __init__(self):
        self.consonants = ['l', 'zh', 's', 'z', 'ng', 'g', 'k', 'th', 'd', 'dh', 'w', 'p', 'n', 't', 'r', 'sh', 'ch', 'hh', 'b', 'jh', 'f', 'm', 'v']
        self.vowels = ['ah', 'aa', 'ih', 'aw', 'w', 'axr', 'ow', 'ao', 'y', 'eh', 'ay', 'uh', 'q', 'ey', 'ae', 'iy', 'oy', 'uw', 'ax', 'er']
        self.consonant = "C"
        self.vowel = "V"
        self.other = "O"

    def _is(self, ph):
        ph = ph.lower()
        if ph in self.vowels:
            return self.vowel
        elif ph in self.consonants:
            return self.consonant
        else:
            return self.other

    def get_v(self):
        return self.vowel

    def get_c(self):
        return self.consonant

    def get_o(self):
        return self.other

class Interval(object):

    def __init__(self):
        self.start = None
        self.end = None
        self.duration = None
        self.label = None
        self.type = None
    
    def set_start(self, s):
        self.start = s
        if self.end is not None:
            self.duration = self.end - self.start

    def set_end(self, e):
        self.end = e
        if self.start is not None:
            self.duration = self.end - self.start

    def set_label(self, l):
        self.label = l

    def set_type(self, t):
        self.type = t

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_label(self):
        return self.label

    def get_dur(self):
        return self.duration

    def get_type(self):
        return self.type

if __name__ == '__main__':

    # parsers
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--action',
                        default='collect_feats',
                        const='collect_feats',
                        nargs='?',
                        choices=['collect_feats', 'plot_perception'])
    parser.add_argument('--input_json', default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/cefr_train_tr/gigaspeech_20220525_prompt/all.json", type=str)
    parser.add_argument('--input_dict', default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/local/dict/lexicon.txt", type=str)
    parser.add_argument('--lexicon_file_path', default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/local/dict/lexicon.txt", type=str)
    parser.add_argument('--output_file_path', default="./test.pkl", type=str)
    parser.add_argument('--phn_from_data', default=True, type=strtobool)
    args = parser.parse_args()

    if args.action == 'collect_feats':

        # variables
        vowel_formant_dict = {}
        input_json = args.input_json
        input_dict = args.input_dict
        phn_from_data = args.phn_from_data
        output_file_path = args.output_file_path
        lexicon_file_path = args.lexicon_file_path

        # initialize model
        g2p_model = G2PModel(lexicon_file_path)
        _w = Which()

        # initialize w2p dictionary
        w2p_dict = opendict(input_dict)

        # read file
        utt_infos = jsonLoad(input_json)['utts']

        # get F1 and F2 in each utterance
        for utt_id, utt_info in utt_infos.items():

            # text and word2phone dictionary
            text = utt_info.get('stt')
            w2p_dict = g2p_model.g2p(text, w2p_dict)

            # from F1 to F5, but we only need F1 and F2
            ctm = utt_info.get('ctm')
            if phn_from_data:
                phn_ctm = process_useless_tokens_phoneme(
                    utt_info.get('phn_ctm')
                )
            else:
                # evenly divide the word duration
                phn_ctm = make_phn_ctm(ctm, w2p_dict)
            total_duration = utt_info.get('feats').get('total_duration')
            formants = np.array(utt_info.get('feats').get('formant'))

            nLabel = []
            # first, we need to get vowels from phoneme-level ctm
            for phn_id, start, duration, conf in phn_ctm:
                
                # skip silent tokens
                if phn_id.lower() in ['sil','@sil']:
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

    if args.action == 'plot_perception':

        # variables
        input_json = args.input_json
        vowel_formant_dict = pikleOpen(input_json)

        # draw a plot: vowel perception

        # compute the radius of circle
