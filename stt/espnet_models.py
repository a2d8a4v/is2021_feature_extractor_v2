import torch

from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.asr_align import CTCSegmentation
from rhythm_models import calculate

import os
import json
import time
import shutil
import logging
import soundfile
import numpy as np
from tqdm import tqdm
from utils import (
    run_cmd,
    from_textgrid_to_ctm,
    process_tltchool_gigaspeech_interregnum_tokens
)


'''
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
'''

# NUMBER_OF_PROCESSES determines how many CTC segmentation workers
# are started. Set this higher or lower, depending how fast your
# network can do the inference and how much RAM you have
NUMBER_OF_PROCESSES = 4

def merge_dict(first_dict, second_dict):
    third_dict = {**first_dict, **second_dict}
    return third_dict

def get_stats(numeric_list, prefix=""):
    # number, mean, standard deviation (std), median, mean absolute deviation
    stats_np = np.array(numeric_list)
    number = len(stats_np) 
    
    if number == 0:
        summ = 0.
        mean = 0.
        std = 0.
        median = 0.
        mad = 0.
        maximum = 0.
        minimum = 0.
    else:
        summ = np.float64(np.sum(stats_np))
        mean = np.float64(np.mean(stats_np))
        std = np.float64(np.std(stats_np))
        median = np.float64(np.median(stats_np))
        mad = np.float64(np.sum(np.absolute(stats_np - mean)) / number)
        maximum = np.float64(np.max(stats_np))
        minimum = np.float64(np.min(stats_np))
    
    stats_dict = {  prefix + "number": number, 
                    prefix + "mean": mean, 
                    prefix + "std": std, 
                    prefix + "median": median, 
                    prefix + "mad": mad, 
                    prefix + "summ": summ,
                    prefix + "max": maximum,
                    prefix + "min": minimum
                 }
    return stats_dict
    
    
class SpeechModel(object):
    def __init__(self, tag, is_download=True, cache_dir="./downloads", device='cpu'):
        # STT related
        if is_download:
            d=ModelDownloader(cachedir=cache_dir)
            asr_model = d.download_and_unpack(tag)
            self.speech2text = Speech2Text.from_pretrained(
                **asr_model,
                # Decoding parameters are not included in the model file
                device='cuda',
                maxlenratio=0.0,
                minlenratio=0.0,
                beam_size=20,
                ctc_weight=0.3,
                lm_weight=0.5,
                penalty=0.0,
                nbest=1
            )
            self.aligner = CTCSegmentation(**asr_model, fs=16000, kaldi_style_text=False) # 1 frame equal as 0.025 seconds
        self.mfa_downloaded = False
        self.mfa_acoustic_model = "english_mfa"
        # Fluency related
        self.sil_seconds = 0.145
        self.long_sil_seconds = 0.495
        self.ignored_words = ["@sil", "sil"]
        self.disfluency_phrases = ['you know', 'i mean', 'well', 'like']
        self.disflunecy_words = ["AH", "UM", "UH", "EM", "ER", "ERR"]
        self.disflunecy_words_ets_baseline = ["UM", "UH"]
        self.special_words = ["<UNK>"]
        # Segment related
        self.longest_audio_segments = 320
        self.samples_to_frames_ratio = int(self.aligner.estimate_samples_to_frames_ratio())
        self.partitions_overlap_frames = 30

    # STT-related features
    def recog(self, speech):
        nbests = self.speech2text(speech)
        text, *_ = nbests[0]
        return text
    
    def get_ctm(self, speech, text):

        # alignment (stt)
        segments = self.aligner(speech, text.split())
        segment_info = segments.segments
        text_info = segments.text
        ctm_info = []
        
        for i in range(len(segment_info)):
            start_time, end_time, conf = segment_info[i]
            start_time = round(start_time, 4)
            end_time = round(end_time, 4)
            duration = round(end_time - start_time, 4)
            conf = round(conf, 4)
            ctm_info.append([text_info[i], start_time, duration, round(np.exp(conf),4)])
        
        return ctm_info
    
    def get_textgrid_mfa(self, text, wav_path, tmp_fp, utt_id, word2phn_dict=None):
        """
        :References:
        # @https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps
        # @https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/dictionary_generating.html#g2p-dictionary-generating
        # @https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/example.html#alignment-example
        """

        text = process_tltchool_gigaspeech_interregnum_tokens(text)

        if not self.mfa_downloaded:
            # download Librispeech model
            _ = run_cmd('mfa', 'model', 'download', 'acoustic', self.mfa_acoustic_model)
            # download g2p
            _ = run_cmd('mfa', 'model', 'download', 'g2p', 'english_us_arpa')
            self.mfa_downloaded = True

        # generate a new lexicon
        utt_text_tmp_file = tmp_fp + '/' + utt_id + '_lexicon.tmp.txt'
        utt_text_lex_file = tmp_fp + '/' + utt_id + '_lexicon.txt'
        if not os.path.isdir(tmp_fp):
            os.makedirs(tmp_fp)
        if word2phn_dict is not None:
            with open(utt_text_lex_file, 'w') as fp:
                for token in text.split():
                    phn_text = " ".join(word2phn_dict[token.lower()])
                    fp.write("{} {}\n".format(token, phn_text))
        else:
            with open(utt_text_tmp_file, 'w') as fp:
                for token in text.split():
                    fp.write("{}\n".format(token))
            _ = run_cmd('mfa', 'g2p', 'english_us_arpa', utt_text_tmp_file, utt_text_lex_file)
            os.remove(utt_text_tmp_file)
        
        # prepare data
        src = wav_path
        dst = tmp_fp + '/' + utt_id + '_input' + '/'
        ali = tmp_fp + '/' + utt_id + '_aligned'
        if not os.path.isdir(dst):
            os.mkdir(dst)
        shutil.copy(src, dst)
        with open(dst + '/' + utt_id + '.lab', 'w') as fp:
            fp.write("{}\n".format(text))

        # Force alignment
        # add --clean to avoid sql searching in mfa database
        _ = run_cmd('mfa', 'align', '--clean', dst, utt_text_lex_file, self.mfa_acoustic_model, ali)
        del _
        textgrid_file_path = ali + '/' + utt_id + '.TextGrid'

        return textgrid_file_path

    def get_ctm_from_textgrid(self, textgrid_file_path):

        ctm_info, _ = from_textgrid_to_ctm(textgrid_file_path)

        return ctm_info

    # if have ctm file already
    def load_ctm(self, ctm_dict, utt_id):
        ctm_info = []
        # channel_num start_time phone_dur phone_id
        for info in ctm_dict[utt_id]:
            ctm_info.append([str(info[3]), round(float(info[1]), 4), round(float(info[2]), 4), 1.0])

        return ctm_info

    # Fluency-related features
    def sil_feats(self, ctm_info, total_duration):
        '''
        TODO
        {sil, long_sil}_rate_response: num_silences / response_duration
        {sil, long_sil}sil_rate_word: num_silences / num_of_words
        '''
        # > 0.145
        sil_list = []
        # > 0.495
        long_sil_list = []

        response_duration = total_duration
        if len(ctm_info) > 0:
            # response time
            start_time = ctm_info[0][1]
            # start_time + duration
            end_time = ctm_info[-1][1] + ctm_info[-1][2]
            response_duration = end_time - start_time

        # BUG: if less than 2 words, we could not have the ability to count the silences
        if len(ctm_info) > 2:
            word, start_time, duration, conf = ctm_info[0]
            prev_end_time = start_time + duration
            
            for word, start_time, duration, conf in ctm_info[1:]:
                interval_word_duration = start_time - prev_end_time
                
                if interval_word_duration > self.sil_seconds:
                    sil_list.append(interval_word_duration)
                
                if interval_word_duration > self.long_sil_seconds:
                    long_sil_list.append(interval_word_duration)
                
                prev_end_time = start_time + duration

        sil_stats = get_stats(sil_list, prefix="sil_")
        long_sil_stats = get_stats(long_sil_list, prefix="long_sil_")
        '''
        {sil, long_sil}_rate1: num_silences / response_duration
        {sil, long_sil}_rate2: num_silences / num_words
        '''
        num_sils = len(sil_list)
        num_long_sils = len(long_sil_list)
        num_words = len(ctm_info)
        
        sil_stats["sil_rate1"] = num_sils / response_duration
        
        if num_words > 0:
            sil_stats["sil_rate2"] = num_sils / num_words
        else:
            sil_stats["sil_rate2"] = 0
        
        long_sil_stats["long_sil_rate1"] = num_long_sils / response_duration 
        
        if num_words > 0:
            long_sil_stats["long_sil_rate2"] = num_long_sils / num_words
        else:
            long_sil_stats["long_sil_rate2"] = 0
        
        sil_dict = merge_dict(sil_stats, long_sil_stats)
        
        return sil_dict, response_duration
    
    def word_feats(self, ctm_info, total_duration):
        '''
        TODO:
        number of repeated words
        '''
        word_count = len(ctm_info)
        word_duration_list = []
        word_conf_list = []
        num_disfluency = 0
        num_disfluency_baseline = 0
        dict_disfluency_phrases = {}
        utt = []
        num_repeat = 0
        prev_words = []
        word_count_dict = defaultdict(int)

        response_duration = total_duration
        if len(ctm_info) > 0:
            # response time
            start_time = ctm_info[0][1]
            # start_time + duration
            end_time = ctm_info[-1][1] + ctm_info[-1][2]
            response_duration = end_time - start_time    

        for word, start_time, duration, conf in ctm_info:
            word_duration_list.append(duration)
            word_conf_list.append(conf)

            if word in self.disflunecy_words:
                num_disfluency += 1

            if word in self.disflunecy_words_ets_baseline:
                num_disfluency_baseline += 1
            
            utt.append(word)

            word_count_dict[word] += 1

            if word in prev_words:
                num_repeat += 1
            
            prev_words = [word]

        utt = " ".join(utt)
        
        for term in self.disfluency_phrases:
            dict_disfluency_phrases.setdefault(term, utt.lower().count(term.lower()))

        # strat_time and duration of last word
        # word in articlulation time
        word_freq = word_count / response_duration # words per seconds
        word_distinct = len(list(word_count_dict.keys()))
        word_duration_stats = get_stats(word_duration_list, prefix = "word_duration_")
        word_conf_stats = get_stats(word_conf_list, prefix="word_conf_")
        
        word_basic_dict = {   
                        "word_count": word_count,
                        "word_freq": word_freq,
                        "word_distinct": word_distinct,
                        "word_num_repeat": num_repeat,
                        "word_num_disfluency": num_disfluency,
                        "word_num_disfluency_baseline": num_disfluency_baseline,
                        "phrase_num_disfluency_phrases": dict_disfluency_phrases
                    }
        word_stats_dict = merge_dict(word_duration_stats, word_conf_stats)
        word_dict = merge_dict(word_basic_dict, word_stats_dict)
        
        return word_dict

    def rhythm_feats(self, ctm_info, word2phn_dict):
        return calculate(ctm_info, word2phn_dict)

    def get_phone_ctm(self, ctm_info, word2phn_dict):
        # use g2p model
        phone_ctm_info = []
        phone_text = []
        
        for word, start_time, duration, conf in ctm_info:
            # phones = self.g2p(word)
            phones = word2phn_dict[word]
            duration /= len(phones)
            
            for phone in phones:
                phone_ctm_info.append([phone, start_time, duration, conf])
                start_time += duration
                phone_text.append(phone)
        
        phone_text = " ".join(phone_text)
        
        return phone_ctm_info, phone_text

    def phone_feats(self, ctm_info, total_duration):
        phone_count_dict = defaultdict(int)
        phone_duration_list = []
        phone_conf_list = []
         
        response_duration = total_duration
        if len(ctm_info) > 0:
            # response time
            start_time = ctm_info[0][1]
            # start_time + duration
            end_time = ctm_info[-1][1] + ctm_info[-1][2]
            response_duration = end_time - start_time        
        
        for phone, start_time, duration, conf in ctm_info:
            phone_duration_list.append(duration)
            phone_conf_list.append(conf)
            phone_count_dict[phone] += 1  
            
        # strat_time and duration of last phone
        # word in articlulation time
        phone_count = sum(list(phone_count_dict.values()))
        phone_freq = phone_count / response_duration
        phone_duration_stats = get_stats(phone_duration_list, prefix = "phone_duration_")
        phone_conf_stats = get_stats(phone_conf_list, prefix="phone_conf_")
        
        phone_basic_dict = { 
                            "phone_count": phone_count,
                            "phone_freq": phone_freq,
                           }
        
        phone_stats_dict = merge_dict(phone_duration_stats, phone_conf_stats)
        phone_dict = merge_dict(phone_basic_dict, phone_stats_dict)
        
        return phone_dict, response_duration