import os
import numpy as np
import re

class GOP(object):
    def __init__(self, conf={}, port=8899):
        
        self.outdir = conf.get("out-dir", None)
        self.dir_name = conf.get("dir-name", "exp/nnet3_chain")
        self.phn_dur_fn = conf.get("phn-dur", None)
        self.lexicon = conf.get("lexicon", "IPA")
        self.lang_dir = conf.get("lang-dir", None)
        self.skip_word_unk = conf.get('skip-word-unk', True)
        self.silence_phones = conf.get("silence-phones", "0:1:2:3:4:5")
        self.special_tokens = ['<unk>', '<UNK>']
        self.s2t = conf.get('s2t', False)
        self.silence_tokens_list = self._open_silence_tokens(
                os.path.abspath(
                    os.path.join(self.dir_name, conf.get('sil-phone-path'))
                )
            ) if conf.get('sil-phone-path', None) is not None else []
        self.disfluency_special_tokens = self._open_disfluency_tokens(
                os.path.abspath(
                    os.path.join(self.dir_name, conf.get('nonsil-phone-path'))
                )
            ) if conf.get('nonsil-phone-path', None) is not None else []
        if self.s2t:
            self.s2t_skip_word_unk = conf.get('s2t-skip-word-unk', True)
            self.disfluency_special_tokens = self._open_disfluency_tokens(
                    os.path.abspath(
                        os.path.join(self.dir_name, conf.get('nonsil-phone-path'))
                    )
                ) if conf.get('nonsil-phone-path', None) is not None else []

        # load words.txt for oov handler
        vocab_file = open(self.dir_name + "/" + self.lang_dir + "/words.txt", "r")
        self.vocab_list = { word.split()[0]: index for index, word in enumerate(vocab_file.readlines()) }
        self.oov_list = []
        vocab_file.close()
         
        if self.phn_dur_fn:
            self.phns_dur_info = np.load(self.dir_name + "/" + self.phn_dur_fn, allow_pickle=True).item()
        else:
            self.phns_dur_info = None

        # phone table (phn, id)
        self.phone_table = {}

        with open(self.dir_name + "/" + self.lang_dir + "/phones.txt", "r") as phn_fn:
            for line in phn_fn.readlines():
                phn, phn_id = line.split("\n")[0].split()
                self.phone_table[phn] = phn_id
        # silence phones
        self.silence_phones = self.silence_phones.split(":")
    
    @classmethod
    def _open_disfluency_tokens(self, file_path):
        filter_list = ['@', '<']
        disfluency_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                for filter in filter_list:
                    token = line.split()[0]
                    if filter == token[0]:
                        disfluency_list.append(token)
        return disfluency_list

    @classmethod
    def _open_silence_tokens(self, file_path):
        # filter_list = ['sil', 'spn', 'hes', 'unk']
        silence_set = set()
        with open(file_path, 'r') as f:
            for line in f.readlines():
                token = line.strip()
                # if token in filter_list:
                silence_set.add(token)
                token = line.strip().split('_')[0]
                # if token in filter_list:
                silence_set.add(token)
        return list(silence_set)

    def sigmoid(self, x, alpha = 1, c = 0):
        # score function
        return 1. / (1 + np.exp(-1 * alpha * (x-c)))
    
    def zs_conversion(self, zs):
        zs_alpha = 30.
        zs_ub = 5.
        zs = abs(zs)
        zs_max = zs_ub - zs_alpha * zs_ub / (100. + zs_alpha)
        if zs >= zs_max:
            zs = zs_max
        fluency_score = (100. + zs_alpha) - (zs_alpha * zs_ub) / (zs_ub - zs)
        return fluency_score

    def process_GOP(self, gop_transcript):
        # GOP: [[Word1, [['Phone1', score], ['Phone2': score], ['average': average]]],
        #      ,[Word2, [['Phone2', score], ...]]]
        # lexicon (IPA or ARPAbet)
        lexicon = self.lexicon
        # phones_duration (default==None)
        phns_dur_info = self.phns_dur_info
        gop_list = list(gop_transcript.split())
        prompt_list = list(self.prompt.split())
        gop_word_list = {"GOP": [], "Sound": [], "Stress": [], "Fluency": []}
        j = 0
        for i in range(len(prompt_list)):
            word = prompt_list[i]
            if self.skip_word_unk and not self.s2t:
                if word in self.disfluency_special_tokens:
                    continue
                if word in self.special_tokens:
                    continue
            
            if self.s2t:
                if self.s2t_skip_word_unk:
                    if word in self.disfluency_special_tokens:
                        continue
                    if word in self.special_tokens:
                        continue

            # print(word)

            phone_list = []
            stress_list = []
            sound_list = []
            fluency_list = []
            gop_summation = 0.
            fluency_summation = 0.
            # NOTE: We'd like to set word-position-dependency=true now (That's not necessary)
            while True:
                # get the GOP_msg of server
                # try:
                phone = gop_list[j]
                    # print("{} - {} - {}".format(word, j , phone))
                # except:
                #     assert 1==2, "{} - {} - {} - {}".format(j, phone, gop_list, prompt_list)
                pred_phone = gop_list[j+1]
                duration = float(gop_list[j+2])

                # -2. skip the silence tokens
                if re.sub("\d+", '', phone.lower().split('_')[0]) in self.silence_tokens_list:
                    j+=4
                    continue

                # -1. silence detection (silence or spoken noise)
                is_silence = (self.phone_table[pred_phone] in self.silence_phones)
                # 0. GOP score
                
                gop_score = self.sigmoid(float(gop_list[j+3])) * 200
                if is_silence:
                    gop_score = gop_score * 0.2
                 
                phone_list.append([phone, gop_score])
                gop_summation += gop_score
                # 1. Stress
                # stress maker for IPA (') and ARPAbet (0, 1, 2, 3)
                if lexicon == "IPA":
                    if ("'" in pred_phone):
                        stress_list.append([1])
                    else:
                        stress_list.append([0])
                elif lexicon == "ARPAbet":
                    # 0, 1, 2, and 3 is stress label in ARPAbet
                    if re.search("[1-3]", pred_phone) != None:
                        stress_list.append([True])
                    else:
                        stress_list.append([False])
                else:
                    # not implement yet.
                    stress_list.append([False])
                # 2. Sounds like
                sound_list.append([pred_phone])
                # 3. Fluency
                # information of the phone duration
                # removing position marker (e.g., _E, _S, _I, and _B)
                pure_phone = phone.split("_")[0]
                # get infomation from phone duration (statistical information)
                if phns_dur_info and not is_silence:
                    duration_info = phns_dur_info[pure_phone]
                    # compute z-score
                    z_score = (duration - duration_info["Mean"]) / duration_info["STD"]
                    fluency_score = self.zs_conversion(z_score)
                else:
                    fluency_score = 0
                fluency_list.append([phone, fluency_score])
                fluency_summation += abs(fluency_score)
                j += 4
                if(phone[-1] != "E" and phone[-1] != "S"):
                    continue
                else:
                    break
            length = len(phone_list)
            # average of the gop
            gop_avg = gop_summation / length
            phone_list.append(['average', gop_avg])
            # average of the fluency
            fluency_avg = fluency_summation / length
            fluency_list.append(['average', fluency_avg])
            
            gop_word_list["GOP"].append([word, phone_list])
            gop_word_list["Stress"].append([word, stress_list])
            gop_word_list["Sound"].append([word, sound_list])
            gop_word_list["Fluency"].append([word, fluency_list])
        return gop_word_list
    
    def set_prompt(self, prompt):
        self.prompt = prompt
