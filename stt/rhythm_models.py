import math
import os
import re
import numpy as np


# References
# @https://github.com/Shahabks/Thought-Speech-analysis-
# @https://chrisearch.wordpress.com/2017/03/11/speech-recognition-using-kaldi-extending-and-using-the-aspire-model/
# @https://stackoverflow.com/questions/37608115/how-can-i-add-new-words-or-vocabulary-into-kaldi-platform


# Class
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


# Functions
def rPVI(whi, num, arr, type=None):
    if type == whi.get_v():
        arr = [i for i in arr if i.get_type() == whi.get_v()]
    elif type == whi.get_c():
        arr = [i for i in arr if i.get_type() == whi.get_c()]
    return (1/(num-1))*sum([abs(arr[i+1].get_dur() - arr[i].get_dur()) for i in range(0, len(arr)-1)])


def nPVI(whi, num, arr, type=None):
    if type == whi.get_v():
        arr = [i for i in arr if i.get_type() == whi.get_v()]
    elif type == whi.get_c():
        arr = [i for i in arr if i.get_type() == whi.get_c()]
    return (1/(num-1))*sum([abs(arr[i+1].get_dur() - arr[i].get_dur())/((arr[i+1].get_dur() + arr[i].get_dur())/2) for i in range(0, len(arr)-1)])


def CCI(whi, num, arr, type=None):
    if type == whi.get_v():
        arr = [i for i in arr if i.get_type() == whi.get_v()]
    elif type == whi.get_c():
        arr = [i for i in arr if i.get_type() == whi.get_c()]
    return (1/(num-1))*sum([abs(arr[i+1].get_dur()/len(arr[i+1].get_label()) - arr[i].get_dur()/len(arr[i].get_label())) for i in range(0, len(arr)-1)])


# def RVDR(whi, num, arr, type=None):
#     """
#     Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8682957
#     Title: REFERENTIAL VOWEL DURATION RATIO AS A FEATURE FOR AUTOMATIC ASSESSMENT OF L2 WORD PROSODY
#     """
#     # TODO(YANN): Add reference arr from l1 speaker with the same content
#     if type == whi.get_v():
#         arr = [i for i in arr if i.get_type() == whi.get_v()]
#     elif type == whi.get_c():
#         arr = [i for i in arr if i.get_type() == whi.get_c()]

#     return sum(
#                 [
#                     abs(np.log(arr[i+1].get_dur()/arr[i].get_dur()).item())*
#                     np.log(arr[i+1].get_dur()/arr[i].get_dur()).item()*
#                     np.sign(np.log(arr[i+1].get_dur()/arr[i].get_dur())).item()
#                      for i in range(0, len(arr)-1)
#                 ]
#             ) / 
#             sum(
#                 [ abs(np.log(arr[i+1].get_dur()/arr[i].get_dur()).item()) for i in range(0, len(arr)-1)]
#             )


# def WVDR():
#     """
#     Paper: https://www.isca-speech.org/archive_v0/SpeechProsody_2020/pdfs/62.pdf
#     Title: Tree-based Clustering of Vowel Duration Ratio Toward Dictionary-based Automatic Assessment of Prosody in L2 English Word Utterances
#     """
#     # TODO(YANN): Build a tree
#     if type == whi.get_v():
#         arr = [i for i in arr if i.get_type() == whi.get_v()]
#     elif type == whi.get_c():
#         arr = [i for i in arr if i.get_type() == whi.get_c()]

#     return sum(
#                 [
#                     abs(np.log(arr[i+1].get_dur()/arr[i].get_dur()).item())*
#                     np.log(arr[i+1].get_dur()/arr[i].get_dur()).item()*
#                     np.sign(np.log(arr[i+1].get_dur()/arr[i].get_dur())).item()
#                      for i in range(0, len(arr)-1)
#                 ]
#             ) / 
#             sum(
#                 [ abs(np.log(arr[i+1].get_dur()/arr[i].get_dur()).item()) for i in range(0, len(arr)-1)]
#             )



# Get information from each file
def calculate(phn_ctm_info):

    # Intialize Which object
    _w = Which()
    
    rtn = {
        "status": 0,
        "num_nucleus": 0,
        "num_consonants": 0,
        "v_to_c_ratio": 0,
        "sum_dur_sec": 0,
        "phones_num": 0,
        "v_sum_dur_sec": 0,
        "v_num": 0,
        "c_sum_dur_sec": 0,
        "c_num": 0,
        "mean_v_dur": 0,
        "mean_c_dur": 0,
        "mean_p_dur": 0,
        "p_sum_variance": 0,
        "v_std_dev": 0,
        "c_std_dev": 0,
        "p_std_dev": 0,
        "varco_P": 0,
        "varco_V": 0,
        "varco_C": 0,
        "percent_v": 0,
        "speech_rate": 0,
        "npvi": 0,
        "npvi_v": 0,
        "npvi_c": 0,
        "rpvi": 0,
        "rpvi_v": 0,
        "rpvi_c": 0,
        "cci": 0,
        "cci_v": 0,
        "cci_c": 0
    }

    nLabel = []

    ## Get alignment information from ctm file
    # phn_ctm_info: text, start_time, duration, word confidence
    for text, start_time, duration, _ in phn_ctm_info:

        phn = re.sub("\d+", '', text.split('_')[0]).lower()

        start_time = start_time
        end_time   = start_time + duration

        _n = Interval()
        _n.set_start(start_time)
        _n.set_end(end_time)
        _n.set_label(phn)
        _n.set_type(_w._is(phn))
        nLabel.append(_n)

    ## Calculate the number of vowels
    # Total number of vowels is counted to estimate average speech rate (i.e. number of syllables per second)
    num_nucleus = 0
    num_consonants = 0
    for i in nLabel:
        if i.get_type() == _w.get_v():
            num_nucleus += 1
        elif i.get_type() == _w.get_c():
            num_consonants += 1
    
    ## BUG: some utterances do not have consonants or vowels
    if num_consonants == 0 or num_nucleus == 0:
        return rtn

    v_to_c_ratio = num_nucleus/num_consonants
    rtn["num_nucleus"] = num_nucleus # Number of vocalic nucleus
    rtn["num_consonants"] = num_consonants # Number of intervocalic consonants
    rtn["v_to_c_ratio"] = v_to_c_ratio # The ratio of vocalic nucleus to intervocalic consonants

    ## Deal with the intervals with continous V or C labels
    # BUG: We do not implement this part. We think both the continuous V and C labels contribute to rhythm
    # for i in range(0, len(nLabel)-1):
    #     label = nLabel[i].get_label()
    #     nextlabel = nLabel[i+1].get_label()
    #     if label == "C" and nextlabel == "C":
    #         Remove right boundary... 1 i
    #         Replace interval text... 1 i i+1 CC C Literals
    #         mLabel = Get number of intervals... 1
    #         nLabel = mLabel

    #     if label == "V" and nextlabel == "V":
    #         Remove right boundary... 1 i
    #         Replace interval text... 1 i i+1 VV V Literals
    #         mLabel = Get number of intervals... 1
    #         nLabel = mLabel

    # Initialization of variables
    sum_dur = 0
    p_num = 0
    v_sum_dur = 0
    v_num = 0
    c_sum_dur = 0
    c_num = 0
    p_arr = []

    # The consonants and vowels are collapsed to make CV sequence only.
    for i in nLabel:

        label = i.get_label()
        _type = i.get_type()

        # This portion is to check whether there are some labels that I missed to change into C's or V's
        # If there are missing labels, those missing labels will be appear not indented (or tabbed)

        if _type == _w.get_o():
            continue

        # Basic duration information of each phone
        if _type == _w.get_v() or _type == _w.get_c():
            pbeg = i.get_start()
            pend = i.get_end()
            pdur = i.get_dur()*1000
            sum_dur += pdur
            p_num += 1
            p_arr.append(i)

        if _type == _w.get_v():
            vbeg = i.get_start()
            vend = i.get_end()
            vdur = i.get_dur()*1000
            v_sum_dur += vdur
            v_num += 1

        if _type == _w.get_c():
            cbeg = i.get_start()
            cend = i.get_end()
            cdur = i.get_dur()*1000
            c_sum_dur += cdur
            c_num += 1

    ## BUG: To deal with the division by zero problem, we need to add up the number when cumulative number is 1
    p_num = p_num if p_num > 1 else 2
    v_num = v_num if v_num > 1 else 2
    c_num = c_num if c_num > 1 else 2

    mean_p_dur = sum_dur/(p_num-1)
    mean_v_dur = v_sum_dur/(v_num-1)
    mean_c_dur = c_sum_dur/(c_num-1)
    sum_dur_sec = sum_dur/1000
    v_sum_dur_sec = v_sum_dur/1000
    c_sum_dur_sec = c_sum_dur/1000

    rtn["sum_dur_sec"] = sum_dur_sec # Total duration of non-silent portion of the speech file
    rtn["phones_num"] = p_num # Number of phones
    rtn["v_sum_dur_sec"] = v_sum_dur_sec # Total duration of vocalic intervals
    rtn["v_num"] = v_num # Total number of vocalic intervals
    rtn["c_sum_dur_sec"] = c_sum_dur_sec # Total duration of intervocalic consonants
    rtn["c_num"] = c_num # Total number of consonants

    # Mean duration of vocalic intervals, intervocalic consonantal intervals
    rtn["mean_v_dur"] = mean_v_dur # Mean of vocalic intervals
    rtn["mean_c_dur"] = mean_c_dur # Mean of intervocalic consonantal intervals
    rtn["mean_p_dur"] = mean_p_dur # Mean of (C & V) intervals

    # Initialization II for standard deviation
    p_sum_variance = 0
    v_sum_variance = 0
    c_sum_variance = 0

    for i in nLabel:

        label = i.get_label()
        _type = i.get_type()

        if _type == _w.get_v():
            vbeg = i.get_start()
            vend = i.get_end()
            vdur = i.get_dur()*1000

            p_variance = (vdur - mean_p_dur)**2
            v_variance = (vdur - mean_v_dur)**2

            p_sum_variance += p_variance
            v_sum_variance += v_variance

        if _type == _w.get_c():
            cbeg = i.get_start()
            cend = i.get_end()
            cdur = i.get_dur()*1000

            p_variance = (cdur - mean_p_dur)**2
            c_variance = (cdur - mean_c_dur)**2

            p_sum_variance += p_variance
            c_sum_variance += c_variance


    rtn["p_sum_variance"] = p_sum_variance # Variance of total amount of phones

    p_std_dev = math.sqrt(p_sum_variance/(p_num-1))
    v_std_dev = math.sqrt(v_sum_variance/(v_num-1))
    c_std_dev = math.sqrt(c_sum_variance/(c_num-1))

    varco_P = (p_std_dev/mean_p_dur)*100
    varco_V = (v_std_dev/mean_v_dur)*100
    varco_C = (c_std_dev/mean_c_dur)*100

    # Standard deivations for phones, vowels, and consonants
    rtn["v_std_dev"] = v_std_dev # Standard Deviation of total amount of vocalic intervals, i.e. delta-V
    rtn["c_std_dev"] = c_std_dev # Standard Deviation of total amount of intervocalic consonantal intervals, i.e. delta-C
    rtn["p_std_dev"] = p_std_dev # Standard Deviation of total amount of phones

    percent_v = (v_sum_dur/sum_dur)*100
    speech_rate = num_nucleus/(sum_dur/1000)

    # Global Interval Proportions (GIP): %v, delta-V, delta-C, varcoV, varcoC, speech rate
    rtn["varco_P"] = varco_P # varcoP
    rtn["varco_V"] = varco_V # varcoV
    rtn["varco_C"] = varco_C # varcoC
    rtn["percent_v"] = percent_v # percentage of vocalic intervals over total intervals
    rtn["speech_rate"] = speech_rate # speech rate
    rtn["npvi"] = nPVI(_w, p_num, p_arr) # nPVI
    rtn["npvi_v"] = nPVI(_w, p_num, p_arr, type=_w.get_v()) # Normalized vocalic scores (nPVI-V)
    rtn["npvi_c"] = nPVI(_w, p_num, p_arr, type=_w.get_c()) # Normalized consonantal scores (nPVI-C)
    rtn["rpvi"] = rPVI(_w, p_num, p_arr) # rPVI
    rtn["rpvi_v"] = rPVI(_w, p_num, p_arr, type=_w.get_v()) # raw vocalic scores (rPVI-V)
    rtn["rpvi_c"] = rPVI(_w, p_num, p_arr, type=_w.get_c()) # raw consonantal scores (rPVI-C)
    rtn["cci"] = CCI(_w, p_num, p_arr) # CCI
    rtn["cci_v"] = CCI(_w, p_num, p_arr, type=_w.get_v()) # CCI of raw vocalic 
    rtn["cci_c"] = CCI(_w, p_num, p_arr, type=_w.get_c()) # CCI of raw consonantal
    rtn["status"] = 1

    return rtn
