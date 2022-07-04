import os
import sys
import json
import math
import pickle
import shutil
import logging
import subprocess
import parselmouth
import numpy as np
from statistics import mean, median
from parselmouth.praat import run_file
# from textgrid_ops import (
#     ctm_to_textgrid,
#     parse_ctm
# )
# from textgrid import *

def pickleStore(savethings , filename):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return

def pikleOpen(filename):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p

def jsonLoad(scores_json):
    with open(scores_json) as json_file:
        return json.load(json_file)

def jsonSave(save_json, file_path):
    with open(file_path, "w") as output_file:
        json.dump(save_json, output_file, indent=4, ensure_ascii=False)

def opendict(file):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            s[l_[0]] = l_[1:]
    return s

def opentext(file, col_start):
    s = set()
    with open(file, "r") as f:
        for l in f.readlines():
            for w in l.split()[col_start:]:
                s.add(w)
    return [w.lower() for w in list(s)]

def openappend(file):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            s[l.split()[0]] = l.split()[1]
    return s

def openctm(file):
    s = set()
    with open(file, "r") as f:
        for l in f.readlines():
            s.add(l.split()[4])
    return list(s)

def getbyFilter(data, filter):
    rtn = [i for i in data if filter in i]
    return sorted(list(set(rtn)))


def readwav(wav_path, rate):
    _, file_extension = os.path.splitext(wav_path)
    if file_extension.lower() == ".wav":
        import soundfile
        speech, rate = soundfile.read(wav_path)
    else:
        import numpy as np
        from pydub import AudioSegment
        speech = np.array(AudioSegment.from_mp3(wav_path).get_array_of_samples(), dtype=float)
    return speech, rate

def dict_miss_words(text, _dict):
    t = []
    for w in text.split():
        w = w.lower()
        if w not in _dict:
            t.append(w)            
    return " ".join(t)

def process_tltchool_gigaspeech_interregnum_tokens(tokens):
    disfluency_tltspeech  = [
        "@eh", "@ehm", "@em", "@mm", "@mmh", "@ns", "@nuh", "@ug", "@uh", "@um", "@whoah", "@unk"
    ]
    mapping = {
        "@ehm": "",
        "@mm": "",
        "@mmh": "",
        "@ns": "",
        "@nuh": "",
        "@ug": "",
        "@whoah": "",
        "@unk": "<unk>",
        "@uh": "UH",
        "@um": "UM",
        "@em": "EM",
        "@eh": "AH"
    }
    n = []
    for t in tokens.split():
        if t.lower() in disfluency_tltspeech:
            t = mapping[t.lower()]
        n.append(t)

    # BUG: not filter out the tokens if length is equal as one
    if not n:
        n = tokens.split()
    return " ".join(n)


def remove_tltschool_interregnum_tokens(tokens):

    disfluency_tltspeech  = [
        "@eh", "@ehm", "@em", "@mm", "@mmh", "@ns", "@nuh", "@ug", "@uh", "@um", "@whoah", "@unk"
    ]

    n = []
    for t in tokens.split():
        if t.lower() in disfluency_tltspeech:
            t = ""
        n.append(t)

    if not n:
        n = tokens.split()

    return " ".join(n)


def remove_gigaspeech_interregnum_tokens(tokens):
    disfluency_gigaspeech = ["AH", "UM", "UH", "EM", "ER", "ERR"]
    other_gigaspeech = ["<UNK>"]
    _gigaspeech = disfluency_gigaspeech+other_gigaspeech
    n = []
    for t in tokens.split():
        if t.upper() not in _gigaspeech:
            n.append(t)
    
    # BUG: not filter out the tokens if length is equal as one
    if not n:
        n = tokens.split()
    return " ".join(n)

def run_praat(file, *args, capture_output=True):

    assert os.path.isfile(file), "Wrong path to praat script"

    try:
        objects = run_file(file, *args, capture_output=capture_output)
        return objects
    except:
        print("Try again the sound of the audio was not clear")
        return None

def run_cmd(process, *args):
    """Run praat with `args` and return results as a c string
    Arguments:
    :param process: process name.
    :param *args: command line arguments to pass to praat.
    """
    p = subprocess.run([process] + list(map(str, list(args))), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)

    if p.returncode:
        raise Error(''.join(p.stderr.readlines()))
    else:
        return p.stdout

def from_textgrid_to_ctm(file_path):

    word_ctm_info = []
    phoneme_ctm_info = []

    tg = textgrid.TextGrid()
    tg.read(file_path)

    # Word-level
    words = tg.tiers[0]
    for i in range(len(words)):
        word = words[i].mark
        if word == "":
            continue
        start_time = round(words[i].minTime, 4)
        end_time = round(words[i].maxTime, 4)
        duration = round(end_time - start_time, 4)
        conf = round(1, 4)
        word_ctm_info.append(
            [word, start_time, duration, conf]
        )

    # Phoneme-level
    phonemes = tg.tiers[0]
    for i in range(len(phonemes)):
        phoneme = phonemes[i].mark
        if phoneme == "":
            continue
        start_time = round(phonemes[i].minTime, 4)
        end_time = round(phonemes[i].maxTime, 4)
        duration = round(end_time - start_time, 4)
        conf = round(1, 4)
        phoneme_ctm_info.append(
            [phoneme, start_time, duration, conf]
        )

    return word_ctm_info, phoneme_ctm_info

def fix_data_type(data_dict):
    rtn = {}
    for term, data in data_dict.items():
        if isinstance(data, np.float32):
            data = np.float64(data)
        if isinstance(data, dict):
            data = fix_data_type(data)
        if isinstance(data, np.ndarray):
            data = data.tolist()
        rtn[term] = data
    return rtn

def ctm2textgrid(word_ctm, phone_ctm, save_file_dir, utt2dur_file_path, utt_id):
    word_ctm = parse_ctm(word_ctm, mode='word')
    phone_ctm = parse_ctm(phone_ctm, mode='phone')
    return ctm_to_textgrid(word_ctm, phone_ctm, save_file_dir, utt2dur_file_path, utt_id)

def getLeft(timepoint, total_duration, formants):
    timepoint = round(timepoint, 4)
    percent = timepoint / total_duration
    len_formant = formants.shape[0]
    time_list = formants[:,0]

    # get the ambiguous frame position
    position = math.ceil(len_formant*percent)

    # check the timestamp of formant is bigger than timepoint
    # we hypothesis two continuous timestamp should have similar frequency
    formant_timestamp = round(time_list[position].item(), 4)
    while True:
        if position <= 0:
            return position
        if position >= time_list.shape[0]-1:
            return position
        if (formant_timestamp == timepoint) or (formant_timestamp <= timepoint and round(time_list[position+1].item(), 4) >= timepoint):
            return position
        elif formant_timestamp > timepoint:
            if position == 0:
                return position
            position = position-1
            formant_timestamp = round(time_list[position].item(), 4)
        elif formant_timestamp < timepoint and round(time_list[position+1].item(), 4) < timepoint:
            # if the ambiguous position has deviation
            if position == time_list.shape[0]-1:
                return position
            position = position+1
            formant_timestamp = round(time_list[position].item(), 4)
    return

def getRight(timepoint, total_duration, formants):
    timepoint = round(timepoint, 4)
    percent = timepoint / total_duration
    len_formant = formants.shape[0]
    time_list = formants[:,0]

    # get the ambiguous frame position
    position = math.floor(len_formant*percent)

    # check the timestamp of formant is bigger than timepoint
    # we hypothesis two continuous timestamp should have similar frequency
    formant_timestamp = round(time_list[position].item(), 4)
    while True:
        if position <= 0:
            return position
        if position >= time_list.shape[0]-1:
            return position
        if (formant_timestamp == timepoint) or (formant_timestamp <= timepoint and round(time_list[position+1].item(), 4) >= timepoint):
            return position
        elif formant_timestamp < timepoint:
            if position == time_list.shape[0]-1:
                return position
            position = position+1
            formant_timestamp = round(time_list[position].item(), 4)
        elif formant_timestamp > timepoint and round(time_list[position+1].item(), 4) > timepoint:
            # if the ambiguous position has deviation
            if position == 0:
                return position
            position = position-1
            formant_timestamp = round(time_list[position].item(), 4)
    return

def splitList(listTemp, times):
    n = math.ceil(len(listTemp)/times)
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]

def filter_items(data_dict, uttid_chunk):
    rtn = {}
    for utt_id in uttid_chunk:
        rtn[utt_id] = data_dict[utt_id]
    return rtn

def copyfile(src, dst):
    shutil.copyfile(
        os.path.abspath(src),
        os.path.abspath(dst)
    )
    return

def movefile(src, dst):
    shutil.move(
        os.path.abspath(src),
        os.path.abspath(dst)
    )
    return

def write_txt(data, dst, write_type='w'):
    if isinstance(data, list):
        with open(dst, write_type) as f:
            for item in data:
                f.write("{}\n".format(item))

def write_txt_with_uttids(data, dst, write_type='w'):
    if isinstance(data, dict):
        with open(dst, write_type) as f:
            for utt_id, item in data.items():
                f.write("{} {}\n".format(utt_id, item))

def open_utt2value(file):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            s[l_[0]] = l_[1]
    return s


def open_text(file):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            s[l_[0]] = " ".join(l_[1:])
    return s


def openlist(file):
    s = []
    with open(file, "r") as f:
        for l in f.readlines():
            s.append(l.split()[0])
    return s

def open_utt2pkl(file):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            s[l_[0]] = pikleOpen(l_[1])
    return s


def printOut(data):
    for i,j in data.items():
        sys.stdout.write(i+'\t')
        sys.stdout.write(" ".join(j))
        sys.stdout.write('\n')
