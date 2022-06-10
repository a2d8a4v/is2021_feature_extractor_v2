#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ming Tu

# This script contains several functions to analyze ctm files output by acoustic model
# and convert them to textgrid format files.
# This code is adapted from corresponding code in Montreal-Forced-aligner 
# (https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git)

import os, re
import sys
import traceback
from textgrid import TextGrid, IntervalTier, Interval
from io import open as io_open


def parse_ctm(ctm_info, mode='word'):

    rtn = []

    for phn_or_word, begin, duration, conf in ctm_info:

        begin = begin
        duration = duration
        end = round(begin + duration, 4)

        rtn.append([begin, end, phn_or_word])

    return rtn

def generate_utt2dur(utt2dur_file):
    mapping = {}
    with open(utt2dur_file, 'r') as fid:
        utt_dur_pairs = fid.readlines()
        for item in utt_dur_pairs:
            utt = item.strip().split()[0]
            dur = item.strip().split()[1]
            mapping[utt] = float(dur)

    return mapping

def ctm_to_textgrid(word_ctm, phone_ctm, out_directory, utt2dur, utt_id, frameshift=0.01):
    textgrid_write_errors = ""
    frameshift = frameshift
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    log_path = os.path.join(out_directory, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    utt2dur_mapping = generate_utt2dur(utt2dur)

    maxtime = utt2dur_mapping[utt_id]

    # insert word tier firstly and phoneme tier secondly
    # try:
    tg = TextGrid(minTime=0, maxTime=maxtime)
    wordtier = IntervalTier(name='words', maxTime=maxtime)
    phonetier = IntervalTier(name='phones', maxTime=maxtime)
    for interval in word_ctm:
        if maxtime - interval[1] < frameshift:  # Fix rounding issues
            interval[1] = maxtime
        interval = Interval(minTime=interval[0], maxTime=interval[1], mark=interval[2])
        wordtier.addInterval(interval)
    for interval in phone_ctm:
        if maxtime - interval[1] < frameshift:
            interval[1] = maxtime
        # BUG: remove B E I information from phoneme
        interval[2] = str(interval[2].split('_')[0]) # interval[2] = re.sub("\d+","",interval[2].split('_')[0])
        interval = Interval(minTime=interval[0], maxTime=interval[1], mark=interval[2])
        phonetier.addInterval(interval)
    tg.tiers.append(wordtier)
    tg.tiers.append(phonetier)
    outpath = os.path.abspath(os.path.join(out_directory, utt_id + '.TextGrid'))
    tg.write(outpath)

    return outpath

    # except Exception as e:
    #     exc_type, exc_value, exc_traceback = sys.exc_info()
    #     textgrid_write_errors = '\n'.join(
    #         traceback.format_exception(exc_type, exc_value, exc_traceback))

    # if textgrid_write_errors:
    #     error_log = os.path.abspath(os.path.join(log_path, 'output_errors.txt'))
    #     with io_open(error_log, 'w', encoding='utf-8') as f:
    #         f.write(
    #             u'The following exceptions were encountered during the ouput of the alignments to TextGrids:\n\n')
    #         f.write(u'{}:\n'.format(utt_id))
    #         f.write(u'{}\n\n'.format(textgrid_write_errors))

    #     return error_log
