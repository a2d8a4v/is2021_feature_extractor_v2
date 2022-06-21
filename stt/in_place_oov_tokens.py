import json
import sys
import argparse
from utils import (
    process_tltchool_gigaspeech_interregnum_tokens
)

parser = argparse.ArgumentParser()

parser.add_argument("--input_text_file",
                    default="data/train/text",
                    type=str)

parser.add_argument("--output_text_file",
                    default="data/train/text",
                    type=str)

parser.add_argument("--words_file",
                    default="Librispeech-model-mct-tdnnf/data/lang/words.txt",
                    type=str)

args = parser.parse_args()


def opentext(file):
    s = set()
    t = dict()
    with open(file, "r") as f:
        for l in f.readlines():
            _l = l.split()
            uttid = _l[0]
            words = _l[1:]
            t[uttid] = " ".join(words)
            # remove the utt_id part
            for word in words:
                s.add(word.lower())
    return list(s), t

def openwords(file):
    s = dict()
    with open(file, "r") as f:
        for l in f.readlines():
            s.setdefault(l.split()[0].lower(), [])
    return s


if __name__ == '__main__':

    # variables
    unk = '<UNK>'
    words_file = args.words_file
    input_text_file = args.input_text_file
    output_text_file = args.output_text_file

    # get words from input
    words_list, text_dict = opentext(input_text_file)

    # get words in dict
    words_dict = openwords(words_file)

    # get oov word list
    oov_words_list = list(set([word for word in words_list if word not in words_dict]))

    # in-place the oov words with unknown word token
    with open(output_text_file, 'w') as f:
        for utt_id, words_str in text_dict.items():
            text_list = []
            words_str = process_tltchool_gigaspeech_interregnum_tokens(words_str)
            for word in words_str.split():
                if word in oov_words_list:
                    text_list.append(unk)
                    continue
                text_list.append(word.upper())
            f.write(
                "{} {}\n".format(
                        utt_id, " ".join(text_list)
                    )
            )
