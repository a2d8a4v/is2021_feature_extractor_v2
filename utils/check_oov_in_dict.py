import json
import sys
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--input_text_file",
                    default="data/train/text",
                    type=str)

parser.add_argument("--words_file",
                    default="Librispeech-model-mct-tdnnf/data/lang_t3/words.txt",
                    type=str)

args = parser.parse_args()


def jsonLoad(scores_json):
    with open(scores_json) as json_file:
        return json.load(json_file)


def jsonSave(path, dict_json):
    with open(path, "w") as f:
        json.dump(dict_json, f, indent=4)


def opentext(file):
    s = set()
    with open(file, "r") as f:
        for l in f.readlines():
            for word in l.split():
                s.add(word.lower())
    return list(s)

def openwords(file):
    s = dict()
    with open(file, "r") as f:
        for l in f.readlines():
            s.setdefault(l[0].lower(), [])
    return s


if __name__ == '__main__':

    # text
    input_text_file = args.input_text_file
    words_list = opentext(input_text_file)

    # words
    words_file = args.words_file
    words_dict = opentext(words_file)

    oov_words_list = list(set([word for word in words_list if word not in words_dict]))

    print(
        "{} words not in words.txt !!".format(
            len(oov_words_list)
        )
    )

    print(
        oov_words_list
    )