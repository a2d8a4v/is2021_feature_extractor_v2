import json
import sys
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--json_file",
                    default="data/train/all.json",
                    type=str)

args = parser.parse_args()


def jsonLoad(scores_json):
    with open(scores_json) as json_file:
        return json.load(json_file)


def jsonSave(path, dict_json):
    with open(path, "w") as f:
        json.dump(dict_json, f, indent=4)


def opentext( file, col_start ):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            f_ = []
            for w in l_[col_start:]:
                f_.append(checkDisfluency(w))
            s[l_[0]] = f_
    return s


if __name__ == '__main__':

    file_path = args.json_file
    json_data = jsonLoad(file_path)

    if 'utts' in json_data:
        json_data = json_data.get('utts')
    
    print(len(json_data.keys()))