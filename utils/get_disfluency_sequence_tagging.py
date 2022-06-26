import json
import pickle
import sys
from utilities import (
    jsonLoad,
    pikleOpen,
    pickleStore,
    printOut
)


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


def checkDisfluency(word):
    
    # Labels
    fluency    = "O"
    disfluency = "D"

    b = ["@breath", "@cough", "@laugh", "@bkg", "@ns", "@sil"]

    # Interregnum, and controllable issues by speakers
    d = ["@sil"]
    o = "@"

    if word in d:
        return disfluency
    else:
        if o in word and word not in b: 
            return disfluency
        return fluency

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_text_file_path",
                        default="data/train/text",
                        type=str)

    args = parser.parse_args()

    ## variables
    input_text_file_path = args.input_text_file_path
    text_data = opentext( input_text_file_path, 1 )

    ## Print data
    printOut(text_data)

