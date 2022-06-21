import sys
import argparse

## parser initializing
parser = argparse.ArgumentParser(description='Train classifier model')
# parser.add_argument('--corr_phn_fn', default="exp/nnet3_chain/model_online/align_lats_kids1500_hires/score_phn_10/kids1500_hires.ctm", type=str)
parser.add_argument('--input_file', default="./data/train/text", type=str)
parser.add_argument('--output_file', default="./data/train/text_phn", type=str)
args = parser.parse_args()


def opendict(file):
    rtn  = {}
    with open(file,'r') as f:
        for l in f.readlines():
            _l = l.split()
            rtn[_l[0]] = _l[1:]
    return rtn

def opentext(file):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            s[l_[0]] = l_[1:]
    return s


## Variables
error = []
output = ""

## START
_dict = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/local/dict/lexicon.txt.new"
_text = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/lang_1char/text_all_cleaned"
_data = opentext(_text)
_dict = opendict(_dict)

## Print data
rtn = {}
for utt_id, text in _data.items():
    rtn[utt_id] = [_dict[w] for w in text]

with open(args.output_file, "w") as f:
    for utt_id, text in rtn.items():
        f.write("{} {}\n".format(utt_id, " ".join(sum(text, []))))
print("DONE!")

# if error:
#     print(error[:2])

