import os
import argparse
from tqdm import tqdm

import glob
import yaml
import json
from espnet.utils.cli_utils import strtobool
from gop_web_parser import GOP

parser = argparse.ArgumentParser()

# models/chinese-model/model_online/gop_L2_DS_test_0928_capt_hires/log/gop.1.log
parser.add_argument("--log_dir",
                     default="models/chinese-model/model_online/gop_L2_DS_test_0928_capt_hires/log",
                     type=str)

parser.add_argument("--json_dir",
                     default="models/chinese-model/model_online/gop_L2_DS_test_0928_capt_hires/json",
                     type=str)

parser.add_argument("--words_fn",
                     default="models/chinese-model/data/lang_hanyu_lg/words.txt",
                     type=str)

parser.add_argument("--text_fn",
                     default="data/gop_L2_DS_test_0928/text",
                     type=str)

parser.add_argument("--conf",
                     default="models/chinese-model/sample_worker_ch.yaml",
                     type=str)

parser.add_argument("--s2t",
                     default=False,
                     type=strtobool)

args = parser.parse_args()

log_dir = args.log_dir
words_fn = args.words_fn
text_fn = args.text_fn
s2t = args.s2t

vocab = {}
oov_vocab = []
text_dict = {}
fns = glob.glob(log_dir + "/gop*log")

# conf
with open(args.conf) as f:
    conf = yaml.safe_load(f)
conf.setdefault('s2t', s2t)

with open(words_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        word_token = info[0].upper() if conf.get('prompt-style') else info[0].lower()
        word_id = info[1]
        vocab[word_token] = word_id

with open(text_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        utt_id = info[0]
        content = " ".join(info[1:])
        # check OOV
        has_oov = False
        for syb in info[1:]:
            syb = syb.upper() if conf.get('prompt-style') else syb.lower()
            if syb not in vocab:
                has_oov = True
                oov_vocab.append(syb)
                break
        
        if has_oov:
            continue
        
        text_dict[utt_id] = content

oov_vocab = set(oov_vocab)
print('Your OOVs vocab: {}'.format(oov_vocab))

gop_parser = GOP(conf)
gop_dict = {}
special_tag = "compute-dnn-bi-gop.cc:137)"

for log_path in tqdm(fns):
    with open(log_path, "r") as fn:
        for line in fn.readlines():
            #LOG (compute-dnn-bi-gop[5.5.474~5-72ca1]:main():compute-dnn-bi-gop.cc:137) ADW-HS01803_da3_zi4_ADW GOP message <GOP> t_B SIL_B 13 -11.5933 A:3_E A:3_E 19 0 ts4_S ttss_h4_B 30 -0.721965  <GOP>
            if special_tag in line:
                info = line.split(special_tag)[1].split()
                utt_id = info[0]
                gop_msg = " ".join(info[1:])
                 
                if utt_id not in text_dict:
                    continue
                
                prompt = text_dict[utt_id]
                # process GOP
                final_msg = gop_msg.split("<GOP>")[1]
                gop_parser.set_prompt(prompt)

                # try:
                # if utt_id != 'speakerIp16_A2_002003002001-promptIp16_A2_en_22_109_102':
                #     continue
                gop_word_dict = gop_parser.process_GOP(final_msg)
                gop_dict[utt_id] = gop_word_dict
                # except:
                #     print(utt_id, prompt)

with open(args.json_dir + "/gop_scores.json", "w") as fn:
    json.dump(gop_dict, fn, indent=4)
