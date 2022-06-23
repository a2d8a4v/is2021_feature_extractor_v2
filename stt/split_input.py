import os
from tqdm import tqdm
import argparse
from utils import (
    jsonLoad,
    pickleStore,
    pikleOpen,
    splitList,
    filter_items,
    copyfile,
    jsonSave
)

parser = argparse.ArgumentParser()

parser.add_argument("--split_number",
                    default=10,
                    required=True,
                    type=int)

parser.add_argument("--output_dir_path",
                    default='data/train_tr/splited_data',
                    required=True,
                    type=str)

parser.add_argument("--wav_scp_file_path",
                    default='data/train_tr/wav.scp',
                    required=True,
                    type=str)

parser.add_argument("--text_file_path",
                    default='data/train_tr/text',
                    required=True,
                    type=str)

parser.add_argument("--utt2dur_file_path",
                    default='data/train_tr/utt2dur',
                    required=True,
                    type=str)

parser.add_argument("--gop_json_file_path",
                    default='gop_result_dir/json/gop.json',
                    required=True,
                    type=str)

parser.add_argument("--lexicon_file_path",
                    default='data/local/dict/lexicon.txt',
                    required=True,
                    type=str)

args = parser.parse_args()

# variables
utt_list = []
wavscp_dict = {}
text_dict = {}
utt2dur_dict = {}

wav_scp_file_path = args.wav_scp_file_path
text_file_path = args.text_file_path
utt2dur_file_path = args.utt2dur_file_path
gop_json_file_path = args.gop_json_file_path
lexicon_file_path = args.lexicon_file_path
output_dir_path = args.output_dir_path

# split wav.scp
with open(wav_scp_file_path, "r") as fn:
    for i, line in enumerate(fn.readlines()):
        info = line.split()
        wavscp_dict[info[0]] = info[1]
        utt_list.append(info[0])

# anno. (punc)
with open(text_file_path, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_dict[info[0]] = " ".join(info[1:]).upper()

# utt2dur
with open(utt2dur_file_path, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        utt2dur_dict[info[0]] = str(info[1])

# gop
gop_dict = jsonLoad(gop_json_file_path)

# validation
assert len(set(utt_list) - set(list(wavscp_dict.keys()))) == 0, 'something wrong in {}'.format(wav_scp_file_path)
assert len(set(list(wavscp_dict.keys())) - set(utt_list)) == 0, 'something wrong in {}'.format(wav_scp_file_path)
assert len(set(utt_list) - set(list(text_dict.keys()))) == 0, 'something wrong in {}'.format(text_file_path)
assert len(set(list(text_dict.keys())) - set(utt_list)) == 0, 'something wrong in {}'.format(text_file_path)
assert len(set(utt_list) - set(list(utt2dur_dict.keys()))) == 0, 'something wrong in {}'.format(utt2dur_file_path)
assert len(set(list(utt2dur_dict.keys())) - set(utt_list)) == 0, 'something wrong in {}'.format(utt2dur_file_path)
assert len(set(utt_list) - set(list(gop_dict.keys()))) == 0, 'something wrong in {}'.format(gop_json_file_path)
assert len(set(list(gop_dict.keys())) - set(utt_list)) == 0, 'something wrong in {}'.format(gop_json_file_path)

# split them!
print('processing...')

listTemp = splitList(utt_list, args.split_number)
for i, chunk_utt_list in enumerate(listTemp):
    chunk_wavscp_dict  = filter_items(wavscp_dict, chunk_utt_list)
    chunk_text_dict    = filter_items(text_dict, chunk_utt_list)
    chunk_utt2dur_dict = filter_items(utt2dur_dict, chunk_utt_list)
    chunk_gop          = filter_items(gop_dict, chunk_utt_list)

    # save chunk file
    # save wav.scp
    file_basename = os.path.basename(wav_scp_file_path)
    file_name, extension = os.path.splitext(file_basename)
    new_file_name = file_name + ".{}".format(i+1) + extension
    with open(os.path.join(output_dir_path, new_file_name), 'w') as fn:
        for utt_id, wav_scp_info in chunk_wavscp_dict.items():
            fn.write("{} {}\n".format(utt_id, wav_scp_info))
        
    # save text
    file_basename = os.path.basename(text_file_path)
    file_name, extension = os.path.splitext(file_basename)
    new_file_name = file_name + ".{}".format(i+1) + extension
    with open(os.path.join(output_dir_path, new_file_name), 'w') as fn:
        for utt_id, text_info in chunk_text_dict.items():
            fn.write("{} {}\n".format(utt_id, text_info))

    # save utt2dur
    file_basename = os.path.basename(utt2dur_file_path)
    file_name, extension = os.path.splitext(file_basename)
    new_file_name = file_name + ".{}".format(i+1) + extension
    with open(os.path.join(output_dir_path, new_file_name), 'w') as fn:
        for utt_id, dur_info in chunk_utt2dur_dict.items():
            fn.write("{} {}\n".format(utt_id, dur_info))

    # save gop
    file_basename = os.path.basename(gop_json_file_path)
    file_name, extension = os.path.splitext(file_basename)
    new_file_name = file_name + ".{}".format(i+1) + extension
    jsonSave(chunk_gop, os.path.join(output_dir_path, new_file_name))

    # save lexicon
    file_basename = os.path.basename(lexicon_file_path)
    file_name, extension = os.path.splitext(file_basename)
    new_file_name = file_name + ".{}".format(i+1) + extension
    copyfile(lexicon_file_path, os.path.join(output_dir_path, new_file_name))

print('Spliting Done!')