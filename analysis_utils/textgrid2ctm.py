import argparse
import textgrid
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input_file', default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/montreal_working/pytobi-test/19-198-0000.TextGrid", type=str)
parser.add_argument('--output_file', default="test", type=str)
args = parser.parse_args()

# read textgrid file
tg = textgrid.TextGrid()
tg.read(args.input_file)

# Word-level
words = tg.tiers[1]

# utt_id channel_num start_time phone_dur phone_id
with open(args.output_file, 'w') as fn:
    head, _ = os.path.splitext(os.path.basename(args.input_file))
    for i in range(len(words)):
        l_ = " ".join(map(str,[head, 1, words[i].minTime, float(words[i].maxTime) - float(words[i].minTime), words[i].mark]))
        fn.write(l_+"\n")
