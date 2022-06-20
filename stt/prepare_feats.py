import os
import json
import soundfile
from tqdm import tqdm
from kaldi_models import SpeechModel
from audio_models import AudioModel
from g2p_model import G2PModel
import argparse
from espnet.utils.cli_utils import strtobool
from utils import (
    pickleStore,
    pikleOpen,
    opendict,
    opentext,
    fix_data_type,
    ctm2textgrid,
    readwav,
    movefile
)

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="/share/nas165/teinhonglo/AcousticModel/2020AESRC/s5/data/cv_56",
                    type=str)

parser.add_argument("--model_name",
                    default="gigaspeech",
                    type=str)

parser.add_argument("--gop_result_dir",
                    default="model/model_online/decode/gop",
                    type=str)

parser.add_argument("--utt2dur_file_path",
                    default="data/train_tr/utt2dur",
                    type=str)

parser.add_argument("--text_path",
                    default="data/train/text",
                    type=str)

parser.add_argument("--gop_json_fn",
                    default="gop_result_dir/json/gop.json",
                    type=str)

parser.add_argument("--sample_rate",
                    default=16000,
                    type=int)

parser.add_argument("--tag",
                    default="",
                    type=str)

# We accept both ESPNet-based and Kaldi-based lexicon
parser.add_argument("--lexicon",
                    default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/acoustic_phonetic_features/data/local/dict/lexicon.txt",
                    type=str,
                    required=True)

parser.add_argument("--long_decode_mode",
                    default=False,
                    type=strtobool)


args = parser.parse_args()

lexicon = args.lexicon
data_dir = args.data_dir
model_name = args.model_name
gop_result_dir = args.gop_result_dir
gop_json_fn = args.gop_json_fn
sample_rate = args.sample_rate
utt2dur_file_path = args.utt2dur_file_path
tmp_apl_decoding = "tmp_apl_decoding_"+args.tag if args.tag else "tmp_apl_decoding"

# Temporary data saved for ToBI
tobi_path = os.path.abspath(os.path.join(data_dir, tmp_apl_decoding, "tobi"))
if not os.path.isdir(tobi_path):
    os.makedirs(tobi_path)

output_dir = os.path.join(data_dir, model_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

lexicon_file_path = os.path.join(output_dir, "lexicon")

wavscp_dict = {}
text_dict = {}
utt_list = []
err_list = []
# stt and ctm
all_info = {}
recog_dict = {}

# lexicon: words to phonemes
word2phn_dict = opendict(
    lexicon_file_path if os.path.exists(lexicon_file_path) else lexicon
)

# write new lexicon result to file
if not os.path.exists(lexicon_file_path):
    with open(lexicon_file_path, "w") as fn:
        for word, phn_list in word2phn_dict.items():
            fn.write("{} {}\n".format(word, " ".join(phn_list)))

with open(data_dir + "/wav.scp", "r") as fn:
    for i, line in enumerate(fn.readlines()):
        info = line.split()
        wavscp_dict[info[0]] = info[1]
        utt_list.append(info[0])

# anno. (punc)
# move text load beforehand to avoid unpredicable error in dict and save more time and your GPU memory
with open(args.text_path, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_dict[info[0]] = " ".join(info[1:]).upper() # Due to the tokens inside gigaspeech ad wsj is upper-case, we need to change the text to upper-case also.

# recog. result
with open(gop_json_fn, "r") as fn:
    gop_json = json.load(fn)
for utt_id, gop_data in gop_json.items():
    word_list = []
    for word_info in gop_data.get('GOP'):
        word = word_info[0]
        word_list.append(word)
    recog_dict[utt_id] = " ".join(word_list)

# initialize models
speech_model = SpeechModel(recog_dict, gop_result_dir, gop_json_fn)
audio_model = AudioModel(sample_rate)
g2p_model = G2PModel(lexicon_file_path)

## Due to the memory problem if load all data into memory
if args.long_decode_mode:
    if os.path.exists(os.path.join(data_dir, tmp_apl_decoding+".list")):
        print("Decoded features loading...")
        with open(os.path.join(data_dir, tmp_apl_decoding+".list"), "r") as fn:
            for l in tqdm(fn.readlines()):
                l_ = l.split()
                # We do not need to keep too much data in memory during inference, we can just load all the data after end the inference
                all_info[l_[0]] = {} if args.long_decode_mode else pikleOpen(l_[1])

print("Decoding Start")

for i, utt_id in enumerate(tqdm(utt_list)):

    if utt_id in all_info:
        continue

    # skip error
    if utt_id in err_list:
        continue

    wav_path = wavscp_dict[utt_id]
    text_prompt = text_dict[utt_id]

    # Confirm the sampling rate is equal to that of the training corpus.
    # If not, you need to resample the audio data before inputting to speech2text
    speech, rate = readwav(wav_path, sample_rate)
    assert rate == sample_rate
    total_duration = speech.shape[0] / rate

    # audio feature
    f0_list, f0_info = audio_model.get_f0(speech)
    energy_list, energy_info = audio_model.get_energy(speech)
    formants_info = audio_model.get_formants(wav_path, speech, rate)

    # fluency feature and confidence feature
    # we already decode the transcript with Kaldi
    text = speech_model.recog(utt_id)

    # BUG: Because of the two dictionaries have different words, we need to extend the dictionary with G2P toolkit
    word2phn_dict = g2p_model.g2p(text, word2phn_dict)

    # alignment (stt)
    ctm_info, phone_ctm_info = speech_model.get_ctm(utt_id)
    _, phone_text = speech_model.get_phone_ctm(ctm_info, word2phn_dict)
    
    sil_feats_info, response_duration = speech_model.sil_feats(ctm_info, total_duration)
    word_feats_info, response_duration = speech_model.word_feats(ctm_info, total_duration)
    phone_feats_info, response_duration = speech_model.phone_feats(phone_ctm_info, total_duration)
    
    rhythm_feats_info = speech_model.rhythm_feats(phone_ctm_info)
    pitch_feats_info = audio_model.get_pitch(speech, ctm_info, f0_list, total_duration, formants_info=formants_info) # we use the timestamp in formant info to save time of calculating f0
    intensity_feats_info = audio_model.get_intensity(speech, ctm_info, energy_list, total_duration, formants_info=formants_info) # we use the timestamp in formant info to save time of calculating energy
    textgrid_file_path = ctm2textgrid(
        ctm_info, phone_ctm_info,
        os.path.join(tobi_path, 'textgrid'),
        utt2dur_file_path,
        utt_id
    )
    tobi_feats_info = audio_model.get_tobi(os.path.abspath(wav_path), tobi_path, os.path.abspath(textgrid_file_path))

    # Save data
    save = fix_data_type(
        { "stt": text, "stt(g2p)": phone_text, "prompt": text_prompt,
                        "wav_path": wav_path, "ctm": ctm_info, "phone_ctm": phone_ctm_info,
                        "feats": {  **f0_info, **energy_info, 
                                    **sil_feats_info, **word_feats_info,
                                    **phone_feats_info,
                                    "pitch": pitch_feats_info,
                                    "intensity": intensity_feats_info,
                                    "formant": formants_info,
                                    "rhythm": rhythm_feats_info,
                                    "total_duration": total_duration,
                                    "tobi": tobi_feats_info,
                                    "tg_path": textgrid_file_path,
                                    "response_duration": response_duration}
        }
    )

    if not args.long_decode_mode:
        all_info[utt_id] = save

    # due to spending too much time on decoding, we need to save each decoding result per utt
    if args.long_decode_mode:
        fp = os.path.join(data_dir, tmp_apl_decoding)
        if not os.path.isdir(fp):
            os.mkdir(fp)
        pickleStore(save, os.path.join(fp, utt_id+".pkl"))
        with open(os.path.join(data_dir, tmp_apl_decoding+".list"), "a") as fn:
            fn.write("{} {}\n".format(utt_id, os.path.join(fp, utt_id+".pkl")))

    # for memory issue, we reset some variables during inference
    del phone_ctm_info
    del ctm_info
    del f0_info
    del energy_info
    del sil_feats_info
    del word_feats_info
    del phone_feats_info
    del pitch_feats_info
    del intensity_feats_info
    del formants_info
    del rhythm_feats_info


# For Saving all data, import data to all_info
if args.long_decode_mode:
    print("Decoded features Saving...")
    with open(os.path.join(data_dir, tmp_apl_decoding+".list"), "r") as fn:
        for l in tqdm(fn.readlines()):
            l_ = l.split()
            all_info[l_[0]] = fix_data_type(pikleOpen(l_[1]))

# Print out the output dir
print(output_dir)

# record error
if err_list:
    with open(output_dir + "/error", "w") as fn:
        for utt_id in err_list:
            fn.write(utt_id + "\n")

with open(output_dir + "/all.json", "w") as fn:
    json.dump(all_info, fn, indent=4, ensure_ascii=False)

# write STT Result to file
if os.path.exists(output_dir + "/text"):
    movefile(output_dir + "/text", output_dir + "/text.bak")
with open(output_dir + "/text", "w") as fn:    
    for utt_id in utt_list:

        # skip error
        if utt_id in err_list:
            continue

        fn.write(utt_id + " " + all_info[utt_id]["stt"] + "\n")

# write alignment results fo file
with open(output_dir + "/ctm", "w") as fn:
    end_time = -100000
    for utt_id in utt_list:

        # skip error
        if utt_id in err_list:
            continue

        ctm_infos = all_info[utt_id]["ctm"]
        for i in range(len(ctm_infos)):
            text_info, start_time, duration, conf = ctm_infos[i]
            # utt_id channel start_time duration text conf
            ctm_info = " ".join([utt_id, "1", str(start_time), str(duration), text_info, str(conf)])
            fn.write(ctm_info + "\n")
