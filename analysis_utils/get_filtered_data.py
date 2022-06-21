import json
import pickle
import nltk
import numpy as np
import math
# from similar_text import similar_text as st


def jsonLoad(scores_json):
    with open(scores_json) as json_file:
        return json.load(json_file)


def pickleStore( savethings , filename ):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return


def pikleOpen( filename ):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p


def pad_list_n(xs, pad_value, padding):
    n_batch = len(xs)
    if padding is None:
        max_len = max(len(x) for x in xs)
    else:
        max_len = padding
    _xs = np.zeros((n_batch, max_len))
    _xs[:,:] = pad_value

    for i, _x in enumerate(xs):
        _xs[i,:len(_x)] = np.array(_x)

    return _xs


def opentext(file, col_start):
    s = set()
    with open(file, "r") as f:
        for l in f.readlines():
            for w in l.split()[col_start:]:
                s.add(w)
    return sorted(list(s))


def open_from_text(file, col_start):
    rtn = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            rtn[l_[0]] = l_[col_start:]
    return rtn


def opendict( file ):
    rtn = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            rtn[l_[0]] = l_[2:]
    return rtn


def genTextOnly(file, col_start):
    s = []
    with open(file, "r") as f:
        for l in f.readlines():
            s.append(l.split()[col_start:])
    return s


def gentokenids( dict_ ):
    rtn = {}
    for i, w in enumerate(dict_):
        rtn[w] = i
    return rtn


def transdict(transcript, transcript_ids):
    new = []
    for utt in transcript:
        utt_ = []
        for w in utt:
            utt_.append(transcript_ids[w])
        new.append(utt_)
    return new


def getbyFilter(data, filter):
    rtn = [i for i in data if filter in i]
    return sorted(list(set(rtn)))


def filterOutNumbers(phone):
    return "".join(list(filter(lambda x: x.isalpha(), phone)))


## TLT-school lexicon align
## ref: data/lang/phones/align_lexicon.txt
def prosPhonemes(phonemes):
    l_ps = len(phonemes)
    rtn  = []
    for i, p in enumerate(phonemes):
        if i == 0:
            rtn.append("{}_B".format(filterOutNumbers(p).lower()))
        elif i != l_ps -1:
            rtn.append("{}_I".format(filterOutNumbers(p).lower()))
        else:
            rtn.append("{}_E".format(filterOutNumbers(p).lower()))
    return rtn


def wordbreak(s, arpabet):
    s = s.lower()
    if s in arpabet:
        return arpabet[s]
    middle = math.ceil(len(s)/2)
    partition = sorted(list(range(len(s))), key=lambda x: (x-middle)**2-x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in arpabet and wordbreak(suf) is not None:
            return [x+y for x,y in iterprod(arpabet[pre], wordbreak(suf))]
    return None


def findWordsInTrans(false_w, transcript, transcript_ids_dict, transcript_ids, tag):
    inv_map = {v: k for k, v in transcript_ids_dict.items()}
    f_i_ = transcript_ids_dict[false_w]
    r = np.argwhere(transcript_ids==f_i_)
    for i_ps in r.tolist():
        # get word
        if int(transcript_ids[i_ps[0], i_ps[1]+tag*1+(1-tag)*(-1)]) != 0:
            _gt_tn_id = transcript_ids[i_ps[0], i_ps[1]+tag*1+(1-tag)*(-1)]
            guess = inv_map[_gt_tn_id]
            _gt_tn_id = transcript_ids[i_ps[0], i_ps[1]]
            normal = inv_map[_gt_tn_id]
            if st(guess, normal) > 17 and "@" not in guess and "<" not in guess:
                return guess, len(false_w)-1
    return None, None


def opendataSTTorPROMPT(json_data, get_type="stt"):
    assert get_type in ['stt', 'prompt']
    accumulate = set()
    for utt_id, data in json_data['utts'].items():
        if get_type == "sst":
            stt = data.get("input")[1].get("stt").split()
            for t in stt:
                accumulate.add(t)
        elif get_type == "prompt":
            ref = data.get("output")[0].get("token_prompt").split()
            for t in ref:
                accumulate.add(t)
    return sorted(list(accumulate))


# @https://stackoverflow.com/questions/69803370/splitting-words-into-syllables-python
# @https://nedbatchelder.com/code/modules/hyphenate.html
# @https://pyphen.org/
# @https://dictionaryapi.com/
def getFlaseStartEndWordPhonemes(data, arpabet, def_dict, new_dict, transcript, transcript_ids):
    rtn = []
    err = []
    for word in data:

        cut = 0

        ## if word empty
        if word == '':
            continue

        ## if one word only
        if len(word) == 1:
            continue

        ## if in dict already
        if word in def_dict:
            continue

        ## if is non character words
        if word in ["-"]:
            continue

        tag = 0
        if word[-1] == "-":
            tag += 1
            word_ = word[:-1]
        else:
            word_ = word[1:]

        if len(word_) == 1:
            continue

        ## Find word first
        _word, cut = findWordsInTrans(word, transcript, transcript_ids, new_dict, tag)

        ## If word found, replace
        if _word is not None:
            subword = _word
        else:
            subword = word_
        
        # print("word", word)
        # print("_word", _word)
        # print("word_", word_)
        # print("subword", subword)
        # input()

        try:
            if cut is not None and cut > 0:
                rtn.append({"word":word, "substring": subword, "phonemes": prosPhonemes(arpabet[subword][0])[:cut]})
                continue
            else:
                raise Exception
        except:
            try:
                rtn.append({"word":word, "substring": subword, "phonemes": prosPhonemes(wordbreak(subword, arpabet)) })
                continue
            except:
                try:
                    counter = len(word)
                    for i in range(0, counter):
                        if tag == 1:
                            substring = word[0:-i]
                        else:
                            substring = word[i:]
                        try:
                            rtn.append({"word":word, "substring": substring, "phonemes": prosPhonemes(arpabet[substring][0])}) 
                            break
                        except Exception as e:
                            continue
                except Exception as e:
                    err.append(word)
                    continue
    return rtn, err


## START
# try:
#     arpabet = nltk.corpus.cmudict.dict()
# except LookupError:
#     nltk.download('cmudict')
#     arpabet = nltk.corpus.cmudict.dict()
dict_ = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/lang/phones/align_lexicon.txt"
dt_t_ = "/share/nas167/a2y3a1N0n2Yann/tlt-school-chanllenge/kaldi/egs/tlt-school/is2021_data-prep-all_baseline/data/lang_1char/text_all_cleaned"

dt_t_ = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/cefr_train_tr/gigaspeech_20220512_s2t/all.json"
dt_t_ = jsonLoad(dt_t_)['utts']
# dt_ts = opendataSTTorPROMPT(dt_t_, get_type="stt")
# dt_ts = opentext(dt_t_, 1) # get all words, no duplicated words
# dt_tt = genTextOnly(dt_t_, 1)


# ## Variables
# padding = 0


# ## Get Default dict
# dict_ = opendict( dict_ )


# ## Get my own dict
# d_ids = gentokenids(dt_ts) # get ids
# dt_tn = transdict(dt_tt, d_ids) # new dict
# dt_tn_np = np.array(pad_list_n(dt_tn, padding, None))


## Get filtered data
# d = getbyFilter(dt_ts, '9')
# print(d)
# a, b = getFlaseStartEndWordPhonemes(d, arpabet, dict_, dt_tn_np, dt_tt, d_ids)
# print(a)
# print(b)

print(dt_t_['speakerIp16_A2_002004001022-promptIp16_A2_en_22_107_101'].keys())

# save = "text_prompt"
# save_dict = {}
# save_path = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/"
# for data_set, data_set_old in zip(["cefr_train_tr", "cefr_train_cv"], ["train", "train"]): # cerf labels only exist in train data set
#     write_ = open(save_path+data_set+"/"+save, "w")
#     texts_des = open_from_text("/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/"+data_set+"/text", 1)
#     texts_fro = open_from_text("/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/"+data_set_old+"/text", 1)
#     for utt_id in texts_des.keys():
#         get_utt_tokens = texts_fro[utt_id]
#         save_dict[utt_id] = get_utt_tokens
#         write_.write("{} {}\n".format(utt_id, " ".join(get_utt_tokens)))
#     write_.close()

