def openctm(file):
    s = set()
    with open(file, "r") as f:
        for l in f.readlines():
            s.add(l.split()[4])
    return list(s)


def getbyFilter(data, filter):
    rtn = [i for i in data if filter in i]
    return sorted(list(set(rtn)))


## START
_ctm_file = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/acoustic_phonetic_features/data/train/gigaspeech/ctm"
words_ctm_ = openctm(_ctm_file)
d = getbyFilter( words_ctm_, '-' )

print(d)
