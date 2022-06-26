## Functions
def opentext( file, col, filter ):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            utt_id  = l.split()[col]
            if sum([1 for cefr in filter if cefr.lower() in utt_id.lower()]) == 0:
                continue
            utt_id_ = [ str(w).lower() for w in utt_id.split("-")[1].split("_") ]
            # speakerIp18_A2_002002001008-promptIp18_A2_en_22_20_103
            if sum([1 for cefr in filter if cefr.lower() in utt_id_]) > 0:
                s[utt_id] = utt_id_[1]
    return s


## CEFR score filter
_filter = ["a1", "b1", "a2"]

## Data
_data = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/train/text"
data = opentext( _data, 0, _filter )

## Save
with open("/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/train/text_cefr", "w") as f:
    for utt_id, cefr in data.items():
        f.write("{} {}\n".format(utt_id, cefr))
