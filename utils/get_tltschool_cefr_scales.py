import argparse

def argparse_function():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_wavscp_file_path",
                        default='data/trn/text',
                        type=str)

    parser.add_argument("--output_cefr_file_path",
                    default='CEFR_LABELS_PATH/trn_cefr_scores.txt',
                    type=str)


    args = parser.parse_args()

    return args

def open_utt2cefr( file, col, filter ):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            utt_id  = l.split()[col]
            if sum([1 for cefr in filter if cefr.lower() in utt_id.lower()]) == 0:
                continue
            utt_id_ = [ str(w) for w in utt_id.split("-")[1].split("_") ]
            # speakerIp18_A2_002002001008-promptIp18_A2_en_22_20_103
            if sum([1 for cefr in filter if cefr in utt_id_]) > 0:
                s[utt_id] = utt_id_[1]
    return s

if __name__ == '__main__':

    # argparse
    args = argparse_function()
    input_wavscp_file_path = args.input_wavscp_file_path
    output_cefr_file_path = args.output_cefr_file_path

    # CEFR score filter
    CEFR_filter = ["A1", "A2", "B1"]
    utt2cefr_dict = open_utt2cefr(
        input_wavscp_file_path,
        0,
        CEFR_filter
    )

    # save
    with open(output_cefr_file_path, "w") as f:
        for utt_id, cefr in utt2cefr_dict.items():
            f.write("{} {}\n".format(utt_id, cefr))
