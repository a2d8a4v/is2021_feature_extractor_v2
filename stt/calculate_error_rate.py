#!/usr/bin/env python3

import argparse
import os
from espnet.utils.cli_utils import strtobool
import disfluency.fisher_annotator as fisher_annotator
from error_rate_models import ErrorRateModel
from utils import (
    jsonLoad,
    jsonSave,
    process_tltchool_gigaspeech_interregnum_tokens,
    remove_gigaspeech_interregnum_tokens
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--output-pickle-path", type=str, required=False)
    parser.add_argument("--save-pickle", type=strtobool, required=False)
    parser.add_argument("--model", type=str, default="./model/swbd_fisher_bert_Edev.0.9078.pt")
    parser.add_argument("--disfluency", type=strtobool, default=True)
    parser.add_argument("--add-err-rate", type=strtobool, default=True)
    parser.add_argument("--get-annotations", type=strtobool, default=False)
    args = parser.parse_args()

    # get the annotator of disfluency
    labels = fisher_annotator.Annotate(
        input_json=args.input_json,
        output_path=args.output_path+".pos_dis_labels",
        save_pickle=args.save_pickle,
        model=args.model,
        disfluency=args.disfluency,
        output_pickle_path=args.output_pickle_path,
        add_err_rate=args.add_err_rate,
        get_annotations=args.get_annotations
    )
    labels.setup()

    dict_json = jsonLoad(args.input_json)['utts']
    pos_dis_json = jsonLoad(args.output_path+".pos_dis_labels")['utts']
    save_json = {'utts':{}}
    predicts = []
    labels = []
    err_model = ErrorRateModel()
    for utt_id, data in dict_json.items():
        ref = remove_gigaspeech_interregnum_tokens(
            process_tltchool_gigaspeech_interregnum_tokens(
                data.get("output")[0].get("token")
            )
        )
        hyp = remove_gigaspeech_interregnum_tokens(
            data.get("input")[1].get("stt")
        )
        hyp_dis = remove_gigaspeech_interregnum_tokens(
            pos_dis_json.get(utt_id).get('pos_dis_labels').get('df_label')
        )
        predicts.append(hyp)
        labels.append(ref)
        fer_der = err_model.fer_der(hyp_dis, ref)
        '''
        Vefore calculating word error rate, we have better to remove all the interregnum inside all the predictions and labels.
        '''
        save_json['utts'][utt_id]= {
            'error_rate': {
                'wer': err_model.wer(hyp, ref),
                'fer': fer_der[0],
                'der': fer_der[1]
            }
        }
    jsonSave(save_json, args.output_path)
    print("The total word error rate for {} is {}".format(args.input_json, err_model.wer(predicts, labels)))


if __name__ == "__main__":
    main()
