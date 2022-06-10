#!/usr/bin/env python3

import argparse
import os
from espnet.utils.cli_utils import strtobool
import disfluency.fisher_annotator as fisher_annotator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--output-pickle-path", type=str, required=True)
    parser.add_argument("--save-pickle", type=strtobool, required=True)
    parser.add_argument("--model", type=str, default="./model/swbd_fisher_bert_Edev.0.9078.pt")
    parser.add_argument("--disfluency", type=strtobool, default=False)
    parser.add_argument("--add-err-rate", type=strtobool, default=False)
    parser.add_argument("--get-annotations", type=strtobool, default=True)
    args = parser.parse_args()

    labels = fisher_annotator.Annotate(
        input_json=args.input_json,
        output_path=args.output_path,
        save_pickle=args.save_pickle,
        model=args.model,
        disfluency=args.disfluency,
        output_pickle_path=args.output_pickle_path,
        add_err_rate=args.add_err_rate,
        get_annotations=args.get_annotations
    )
    labels.setup()

if __name__ == "__main__":
    main()
