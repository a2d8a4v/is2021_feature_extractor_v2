#!/usr/bin/env python3
# encoding: utf-8

import argparse
import json
import os
from utilities import (
    jsonLoad,
    jsonSave
)

def get_parser():
    parser = argparse.ArgumentParser(
        description="merge json files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_jsons_file_path",
        type=str,
        nargs="+",
        action="append",
        default=[],
        help="The json files except for the input and outputs",
    )
    parser.add_argument("--output_file_path", type=str, help="Output json file")
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    # variables
    input_jsons_file_path = args.input_jsons_file_path
    output_file_path = args.output_file_path
    json_utt_dict = {}

    # merge
    for json_files_path in input_jsons_file_path:
        for json_file_path in json_files_path:
            if os.path.isfile(json_file_path):
                j = jsonLoad(json_file_path)
                json_utt_dict.update(j['utts'])

    # save
    jsonSave(json_utt_dict, output_file_path)