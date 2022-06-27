import json
import sys
import argparse
from utilities import (
    jsonLoad,
    jsonSave
)

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_file_path",
                        default="data/train/all.json",
                        type=str)

    parser.add_argument("--save_json_file_path",
                        default="data/train/all.json",
                        type=str)

    args = parser.parse_args()

    # varaibles
    json_file_path = args.json_file_path
    save_json_file_path = args.save_json_file_path
    json_data = jsonLoad(json_file_path)

    if 'utts' not in json_data:
        json_data = {'utts': json_data}
        jsonSave(json_data, save_json_file_path)
    else:
        print("we do not need to process this file.")