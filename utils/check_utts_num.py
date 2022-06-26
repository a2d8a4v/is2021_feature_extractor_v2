import json
import sys
import argparse
from utilities import (
    jsonLoad,
)

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_file",
                        default="data/train/all.json",
                        type=str)

    args = parser.parse_args()

    # varaibles
    file_path = args.json_file
    json_data = jsonLoad(file_path)

    if 'utts' in json_data:
        json_data = json_data.get('utts')
    
    print("There are {} uttids insidie the {} file".format(
        len(json_data.keys())),
        file_path
    )