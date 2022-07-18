import sys
import csv
import argparse

columns_with_positions = ['id', 'score', 'sst', 'l1', 'text']

def argparse_function():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_tsv_files_path",
                        default='data/trn/text.tsv',
                        nargs='+',
                        type=str)

    parser.add_argument("--output_text_file_path",
                    default='CEFR_LABELS_PATH/trn_cefr_scores.txt',
                    type=str)

    args = parser.parse_args()

    return args

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines

def create_examples(lines):
    """Creates examples for the training and dev sets."""
    lines_count = len(lines)-1
    data_dict = {}
    columns = {}
    for (i, line) in enumerate(lines):
        if i == 0:
            columns = {key:header_index for header_index, key in enumerate(line)}
            continue
        for item in columns.keys():
            data_dict.setdefault(item, []).append(line[columns[item]])
    return lines_count, data_dict

## Data preparation
"""
The experiments in our work were performed on responses collected to Cambridge Assessment's [BULATs examination](https://www.cambridgeenglish.org/exams-and-tests/bulats), which is not publicly available. However, you can provide any TSV file (containing a header) of transcriptions containing the following columns:
- text (required): the transcription of the speech (spaces are assumed to signify tokenization)
- score (required): the numerical score assigned to the speech (by default, a scale between 0 - 6 is used to match CEFR proficiency levels)
- pos (optional): Penn Treebank part of speech tags. These should be space-separated and aligned with a token in text (i.e. there should be an identical number of tokens and POS tags)
- deprel (optional): Universal Dependency relation to head/parent token. These should be space-separated and aligned with a token in text (i.e. there should be an identical number of tokens and Universal Dependency relation labels)
- l1 (optional): native language/L1 of the speaker. Our experiments included L1 speakers of Arabic, Dutch, French, Polish, Thai and Vietnamese.
"""

if __name__ == '__main__':

    # argparse
    args = argparse_function()

    # variables
    input_tsv_files_path = args.input_tsv_files_path
    output_text_file_path = args.output_text_file_path

    input_columns_list = []
    collect_data_dict = {}
    lines_count_list = []

    # combine files
    for i, file_path in enumerate(input_tsv_files_path):
        lines_count, get_file_content_dict = create_examples(read_tsv(file_path))
        lines_count_list.append(lines_count)
        get_file_columns = list(get_file_content_dict.keys())
        input_columns_list.append(get_file_columns)
        collect_data_dict.setdefault(i, get_file_content_dict)

    # check columns in all files are the same, insensitive case of postions of the columns
    count_list = list(set([ item for column_list in input_columns_list for item in column_list ] + columns_with_positions)).sort()
    assert count_list == columns_with_positions.sort(), 'columns in different input tsv has no equal columns'
    count_list = sorted(list(map(len, list(map(set, input_columns_list)))))
    assert count_list[0] == sorted(count_list)[-1], 'columns in different input tsv are missing'

    max_seq_len = 0
    with open(output_text_file_path, 'w') as f:
        f.write("{}\n".format(
                "\t".join(columns_with_positions)
            )
        )
        for i, data_info_dict in collect_data_dict.items():
            get_line_count = lines_count_list[i]
            for j in range(0, get_line_count, 1):
                line_content_list = []
                for item in columns_with_positions:

                    if item == 'text':
                        text_len = len(data_info_dict[item][j].split())
                        if text_len > max_seq_len:
                            max_seq_len = text_len

                    line_content_list.append(
                        data_info_dict[item][j]
                    )
                f.write("{}\n".format(
                        "\t".join(line_content_list)
                    )
                )

    if max_seq_len > 0:
        print("Max length from all sequences is {}".format(max_seq_len))

        

