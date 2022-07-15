import argparse

import numpy as np
from espnet.utils.cli_utils import strtobool
from utilities import (
    open_utt2value
)

def plotSaveConts(word_freq_dict, output_file_path, color_list):
    
    # Data set
    counts = list(word_freq_dict.values())
    words = tuple(word_freq_dict.keys())
    y_pos = np.arange(len(words))

    # Basic bar plot
    plt.bar(y_pos, counts, color=color_list)
    
    # Custom Axis title
    plt.xlabel('Word Frequency', fontweight='bold', color = 'orange', fontsize='17', horizontalalignment='center')

    plt.savefig(output_file_path)


if __name__ == '__main__':

    # parsers
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_word_frequency_file_path', default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/cefr_train_tr/gigaspeech_20220525_prompt/all.json", type=str)
    parser.add_argument('--output_file_path', default="./test.pkl", type=str)
    parser.add_argument('--select_bigger_than', default=10, type=int, required=True)
    args = parser.parse_args()

    # variables
    input_word_frequency_file_path = args.input_word_frequency_file_path
    output_file_path = args.output_file_path
    select_bigger_than = args.select_bigger_than

    assert select_bigger_than <= 17, 'select_bigger_than can only up to 17.'

    word_frequency_dict = open_utt2value(input_word_frequency_file_path)

    # color
    color_list = [
        '#003f5c',
        '#58508d',
        '#bc5090',
        '#ff6361',
        '#ffa600',
        '#00ff00',
        '#feb300',
        '#ff0000',
        '#007900',
        '#ba00ff',
        '#535f2c',
        '#aac9a8',
        '#e9a2a5',
        '#d73068',
        '#ffb06d',
        '#fde398',
        '#563432'
    ]

    # draw the chart
    plotSaveConts(
        word_frequency_dict,
        output_file_path
        color_list[:select_bigger_than]
    )

    print('Your word frequency chart is saved at {}'.format(output_file_path))