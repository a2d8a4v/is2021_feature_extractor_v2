#!/bin/bash

# Copyright (c) 2022 National Taiwan Normal University
# License: Apache 2.0

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=0
stop_stage=10000
specific_scale=None
data_root=data
test_sets="cerf_train_tr cerf_train_cv"

. ./utils/parse_options.sh
set -euo pipefail

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ] ; then
    # filter the data we want
    echo "A. filter the data we want"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then
    echo "B. Prepare train test data"
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set

        python local.apl.v3/utils/prepare_scale_feats.py \
            --input_text_file_path $data_dir/text \
            --input_cefr_label_file_path $data_dir/scale \
            --output_text_file_path $data_dir/text.tsv \
            --remove_filled_pauses false
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ] ; then
    echo "C. Prepare all data"
    for test_set in all; do
        mkdir -pv data/${test_set} > /dev/null 2>&1
    done

    cefr_scores_names=""
    text_names=""
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        cefr_scores_names+="${data_dir}/scale "
        text_names+="${data_dir}/text "
    done

    for test_set in all; do
        data_dir=$data_root/$test_set

        cat $text_names > $data_dir/text
        cat $cefr_scores_names > $data_dir/scale

        python local.apl.v3/utils/prepare_scale_feats.py \
            --input_text_file_path $data_dir/text \
            --input_cefr_label_file_path $data_dir/scale \
            --output_text_file_path $data_dir/text.tsv \
            --remove_filled_pauses false
    done
fi