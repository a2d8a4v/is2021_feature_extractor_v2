#!/bin/bash

# Copyright (c) 2022 National Taiwan Normal University
# License: Apache 2.0

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=0
stop_stage=10000
specific_scale=None
combine_same_speakerids=false
remove_filled_pauses=false
model_name="librispeech"
tag="20220617_prompt"
data_root=data
test_sets="cerf_train_tr cerf_train_cv"
s2t=false

. ./utils/parse_options.sh
set -euo pipefail

if [ ! -z ${tag} ]; then
    model_name=${model_name}_${tag}
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ] ; then
    # filter the data we want
    echo "A. filter the data we want"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then
    echo "B. Prepare train test data"
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        dest_dir=$data_dir/$model_name


        python local.apl.v3/utils/prepare_auto_grader_feats.py \
            --s2t $s2t \
            --input_json_file_path $dest_dir/all.json \
            --input_text_file_path $data_dir/text \
            --input_spk2utt_file_path $data_dir/spk2utt \
            --input_cefr_label_file_path $data_dir/scale \
            --input_spk2momlang_file_path $data_dir/momlanguage \
            --output_text_file_path $data_dir/text.tsv \
            --remove_filled_pauses $remove_filled_pauses \
            --combine_same_speakerids $combine_same_speakerids
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ] ; then
    echo "C. Prepare all data"
    for test_set in all; do
        mkdir -pv data/${test_set} > /dev/null 2>&1
    done

    momlang_names=""
    cefr_scores_names=""
    text_names=""
    sp2utt_names=""
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        momlang_names+="${data_root}/momlanguage "
        cefr_scores_names+="${data_dir}/scale "
        text_names+="${data_dir}/text "
        sp2utt_names+="${data_dir}/spk2utt "
    done

    for test_set in all; do
        data_dir=$data_root/$test_set

        cat $text_names > $data_dir/text
        cat $cefr_scores_names > $data_dir/scale
        cat $sp2utt_names > $data_dir/spk2utt

        python local.apl.v3/utils/prepare_auto_grader_feats.py \
            --s2t $s2t \
            --input_json_file_path $dest_dir/all.json \
            --input_text_file_path $data_dir/text \
            --input_spk2utt_file_path $data_dir/spk2utt \
            --input_spk2momlang_file_path $data_dir/momlanguage \
            --input_cefr_label_file_path $data_dir/scale \
            --output_text_file_path $data_dir/text.tsv \
            --remove_filled_pauses $remove_filled_pauses \
            --combine_same_speakerids $combine_same_speakerids
    done
fi
