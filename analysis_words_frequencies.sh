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
remove_stop_tokens=false
model_name="librispeech"
tag="20220617_prompt"
data_root=data
test_sets="cerf_train_tr cerf_train_cv"
s2t=false
sort_by=value

. ./utils/parse_options.sh
set -euo pipefail

if [ ! -z ${tag} ]; then
    model_name=${model_name}_${tag}
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ] ; then
    echo "A. Prepare train test data"
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        dest_dir=$data_dir/$model_name
        output_dir=$dest_dir/analysis
        mkdir -pv $output_dir > /dev/null 2>&1

        python local.apl.v3/analysis/get_words_frequencies.py \
            --s2t $s2t \
            --sort_by $sort_by \
            --input_json_file_path $dest_dir/all.json \
            --input_text_file_path $data_dir/text \
            --input_spk2utt_file_path $data_dir/spk2utt \
            --input_cefr_label_file_path $data_dir/scale \
            --input_spk2momlang_file_path $data_dir/momlanguage \
            --output_text_file_path $output_dir/word_frequency \
            --remove_filled_pauses $remove_filled_pauses \
            --remove_stop_tokens $remove_stop_tokens \
            --combine_same_speakerids $combine_same_speakerids

        python local.apl.v3/utils/draw_word_freq_chart.py \
            --input_word_frequency_file_path $output_dir/word_frequency \
            --output_file_path $output_dir/word_frequency.png
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then
    echo "B. Prepare all data"
    for test_set in all; do
        mkdir -pv data/${test_set} > /dev/null 2>&1
    done

    momlang_names=""
    cefr_scores_names=""
    text_names=""
    sp2utt_names=""
    json_names=""
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        dest_dir=$data_dir/$model_name
        momlang_names+="${data_root}/momlanguage "
        cefr_scores_names+="${data_dir}/scale "
        text_names+="${data_dir}/text "
        sp2utt_names+="${data_dir}/spk2utt "
        json_names+="${dest_dir}/all.json "
    done

    for test_set in all; do
        data_dir=$data_root/$test_set
        dest_dir=$data_dir/$model_name
        output_dir=$dest_dir/analysis
        mkdir -pv $output_dir > /dev/null 2>&1

        cat $text_names > $data_dir/text
        cat $cefr_scores_names > $data_dir/scale
        cat $sp2utt_names > $data_dir/spk2utt

        python local.apl.v3/utils/combine_jsons.py \
            --input_jsons_file_path $json_names \
            --output_file_path $dest_dir/all.json

        python local.apl.v3/analysis/get_words_frequencies.py \
            --s2t $s2t \
            --sort_by $sort_by \
            --input_json_file_path $dest_dir/all.json \
            --input_text_file_path $data_dir/text \
            --input_spk2utt_file_path $data_dir/spk2utt \
            --input_spk2momlang_file_path $data_dir/momlanguage \
            --input_cefr_label_file_path $data_dir/scale \
            --output_text_file_path $dest_dir/word_frequency \
            --remove_filled_pauses $remove_filled_pauses \
            --remove_stop_tokens $remove_stop_tokens \
            --combine_same_speakerids $combine_same_speakerids

        python local.apl.v3/utils/draw_word_freq_chart.py \
            --input_word_frequency_file_path $output_dir/word_frequency \
            --output_file_path $output_dir/word_frequency.png
    done
fi
