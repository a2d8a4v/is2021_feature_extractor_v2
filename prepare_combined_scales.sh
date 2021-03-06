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
main_corpus_data_root="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data"
main_test_sets="cerf_train_tr cerf_train_cv"
other_corpus_data_root="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/nict_jle/asr3/data"
s2t=false

. ./utils/parse_options.sh
set -euo pipefail

main_model_name=$model_name
if [ ! -z ${tag} ]; then
    main_model_name=${main_model_name}_${tag}
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ] ; then
    # filter the data we want
    echo "A. Get CEFR scale label from utterance ids, and their mother tongue"

    for test_set in $main_test_sets; do
        data_dir=$main_corpus_data_root/$test_set

        python local.apl.v3/utils/get_tltschool_cefr_scales.py \
            --input_wavscp_file_path $data_dir/wav.scp \
            --output_cefr_file_path $data_dir/scale

        if [ -f $data_dir/momlanguage ]; then
            rm $data_dir/momlanguage
        fi
        touch $data_dir/.momlanguage
        lines=$(cat $data_dir/wav.scp | wc -l)
        for j in $(seq 1 $lines); do
            echo 'italiano' >> $data_dir/.momlanguage
        done
        cat $data_dir/wav.scp | cut -f1 -d' ' > $data_dir/utt_list
        paste -d ' ' $data_dir/utt_list $data_dir/.momlanguage > $data_dir/momlanguage
        rm $data_dir/utt_list $data_dir/.momlanguage
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then
    echo "B. Prepare train test data of main data"
    for test_set in $main_test_sets; do
        data_dir=$main_corpus_data_root/$test_set
        dest_dir=$data_dir/$main_model_name

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
    echo "C. Prepare train data"
    for test_set in cerf_train_tr; do
        main_data_dir=$main_corpus_data_root/$test_set
        other_data_dir=$other_corpus_data_root/trn
        dest_data_dir=$main_corpus_data_root/trn_combo
        mkdir -pv $dest_data_dir > /dev/null 2>&1

        python local.apl.v3/utils/combine_auto_grader_feats.py \
            --input_tsv_files_path $main_data_dir/text.tsv $other_data_dir/text.tsv \
            --output_text_file_path $dest_data_dir/text.tsv
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ] ; then
    echo "D. Prepare dev data"
    for test_set in cerf_train_cv; do
        main_data_dir=$main_corpus_data_root/$test_set
        other_data_dir=$other_corpus_data_root/dev
        dest_data_dir=$main_corpus_data_root/dev_combo
        mkdir -pv $dest_data_dir > /dev/null 2>&1

        python local.apl.v3/utils/combine_auto_grader_feats.py \
            --input_tsv_files_path $main_data_dir/text.tsv $other_data_dir/text.tsv \
            --output_text_file_path $dest_data_dir/text.tsv
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ] ; then
    echo "E. Prepare test data"
    for test_set in cerf_train_cv; do
        main_data_dir=$main_corpus_data_root/$test_set
        other_data_dir=$other_corpus_data_root/eval
        dest_data_dir=$main_corpus_data_root/eval_combo
        mkdir -pv $dest_data_dir > /dev/null 2>&1

        python local.apl.v3/utils/combine_auto_grader_feats.py \
            --input_tsv_files_path $main_data_dir/text.tsv $other_data_dir/text.tsv \
            --output_text_file_path $dest_data_dir/text.tsv
    done
fi
