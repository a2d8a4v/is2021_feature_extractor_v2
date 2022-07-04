#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=0
stop_stage=3
test_sets="cerf_train_tr cerf_train_cv"
tag="20220617_prompt"
data_root=data
scale_select=A1
CUDA=

# cmd=queue.pl
cmd=run.pl

echo "$0 $@"
. parse_options.sh

set -euo pipefail


if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for test_set in $test_sets; do

        data_dir=$data_root/$test_set
        tmp_decoding_file_path=$data_dir/tmp_apl_decoding_${tag}.list
        output_utts_list_file_path=$data_dir/scale_${scale_select}_utts_list

        echo "filter the $scale_select from $test_set..."
        python local.apl.v3/utils/get_utts_filter_by_scale.py \
            --input_tmp_decoding_list_file_path $tmp_decoding_file_path \
            --input_scale_file_path $data_dir/scale \
            --filter_scale $scale_select \
            --output_file_path $output_utts_list_file_path
    done
fi

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for test_set in $test_sets; do

        data_dir=$data_root/$test_set
        input_utts_list_file_path=$data_dir/scale_${scale_select}_utts_list
        input_tmp_decoding_file_path=$data_dir/tmp_apl_decoding_${tag}.list

        echo "count the disfluency from $test_set..."
        python local.apl.v3/utils/get_utts_disfluency_counts.py \
            --input_utts_list_file_path $input_utts_list_file_path \
            --input_tmp_decoding_list_file_path $input_tmp_decoding_file_path \
            --scale $scale_select
    done
fi
