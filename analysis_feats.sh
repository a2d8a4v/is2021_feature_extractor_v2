#!/bin/bash

# Copyright (c) 2022 National Taiwan Normal University
# License: Apache 2.0

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=0
stop_stage=10000
test_sets="cerf_train_tr cerf_train_cv"
model_name="librispeech"
tag="20220617_gop_s2t"
data_root=data
dumpdir=dump

. ./utils/parse_options.sh
set -euo pipefail

if [ ! -z ${tag} ]; then
    model_name=${model_name}_${tag}
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ] ; then    
    # prepare training data
    echo "A. Get the features from the output json files"
    for test_set in $test_sets; do

        data_dir=$data_root/$test_set
        dest_dir=$data_dir/$model_name
        output_dir=$dest_dir/analysis
        mkdir -pv $output_dir > /dev/null 2>&1

        echo "Retrieve feats from $test_set..."
        python local.apl.v3/utils/json2csv_feats.py \
            --input_json_file_path $dest_dir/all.json \
            --input_scale_file_path $data_dir/scale \
            --save_csv_file_path $output_dir/apl_features_${model_name}_${test_set}.csv
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then    
    echo "B. Analysis feats"
    for test_set in $test_sets; do

        data_dir=$data_root/$test_set
        dest_dir=$data_dir/$model_name
        output_dir=$dest_dir/analysis
        mkdir -pv $output_dir > /dev/null 2>&1

        echo "Analysis feats for $test_set..."
        python local.apl.v3/analysis/get_rsquared_rmse_other_coefficients.py \
            --input_csv_file_path $output_dir/apl_features_${model_name}_${test_set}.csv \
            --output_rmse_file_path $output_dir/apl_features_${model_name}_${test_set}_rmse_list \
            --output_rsquared_file_path $output_dir/apl_features_${model_name}_${test_set}_rsquared_list \
            --output_accuracy_file_path $output_dir/apl_features_${model_name}_${test_set}_accuracy_list
    done
fi