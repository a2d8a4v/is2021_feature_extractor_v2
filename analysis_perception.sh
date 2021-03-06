#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=0
stop_stage=3
test_sets="cerf_train_tr cerf_train_cv"
model_name="librispeech"
tag="20220617_prompt"
graph_affix=_tgt3
data_root=data
CUDA=
max_nj_cuda=20

# cmd=queue.pl
cmd=run.pl

echo "$0 $@"
. parse_options.sh

set -euo pipefail

if [ ! -z ${tag} ]; then
    model_name=${model_name}_${tag}
fi

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for test_set in $test_sets; do

        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        
        if [ $nspk -ge $max_nj_cuda ]; then
            nspk=$max_nj_cuda;
        fi

        data_dir=$data_root/$test_set
        dest_dir=$data_dir/$model_name
        output_dir=$dest_dir/analysis
        mkdir -pv $output_dir > /dev/null 2>&1

        echo "split $test_set into $nspk pieces..."
        python local.apl.v3/utils/split_uttids.py --tmp_decoding_list_file_path $data_dir/tmp_apl_decoding_${tag}.list \
                                                  --output_dir_path $output_dir \
                                                  --split_number $nspk
    done
fi

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for test_set in $test_sets; do

        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        
        if [ $nspk -ge $max_nj_cuda ]; then
            nspk=$max_nj_cuda;
        fi

        data_dir=$data_root/$test_set
        dest_dir=$data_dir/$model_name
        output_dir=$dest_dir/analysis
        logdir=$output_dir/log

        echo "start to statistic the F1 and F2 in $test_set..."
        $cmd JOB=1:$nspk $logdir/analysis_perception.JOB.log \
            CUDA_VISIBLE_DEVICES=${CUDA} python local.apl.v3/analysis/get_vowel_perception.py \
                --input_file $output_dir/utt_pkl.JOB.list \
                --lexicon_file_path $output_dir/lexicon \
                --output_file_path $output_dir/analysis_perception_output.JOB.pkl \
                --phn_from_data true
    done
fi

if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for test_set in $test_sets; do

        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        
        if [ $nspk -ge $max_nj_cuda ]; then
            nspk=$max_nj_cuda;
        fi

        data_dir=$data_root/$test_set
        dest_dir=$data_dir/$model_name
        output_dir=$data_dir/$model_name/analysis

        all_json_files_path=""
        for i in $(seq 1 $nspk); do
            all_json_files_path+="$output_dir/analysis_perception_output.$i.pkl "
        done

        echo "start to combine output in $test_set..."
        python local.apl.v3/utils/combine_pickles.py \
            --input_json_files $all_json_files_path \
            --output_file_path $output_dir/analysis_perception_output.pkl
    done
fi

if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    for test_set in $test_sets; do

        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        
        if [ $nspk -ge $max_nj_cuda ]; then
            nspk=$max_nj_cuda;
        fi

        data_dir=$data_root/$test_set
        dest_dir=$data_dir/$model_name
        output_dir=$data_dir/$model_name/analysis

        echo "illustrate vowel perception image and save in $test_set..."
        python local.apl.v3/utils/draw_vowel_ellipse.py \
                --input_file_path $output_dir/analysis_perception_output.pkl \
                --output_dir_path $output_dir \
                --combine_to_basic_vowels true
    done
fi

echo "successful illustrate vowel perception"