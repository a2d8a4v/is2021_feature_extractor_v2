#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=0
stop_stage=10000
test_sets="cerf_train_tr cerf_train_cv"
model_dir="Librispeech-model-mct-tdnnf"
model_name="librispeech"
tag="20220617_prompt"
graph_affix=_tgt3
data_root=data
CUDA=
max_nj_cuda=20

cmd=queue.pl
# cmd=run.pl

echo "$0 $@"
. parse_options.sh

set -euo pipefail

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for test_set in $data_sets; do

        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        
        if [ $nspk -ge $max_nj ]; then
            nspk=$max_nj_cuda;
        fi

        data_dir=$data_root/$test_set
        dest_dir=$dest_dir/$model_name

        $cmd JOB=1:$nspk $logdir/analysis_perception.JOB.log \
            python local.apl.v3/analysis_utils/get_F1_F2_perception.py \
                --input_json $dest_dir/all.JOB.json \
                --input_dict $data_root/local/dict/lexicon.txt \
                --output_file_path analysis_perception_output.JOB.pkl \
                --phn_from_data true
    done
fi