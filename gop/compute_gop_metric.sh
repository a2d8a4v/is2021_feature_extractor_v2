#!/bin/bash

# Copyright 2017   Author: Ming Tu
# Arguments:
# audio-dir: where audio files are stored
# data-dir: where extracted features are stored
# result-dir: where results are stored                               

set -e
#set -x
stage=0
split_per_speaker=true # split by speaker (true) or sentence (false)
#dataset="L2_kids_phonics_21"
dataset="test_clean"
exp_root=../models
model_name=Librispeech-model-mct-tdnnf
ivectors_dir=$exp_root/$model_name/model_online
#lang=$exp_root/$model_name/data/lang_phonics
lang=$exp_root/$model_name/data/lang
data_root=data
reduce="true"
models=$exp_root/$model_name/model
#lexicon_fn=$exp_root/$model_name/data/local/dict_phonics/lexicon.txt
lexicon_fn=$exp_root/$model_name/data/local/dict/lexicon.txt
ivec_extractor_dir=$exp_root/$model_name/model_online/ivector_extractor
mfcc_conf=$exp_root/$model_name/conf/mfcc_hires.conf

# Enviroment preparation
. ./cmd.sh
. ./path.sh

. parse_options.sh || exit 1;


if [ $stage -le 0 ]; then
    
    for dset in $dataset; do
        
        if [ ! -d $data_root/${dset} ]; then
            utils/fix_data_dir.sh ../corpus/${dset}/data
            utils/copy_data_dir.sh ../corpus/${dset}/data $data_root/${dset}
        fi
        utils/copy_data_dir.sh  $data_root/${dset} $data_root/${dset}_capt_hires
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
        steps/make_mfcc.sh --nj $nspk \
          --mfcc-config $mfcc_conf \
          --cmd "$train_cmd" $data_root/${dset}_capt_hires || exit 1;
        steps/compute_cmvn_stats.sh $data_root/${dset}_capt_hires || exit 1;
        utils/fix_data_dir.sh $data_root/${dset}_capt_hires
    done
fi

if [ $stage -le 1 ]; then
    for dset in $dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
        steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd --num-threads 5" --nj $nspk \
          $data_root/${dset}_capt_hires $ivec_extractor_dir \
          $ivectors_dir/ivectors_${dset}_capt_hires || exit 1;
   done
fi


if [ $stage -le 3 ]; then
    for dset in $dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
        datadir=$data_root/${dset}_capt_hires
        ivectors_data_dir=$ivectors_dir/ivectors_${dset}_capt_hires
        for model in $models; do
            echo "Using DNN model!"
            result_dir=${model}_online/gop_${dset}_capt_hires
            local/gop/compute-dnn-bi-gop.sh --nj "$nspk" --cmd "queue.pl" --split_per_speaker "$split_per_speaker" $datadir $ivectors_data_dir \
              $lang $model $result_dir    ### dnn model
        done
   done
fi


eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
if [ $stage -le 4 ]; then
    for dset in $dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
        datadir=$data_root/${dset}_capt_hires
        ivectors_data_dir=$ivectors_dir/ivectors_${dset}_capt_hires
        for model in $models; do
            echo "LOG parser"
            result_dir=${model}_online/gop_${dset}_capt_hires
            log_dir=${result_dir}/log
            json_dir=${result_dir}/json
            text_fn=$data_root/$dset/text

            if [ ! -d $json_dir ]; then
                mkdir -p $json_dir
            fi            

            python local/gop/gop_log_parser.py --log_dir $log_dir --json_dir $json_dir --words_fn $lang/words.txt --text_fn $text_fn --conf $exp_root/$model_name/sample_worker_en.yaml
        done
   done
fi

if [ $stage -le 5 ]; then
    for dset in $dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
        datadir=$data_root/${dset}_capt_hires
        ivectors_data_dir=$ivectors_dir/ivectors_${dset}_capt_hires
        for model in $models; do
            echo "Calc. correlation"
            result_dir=${model}_online/gop_${dset}_capt_hires
            log_dir=${result_dir}/log
            json_dir=${result_dir}/json
            # parpameters
            json_fn=${result_dir}/json/gop_scores.json
            anno_fn=../corpus/${dset}/data/annotation.txt
            anno_ts_fn=../corpus/${dset}/data/annotation.ts.txt
            text_fn=$data_root/$dset/text
            lexicon_fn=$lexicon_fn

            if [ ! -d $json_dir ]; then
                mkdir -p $json_dir
            fi
            
            python local/gop/gop_json_parser.py --json_fn $json_fn --anno_fn $anno_fn \
                                                --text_fn $text_fn --lexicon_fn $lexicon_fn \
                                                --result_dir $result_dir
            
             
        done
   done
fi

conda deactivate
