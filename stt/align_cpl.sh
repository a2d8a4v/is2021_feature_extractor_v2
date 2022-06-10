#!/bin/bash

# Copyright 2017   Author: Ming Tu
# Arguments:
# audio-dir: where audio files are stored
# data-dir: where extracted features are stored
# result-dir: where results are stored                               

set -e
#set -x
stage=0
dataset="train_tr train_cv test"
data_root=data/speechocean
exp_root=../../models/Librispeech-model-tdnnf
reduce="true"
models="$exp_root/model"
ivec_extractor_dir=$exp_root/extractor
config_dir=../../models/Librispeech-model-tdnnf/conf
ivectors_dir=../../models/Librispeech-model-tdnnf/model_online
lang=../../models/Librispeech-model-tdnnf/data/lang_ext

# Enviroment preparation
. ./cmd.sh
. ./path.sh

. parse_options.sh || exit 1;


if [ $stage -le 0 ]; then
    
    for dset in $dataset; do
        
        utils/copy_data_dir.sh  $data_root/${dset} $data_root/${dset}_hires
         
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 6 ]; then
            nspk=6;
        fi
        
        steps/make_mfcc.sh --nj $nspk \
          --mfcc-config $config_dir/mfcc_hires.conf \
          --cmd "$train_cmd" $data_root/${dset}_hires || exit 1;
        steps/compute_cmvn_stats.sh $data_root/${dset}_hires || exit 1;
        utils/fix_data_dir.sh $data_root/${dset}_hires
    done
fi

if [ $stage -le 1 ]; then
    for dset in $dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 6 ]; then
            nspk=6;
        fi
        steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd --num-threads 5" --nj $nspk \
          $data_root/${dset}_hires $ivec_extractor_dir \
          $ivectors_dir/ivectors_${dset}_hires || exit 1;
   done
fi


if [ $stage -le 2 ]; then
    for dset in $dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 6 ]; then
            nspk=6;
        fi
        data_dir=$data_root/${dset}_hires
        ivectors_data_dir=$ivectors_dir/ivectors_${dset}_hires
        for model in $models; do
            echo "Align $data_dir with $model"
            result_dir=${model}/align_${dset}_hires
            # steps/chain/align_lats_ctm.sh <data-dir> <lang-dir> <src-dir> <align-dir>
            steps/chain/align_lats_ctm.sh --cmd "queue.pl" --nj $nspk --online-ivector-dir  $ivectors_data_dir $data_dir $lang $model $result_dir
        done
   done
fi


eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"

if [ $stage -le 3 ]; then
    feats_dir=phn_id
    if [ ! -d $feats_dir ]; then
        mkdir -p $feats_dir
    fi
    
    for dset in $dataset; do
        data_dir=$data_root/${dset}
        phnid_data_dir=$data_root/${dset}_phnid
        ivectors_data_dir=$ivectors_dir/ivectors_${dset}_hires
        words_txt=$lang/words.txt
        
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 6 ]; then
            nspk=6;
        fi
        
        if [ ! -d $phnid_data_dir ]; then
            utils/copy_data_dir.sh $data_dir $phnid_data_dir
        fi
        
        for model in $models; do
            echo "Process alignment information"
            #result_dir=${model}/align_${dset}_hires
            ctm_fn=${model}/align_${dset}_hires/ctm_word/ctm
            
            python local/align/process_phone_id_from_align.py --ctm_fn $ctm_fn \
                            --phn_tbl_fn $words_txt --data_dir $phnid_data_dir \
                            --new_feats_dir phn_id
            cat phn_id/phn_id*${dset}*scp > $data_root/${dset}_phnid/feats.scp
            utils/fix_data_dir.sh $data_root/${dset}_phnid
        done
   done
fi

conda deactivate

if [ $stage -le 4 ]; then
    feats_dir=phn_id
    if [ ! -d $feats_dir ]; then
        mkdir -p $feats_dir
    fi
    
    for dset in $dataset; do
        data_dir=$data_root/${dset}
        phnid_data_dir=$data_root/${dset}_phnid
        dest_data_dir=$data_root/${dset}_fbank_phnid
        ivectors_data_dir=$ivectors_dir/ivectors_${dset}_hires
        words_txt=$lang/words.txt
        
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 6 ]; then
            nspk=6;
        fi
        
        steps/append_feats.sh --nj $nspk --cmd "queue.pl" $data_dir $phnid_data_dir $dest_data_dir exp/make_fbank_phnid/log fbank_phnid
   done
fi
