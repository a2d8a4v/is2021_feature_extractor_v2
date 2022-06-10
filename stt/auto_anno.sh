#!/usr/bin/env bash


set -e
#set -x
stage=0
dataset="test_clean"
graph_affix=_tgsmall
model_dir=../models/Librispeech-model-mct-tdnnf
data_root=data

# Enviroment preparation
. ./cmd.sh
. ./path.sh

. parse_options.sh || exit 1;

model=$model_dir/model
ivec_extractor=$model_dir/extractor
lang=$model_dir/data/lang
graph_dir=$model_dir/model/graph${graph_affix}
mfcc_config=$model_dir/conf/mfcc_hires.conf

if [ $stage -le 0 ]; then  
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
    echo "preparing online config"
    if [ ! -d ${model}_online ]; then
        steps/online/nnet3/prepare_online_decoding.sh \
           --mfcc-config $mfcc_config \
           $lang $ivec_extractor $model ${model}_online
    fi
fi

if [ $stage -le 1 ]; then
    echo "online decoding"
    for dset in $dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
        decode_dir=${model}_online/decode_${dset}${graph_affix}
        steps/online/nnet3/decode.sh \
            --acwt 1.0 --post-decode-acwt 10.0 \
            --nj $nspk --cmd "$decode_cmd" \
            $graph_dir $data_root/${dset} ${decode_dir} || exit 1
   done
fi


if [ $stage -le 2 ]; then
    echo "get ctm file";
    for dset in $dataset; do
        nspk=$(wc -l <$data_root/$dset/spk2utt)
        if [ $nspk -ge 20 ]; then
            nspk=20;
        fi
        decode_dir=${model}_online/decode_${dset}${graph_affix}
        steps/get_ctm_conf_added_phn.sh --cmd "$decode_cmd" $data_root/$dset $lang $decode_dir
   done
fi

