#!/usr/bin/env bash

set -euo pipefail

data_root=data
data_sets="cerf_train_tr cerf_train_cv"
verbose=0
CUDA=
stage=0
stop_stage=
tag=
dumpdir=dump
datajsoninput=data.new.json
datajsonoutput=data.new.err_rate.json
s2t=
s2t_label=

echo "$0 $@"
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

if [ $s2t = true ]; then
    s2t_label=s2t
elif [ $s2t = false ]; then
    s2t_label=prompt
fi

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for data_set in $data_sets; do
        cp $dumpdir/$data_set/deltafalse/${datajsoninput} $dumpdir/$data_set/deltafalse/data.${s2t_label}.tmp.2.json
        json=$dumpdir/$data_set/deltafalse/${datajsoninput}
        CUDA_VISIBLE_DEVICES=${CUDA} python local.apl.v3/stt/calculate_error_rate.py --input-json $dumpdir/$data_set/deltafalse/data.${s2t_label}.tmp.2.json \
            --output-path $dumpdir/$data_set/deltafalse/${datajsonoutput}.tmp \
            --model /share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/disfluency/model/swbd_fisher_bert_Edev.0.9078.pt
        rm $dumpdir/$data_set/deltafalse/data.${s2t_label}.tmp.2.json
    done
fi

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for data_set in $data_sets; do
        json=$dumpdir/$data_set/deltafalse/${datajsoninput}
        appd=$dumpdir/$data_set/deltafalse/${datajsonoutput}.tmp
        addjson.py --verbose ${verbose} -i true \
            ${json} ${appd} > $dumpdir/$data_set/deltafalse/${datajsonoutput}
    done
fi
