#!/usr/bin/env bash

set -euo pipefail

data_root=data
data_sets="cerf_train_tr cerf_train_cv"
verbose=0
dumpdir=dump
CUDA=
stage=0
stop_stage=
tag=
datajsoninput=data.new.json
datajsonoutput=data.new.dis.json
# split=False
savepickle=True
s2t=
s2t_label=

echo "$0 $@"
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

if [ $s2t == true ]; then
    s2t_label=s2t
elif [ $s2t == false ]; then
    s2t_label=prompt
fi

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for data_set in $data_sets; do
        cp $dumpdir/$data_set/deltafalse/${datajsoninput} $dumpdir/$data_set/deltafalse/data.${s2t_label}.tmp.json
        json=$dumpdir/$data_set/deltafalse/${datajsoninput}
        if [ $savepickle = True ]; then
            appd=$dumpdir/$data_set/deltafalse/data.dis.pk.json
        else
            appd=$dumpdir/$data_set/deltafalse/data.dis.json
        fi
        CUDA_VISIBLE_DEVICES=${CUDA} python local.apl.v3/disfluency/get_bert_disfluency.py --input-json $dumpdir/$data_set/deltafalse/data.${s2t_label}.tmp.json \
            --output-path $appd \
            --output-pickle-path $dumpdir/$data_set/deltafalse/split_utts_${tag} \
            --save-pickle $savepickle \
            --model /share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/disfluency/model/swbd_fisher_bert_Edev.0.9078.pt
        rm $dumpdir/$data_set/deltafalse/data.${s2t_label}.tmp.json
        addjson.py --verbose ${verbose} -i true \
            ${json} ${appd} > dump/$data_set/deltafalse/${datajsonoutput}
    done
fi

# if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#     if [ $split != True ]; then
#         for data_set in $data_sets; do
#             json=$dumpdir/$data_set/deltafalse/${datajson}
#             appd=$dumpdir/$data_set/deltafalse/data.dis.json
#             addjson.py --verbose ${verbose} -i true \
#                 ${json} ${appd} > $dumpdir/$data_set/deltafalse/data.new.dis.json
#         done
#     fi
# fi

# if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#     if [ $split = True ]; then
#         for data_set in $data_sets; do
#             json=$dumpdir/$data_set/deltafalse/${datajson}
#             disj=$dumpdir/$data_set/deltafalse/data.dis.json
#             appd=$dumpdir/$data_set/deltafalse/data.dis.pk.json
#             python local.apl.v1/disfluency/split_to_pickle.py --input-json ${disj} \
#                 --output-path $dumpdir/$data_set/deltafalse/split_utts \
#                 --output-json ${appd}
#             addjson.py --verbose ${verbose} -i true \
#                 ${json} ${appd} > $dumpdir/$data_set/deltafalse/data.new.dis.json
#         done
#     fi
# fi
