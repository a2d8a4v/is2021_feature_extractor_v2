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
verbose=0
data_root=data
replace_text=true
CUDA=
max_nj=30
max_nj_cuda=6
long_decode_mode=true
skip_decode=false
text=text_prompt
queue_split=queue_split
dumpdir=dump
savejson=data.new.json
s2t=false

cmd=queue.pl

echo "$0 $@"
. parse_options.sh

set -euo pipefail

if [ ! -z ${tag} ]; then
    model_name=${model_name}_${tag}
fi

if [ ${s2t} == false ]; then
    echo "prompt mode!"
    if [ $stage -lt 1 ]; then
        echo "reset stage to 1, do not need to extract audio features and decode..."
        stage=1
    fi
fi

model=$model_dir/model
ivec_extractor=$model_dir/extractor
ivec_dir=$model_dir/model_online
lang=$model_dir/data/lang
mfcc_config=$model_dir/conf/mfcc_hires.conf
cmvn_config=$model_dir/conf/online_cmvn.conf

if [ -z $graph_affix ]; then
    graph_affix=_tgt3
fi

if [ ${skip_decode} == true ]; then
    skip_decode=1
else
    skip_decode=0
fi

graph_dir=$model_dir/model/graph${graph_affix}

if [ $stage -le -2 ] && [ $stop_stage -ge -2 ] ; then    
    for test_set in $test_sets; do 
        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        if [ $nspk -ge $max_nj ]; then
            nspk=$max_nj;
        fi
        
        steps/make_mfcc.sh --nj $nspk \
          --mfcc-config $mfcc_config \
          --cmd "$train_cmd" $data_root/${test_set} || exit 1;
        steps/compute_cmvn_stats.sh $data_root/${test_set} || exit 1;
        utils/fix_data_dir.sh $data_root/${test_set}
    done
fi

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ] ; then    
    for test_set in $test_sets; do
        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        if [ $nspk -ge $max_nj ]; then
            nspk=$max_nj;
        fi
        steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nspk \
          $data_root/${test_set} $ivec_extractor \
          $ivec_dir/ivectors_${test_set} || exit 1;
    done
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ] ; then    
    # note: if the features change (e.g. you add pitch features), you will have to
    # change the options of the following command line.
    if [ ! -d ${model}_online ]; then
        steps/online/nnet3/prepare_online_decoding.sh \
            --mfcc-config $mfcc_config \
            --online_cmvn_config $cmvn_config \
            $lang $ivec_extractor $model ${model}_online
    fi

    for test_set in $test_sets; do
        nspk=$(wc -l <$data_root/${test_set}/spk2utt)
        # note: we just give it "$data_rott/${test_set}" as it only uses the wav.scp, the
        # feature type does not matter.
        if [ $nspk -gt $max_nj ]; then
            nspk=$max_nj
        fi
        decode_dir=${model}_online/decode_${test_set}${graph_affix}
        steps/online/nnet3/decode.sh \
          --stage ${skip_decode} \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nspk --cmd "$decode_cmd" \
          $graph_dir $data_root/${test_set} ${decode_dir} || exit 1

    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then    
    for test_set in $test_sets; do
        decode_dir=${model}_online/decode_${test_set}${graph_affix}
        dest_dir=$data_root/$test_set/$model_name
        mkdir -pv $dest_dir > /dev/null 2>&1

        
        utils/copy_data_dir.sh $data_root/${test_set} $dest_dir
        if [ $replace_text == true ] && [ $s2t == true ]; then
            best_wer=${decode_dir}/scoring_kaldi/best_wer
            recog_fn=`awk '{print $NF}' $best_wer | awk -F"/" '{print $NF}' | awk -F"_" '{print "penalty_"$3"/"$2".txt"}'`
            recog_text=$decode_dir/scoring_kaldi/$recog_fn
            echo "Copy from $recog_text to $dest_dir/text"
            cp $recog_text $dest_dir/text
        fi

        if [ $s2t == false ]; then
            echo "replace the text in ${dest_dir}"
            python local.apl.v3/utils/in_place_oov_tokens.py --input_text_file $data_root/${test_set}/text \
                                                           --output_text_file $dest_dir/text \
                                                           --words_file $lang/words.txt
        fi
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ] ; then    
    for test_set in $test_sets; do
        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        
        if [ $nspk -ge $max_nj ]; then
            nspk=$max_nj;
        fi
        
        dest_dir=$data_root/$test_set/$model_name
        data_dir=$dest_dir
        
        echo "Align $data_dir with $model"
        ivectors_data_dir=$ivec_dir/ivectors_${test_set}
        decode_dir=${model}_online/decode_${test_set}${graph_affix}
        result_dir=${decode_dir}/align_${model_name}
        # steps/chain/align_lats_ctm.sh <data-dir> <lang-dir> <src-dir> <align-dir>
        local.apl.v3/stt/align_lats_ctm.sh --cmd "queue.pl" --nj $nspk --online-ivector-dir $ivectors_data_dir --generate_ali_from_lats true $data_dir $lang $model $result_dir
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ] ; then    
    for test_set in $test_sets; do
        
        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        
        if [ $nspk -ge $max_nj ]; then
            nspk=$max_nj;
        fi
        
        dest_dir=$data_root/$test_set/$model_name
        data_dir=$dest_dir
        ivectors_data_dir=$ivec_dir/ivectors_${test_set}
        decode_dir=${model}_online/decode_${test_set}${graph_affix}
        result_dir=${decode_dir}/gop_${model_name}
        json_dir=${result_dir}/json_${model_name}
        log_dir=${result_dir}/log
        
        echo "Computing GOP of $data_dir with $model"
        
        local.apl.v3/gop/compute-dnn-bi-gop.sh --nj "$nspk" --cmd "queue.pl" --split_per_speaker "true" $data_dir $ivectors_data_dir \
              $lang $model $result_dir    ### dnn model    
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ] ; then    
    for test_set in $test_sets; do
        
        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        
        if [ $nspk -ge $max_nj ]; then
            nspk=$max_nj;
        fi
        
        dest_dir=$data_root/$test_set/$model_name
        data_dir=$dest_dir
        ivectors_data_dir=$ivec_dir/ivectors_${test_set}
        decode_dir=${model}_online/decode_${test_set}${graph_affix}
        result_dir=${decode_dir}/gop_${model_name}
        json_dir=${result_dir}/json
        log_dir=${result_dir}/log
        text_fn=$dest_dir/text
        mkdir -p $json_dir > /dev/null 2>&1
        
        echo "Processing GOP result of $data_dir with $model"
        echo "python local/gop/gop_log_parser.py --log_dir $log_dir --json_dir $json_dir --words_fn $lang/words.txt --text_fn $text_fn --conf $model_dir/sample_worker_en.yaml"
         
        python local.apl.v3/gop/gop_log_parser.py \
            --log_dir $log_dir \
            --json_dir $json_dir \
            --words_fn $lang/words.txt \
            --text_fn $text_fn \
            --s2t $s2t \
            --conf $model_dir/sample_worker_en.yaml
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ] ; then    
    for test_set in $test_sets; do

        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        
        if [ $nspk -ge $max_nj_cuda ]; then
            nspk=$max_nj_cuda;
        fi

        data_dir=$data_root/$test_set        
        dest_dir=$data_root/$test_set/$model_name
        decode_dir=${model}_online/decode_${test_set}${graph_affix}
        result_dir=${decode_dir}/gop_${model_name}
        json_dir=${result_dir}/json

        mkdir -pv $dest_dir/$queue_split > /dev/null 2>&1

        python local.apl.v3/stt/split_input.py --split_number $nspk \
                                        --output_dir_path $dest_dir/$queue_split \
                                        --wav_scp_file_path $dest_dir/wav.scp \
                                        --text_file_path $dest_dir/text \
                                        --utt2dur_file_path $dest_dir/utt2dur \
                                        --gop_json_file_path $json_dir/gop_scores.json \
                                        --lexicon_file_path $data_root/local/dict/lexicon.txt
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ] ; then    
    for test_set in $test_sets; do

        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        
        if [ $nspk -ge $max_nj_cuda ]; then
            nspk=$max_nj_cuda;
        fi
        
        data_dir=$data_root/$test_set
        dest_dir=$data_root/$test_set/$model_name
        decode_dir=${model}_online/decode_${test_set}${graph_affix}
        result_dir=${decode_dir}/gop_${model_name}
        json_dir=${result_dir}/json
        logdir=$dest_dir/logdir
        mkdir -pv $logdir > /dev/null 2>&1
        
        echo "Parallel processing of feature extracting for $test_set..... "
        run.pl JOB=1:$nspk $logdir/prepare_feats.JOB.log \
            CUDA_VISIBLE_DEVICES=${CUDA} python local.apl.v3/stt/prepare_feats_queue.py \
                --s2t $s2t \
                --data_dir $data_dir \
                --split_number JOB \
                --model_name $model_name --gop_result_dir $result_dir \
                --tag $tag --long_decode_mode $long_decode_mode \
                --gop_json_fn $dest_dir/$queue_split/gop_scores.JOB.json \
                --utt2dur_file_path $dest_dir/$queue_split/utt2dur.JOB \
                --lexicon $dest_dir/$queue_split/lexicon.JOB.txt \
                --wav_scp_file_path $dest_dir/$queue_split/wav.JOB.scp \
                --conf_file_path $model_dir/sample_worker_en.yaml \
                --input_word_file_path $lang/words.txt \
                --text_file_path $dest_dir/$queue_split/text.JOB || exit 1
        # We can use the dicts generated by Kaldi toolkit or ESPNet toolkit here
    done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ] ; then    
    for test_set in $test_sets; do

        nspk=$(wc -l <$data_root/$test_set/spk2utt)
        
        if [ $nspk -ge $max_nj_cuda ]; then
            nspk=$max_nj_cuda;
        fi

        data_dir=$data_root/$test_set
        dest_dir=$data_dir/$model_name
        if [ $(find $dest_dir -name "error.*" | wc -l) -gt 0 ]; then
            cat $dest_dir/error.* > $dest_dir/error
        fi
        # cat $dest_dir/text.* > $dest_dir/text
        # cat $dest_dir/ctm.* > $dest_dir/ctm
        # cat $data_dir/tmp_apl_decoding_${tag}.*.list > $data_dir/tmp_apl_decoding_${tag}.list

        json_files=""
        for f in $(find "${dest_dir}" -name "all.*.json"); do
            json_files+="$f "
        done

        echo "Processing files combining for $test_set..... "
        local.apl.v3/utils/combine_jsons.py \
            --input_jsons_file_path ${json_files} \
            --output_file_path $dest_dir/all.json
    done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ] ; then
    for test_set in $test_sets; do
        json=$dumpdir/$test_set/deltafalse/data.json
        appd=$data_root/$test_set/$model_name/all.json
        # @https://espnet.github.io/espnet/apis/utils_py.html#addjson-py
        addjson.py --verbose ${verbose} -i true \
            ${json} ${appd} > $dumpdir/$test_set/deltafalse/${savejson}
    done
fi

echo "Extracting Done."
