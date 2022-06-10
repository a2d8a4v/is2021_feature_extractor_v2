from collections import defaultdict
import numpy as np
import itertools

class GOPModel(object):
    def __init__(self, capt_data_dir, model_dir, dont_care_phone_id, use_sigmoid = True):
        # preprocess
        self.annotation_seq = self.__prepAnnotation(capt_data_dir)
        self.phone_map = self.__prepPhoneMap(model_dir)
        self.phone_seq = self.__prepPhoneSeq(model_dir)
        self.gop_seq = self.__prepGOPSeq(model_dir, use_sigmoid)
        [self.gop_seq, self.phone_seq] = self.__normalize(self.gop_seq, self.phone_seq, dont_care_phone_id)
        
    def __prepAnnotation(self, capt_data_dir):
        ''' Read and process (data related) '''
        # phone_text (phone_text) {'utt': ["p1", ..., "pn"]}
        # annotation_seq (annotation_seq) (ignore insertion) {'utt': [1,0]}
        phone_text = {}
        annotation_seq = {}
        with open(capt_data_dir + "/annotation.txt", "r") as anno_file:
            for line1, line2, line3, line4 in itertools.zip_longest(*[anno_file]*4):
                ref_info = line1.split()
                utt_id = ref_info[0]
                
                ref_info = ref_info[2:]
                hyp_info = line2.split()[2:]
                op_info = line3.split()[2:]
                # ignore 4rd line (non-sense)
                assert len(ref_info) == len(hyp_info) and len(hyp_info) == len(op_info)
                ann_seq = [ (t=="C") for t in op_info if t != "I"]
                annotation_seq[utt_id] = ann_seq
        return annotation_seq

    def __prepPhoneMap(self, model_dir):
        ''' Read and process (model related) '''
        phone_map = {}
        # phones.txt (phone_map) {'phone_id': phone}
        # NOTE: ignore phone_id 0
        with open(model_dir + "/phones.txt", "r") as phone_file:
            for line in phone_file.readlines():
                line = line.split("\n")[0].split(" ")
                phone_map[int(line[1])] = line[0]
        return phone_map

    def __prepPhoneSeq(self, model_dir):
        # phone.ctm (phone_seq) {'utt_id': [phone_id1, phone_id2, ..., phone_idN]}
        phone_seq = {}
        with open(model_dir + "/phone.ctm", "r") as ctm_file:
            for line in ctm_file.readlines():
                line = line.split("\n")[0].split(" ")
                utt_id = line[0]
                phone_id = line[-1]
                if utt_id in phone_seq:
                    phone_seq[utt_id].append(int(phone_id))
                else:
                    phone_seq[utt_id] = [int(phone_id)]
        return phone_seq

    def __prepGOPSeq(self, model_dir, use_sigmoid = True):
        # gop.txt (gop_seq) {'utt_id': [gop1, gop2, ..., gopN]}
        # use sigmoid (tanh) or not
        gop_seq = {}
        with open(model_dir + "/gop.txt", "r") as gop_file:
            for line in gop_file.readlines():
                line = line.split("\n")[0].split(" ")
                utt_id = line[0]
                if use_sigmoid:
                    gops = [ self.__sigmoid(float(g)) for g in line[3:-1] ]
                else:
                    gops = [ float(g) for g in line[3:-1] ]
                gop_seq[utt_id] = gops
        return gop_seq

    def __normalize(self, gop_seq, phone_seq, dont_care_phone_id):
        ''' Preparing dict for threshold adjusting '''
        print("Preparing dict for threshold adjusting")
        gop_seq_re = defaultdict(list)
        phone_seq_re = defaultdict(list)
        # remove some phone that we don't care
        for utt_id in phone_seq.keys():
            for i, phn_id in enumerate(phone_seq[utt_id]):
                if phn_id in dont_care_phone_id:
                    # do nothing
                    pass
                else:
                    gop_seq_re[utt_id].append(gop_seq[utt_id][i])
                    phone_seq_re[utt_id].append(phone_seq[utt_id][i])
        return [gop_seq_re, phone_seq_re]

    def __sigmoid(self, x):
        return 2 * (1. / (1 + np.exp(-x)))
    
    def getPhoneGOPAnn(self):
        total=0
        phone_gop_ann = defaultdict(list)
        phone_uttid = defaultdict(list)
        phone_seq = self.phone_seq
        gop_seq = self.gop_seq
        annotation_seq = self.annotation_seq
        # convert annotation_seq, phone_seq_re and gop_seq_re to {'phone_id1': [[gop, T], [gop, F]]}
        for utt_id in phone_seq.keys():
            for i, phn_id in enumerate(phone_seq[utt_id]):
                if len(phone_seq[utt_id]) != len(gop_seq[utt_id]) or len(phone_seq[utt_id]) != len(annotation_seq[utt_id]):
                    print(utt_id, phone_seq[utt_id], gop_seq[utt_id], phone_seq[utt_id], annotation_seq[utt_id])
                    exit(0)
                gop_score = gop_seq[utt_id][i]
                anno = annotation_seq[utt_id][i]
                phone_gop_ann[phn_id].append([gop_score, anno])
                phone_uttid[phn_id].append(utt_id)
                # utt_id + "_" + phn_id
                total += 1
        return [phone_gop_ann, phone_uttid]

    def getAnnotation(self):
        return self.annotation_seq
    
    def getPhoneMap(self):
        return self.phone_map

    def getPhoneSeq(self):
        return self.phone_seq

    def getGOPSeq(self):
        return self.gop_seq
