import pickle
import numpy as np
from kaldiio import ReadHelper
import torch
import math


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.
    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).
    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def pad_list_n(xs, pad_value, padding):
    n_batch = len(xs)
    if padding is None:
        max_len = max(len(x) for x in xs)
    else:
        max_len = padding
    _xs = []

    for i in range(n_batch):
        _xs.append(" ".join(xs[i] + [pad_value]*(max_len-len(xs[i]))))

    return _xs


def pickleStore(savethings, filename):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return


def pikleOpen(filename):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p


def feats_scp_reader(path, name): 
    p = path+"/"+name+".scp"
    rtn = {}
    tl = {}
    with ReadHelper('scp:{}'.format(p)) as reader:
        for i , ( key, numpy_array ) in enumerate( reader ):
            rtn[key] = numpy_array
            tl[i] = key
        return rtn , tl


def get_source_from_file(path, name, type):
    p = path+"/"+name
    rtn = {}
    with open(p,"r") as r:
        for l in r.readlines():
            rtn[l.split()[0]] = l.split()[1:]

    utt2idx = dict(zip(list(rtn.keys()), list(range(0, len(rtn)))))
    idx2utt = dict(zip(list(range(0, len(rtn))), list(rtn.keys())))

    _rtn = {}
    for i, j in rtn.items():
        _rtn[utt2idx[i]] = j

    if type == 'src':
        return _rtn, idx2utt
        
    return _rtn


def get_target_from_file(path, name):
    p = path+"/"+name
    rtn = {}
    with open(p, "r") as r:
        for l in r.readlines():
            rtn[l.split()[0]] = l.split()[1:]

    utt2idx = dict(zip(list(rtn.keys()), list(range(0, len(rtn)))))
    idx2utt = dict(zip(list(range(0, len(rtn))), list(rtn.keys())))

    _rtn = {}
    for i, j in rtn.items():
        _rtn[utt2idx[i]] = j

    return _rtn


def data_to_tokenids(data, dictlist):

    _dict = {}
    ## For tgt
    if isinstance(dictlist, dict):
        _dict = dictlist
    else:
        with open(dictlist, 'r') as p:
            for l in p.readlines():
                _dict[l.split()[0]] = l.split()[1]

    for i, j in data.items():
        data[i] = np.array([int(_dict[t]) for t in j])

    return data


def data_to_padding(data, device, padding=None):

    if isinstance(data, dict):
        rtn = []
        for j in data.values():
            rtn.append(j)

        rtn = pad_list(
            [torch.from_numpy(x.real).float() for x in rtn], 0
        ).to(device, dtype=torch.int32).long() # For HuggingFace, using torch.int and long()

    else:
        rtn = pad_list_n(data, "[PAD]", padding)

    return rtn


def data_processing(path, name, device, _dict, padding=None, type="src", totokids=True):
    if type == "src":
        _src, idx2utt = get_source_from_file(path, name, type)
        if totokids:
            _src = data_to_tokenids(_src, _dict)
            _src = data_to_padding(_src, device, padding)
            return _src, idx2utt
        else:
            _src = [j for j in _src.values()]
            _src = data_to_padding(_src, device, padding)
            return _src, idx2utt
    elif type == "tgt":
        _tgt = get_target_from_file(path, name)
        if totokids:
            _tgt = data_to_tokenids(_tgt, _dict)
            _tgt = data_to_padding(_tgt, device, padding)
            return _tgt
        else:
            _tgt = [j for j in _tgt.values()]
            _tgt = data_to_padding(_tgt, device, padding)
            return _tgt


def countForSegment(tokenizer, input_, input_enc, padding):
    rtn = []
    enc_att = input_enc['attention_mask']
    for j_, l in enumerate(list(input_)):
        l_ = [ e for e in l.split() if e != '[PAD]']
        t  = tokenizer(l_, padding=True, truncation=True, max_length=padding, return_tensors="pt")
        a  = t['attention_mask']
        t  = t['input_ids']

        # get dim
        d = t.shape[1]
        c = -1 # seen as index
        rtn_ = []
        c += 1 # For first token [CLS]
        for i_, i in enumerate(range(t.shape[0])):

            v = t[i,:]
            # remove [CLS] 101 and [SEP] 102 tokens
            z_ = torch.tensor([0])
            v_ = torch.where(v == 101, z_, v)
            v_ = torch.where(v_ == 102, z_, v_)

            s = sum([1 for i in range(v_.shape[0]) if v_[i].item()>0 ])
            if s == 1:
                c += 1 # index go forward
            else:
                st = c+1 # start index
                ed = st+s-1 # end index
                c += s # index go forward
                rtn_.append([st, ed])

        ## Generate mask
        enc_att[j_,:] = 0
        for pa in rtn_:
            enc_att[j_,:][pa[0]:pa[1]+1] = 1

        ## Append
        rtn.append(rtn_)
    # print(rtn)
    # input()
    return rtn, enc_att