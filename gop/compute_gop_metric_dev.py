from collections import defaultdict
import numpy as np
import argparse
from gop_preprocess_v2 import GOPModel
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser()

parser.add_argument("--dev_set",
                     default="exp/nnet3_cleaned/tdnn1c_sp/gop_train_l2",
                     type=str)

parser.add_argument("--test_set",
                     default="exp/nnet3_cleaned/tdnn1c_sp/gop_test_l2",
                     type=str)

parser.add_argument("--dont_care_phones",
                     default="0,1",
                     type=str)

args = parser.parse_args()

''' Parameter Initialization '''
dev_model_dir = args.dev_set
dev_data_dir = dev_model_dir + "/capt"

test_model_dir = args.test_set
test_data_dir = test_model_dir + "/capt"

# preprocess
phone_text = {}
annotation_seq = {}
phone_map = {}
phone_seq = {}
gop_seq = {}
# threshold
phone_gop_ann = defaultdict(list)
phone_uttid = defaultdict(list)
dont_care_phone_id = [ int(pid) for pid in args.dont_care_phones.split(",") ]

# score function using sigmoid
def sigmoid(x):
    return 2 * (1. / (1 + np.exp(-x)))

def calc_recall_precision_f1(y_true, y_pred):
    from sklearn.metrics import classification_report
    results = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True)
    classes = ['0', '1']
    measures = ["f1-score"] # ["recall", "precision", "f1-score"]
    return results["weighted avg"]["f1-score"]

def threshold_decision(gop_ann, threshold=None):
    # return y_true, y_pred
    # true = correct pronounce (1)
    # false = mispronounce (0)
    max_f1_score = -1
    ret_y_pred = []
    best_threshold = -10000
    
    if threshold == None:
        thresholds = np.linspace(0,1,11)
    else:
        thresholds = [threshold]

    for th in thresholds:
        y_true = []
        y_pred = []
        # convert gop to cor or mis
        for ga in gop_ann:
            g, a = ga
            # y_true and y_pred (0 = mis, 1 = corr)
            if a == True:
                y_true.append(1)
            else:
                y_true.append(0)
            # y_pred: pred corr if larger than threshold value, 
            # and pred small if smaller than threshold value
            if g >= th:
                y_pred.append(1)
            else:
                y_pred.append(0)
        # calculate mis and cor
        f1_score = calc_recall_precision_f1(y_true, y_pred)
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            ret_y_pred = y_pred
            best_threshold = th
        # the more closer to 0.5, the better
        elif f1_score == max_f1_score and abs(best_threshold - 0.5) > abs(th - 0.5):
            max_f1_score = f1_score
            ret_y_pred = y_pred
            best_threshold = th
    
    return y_true, ret_y_pred, best_threshold, max_f1_score

def conv_GOP2Pred_by_threshold(phone_gop_ann, phone_uttid, prior_thresholds = None):
    ''' adjust threshold and calculate f1-score '''
    print("Threashold and calculate f1-score")
    uttid_results = defaultdict(list)
    # evalution
    ground_truth = []
    prediction = []
    use_threshold = False
    if prior_thresholds == None:
        thresholds = {}
    else:
        use_threshold = True
        print("calculate by threshold")
        thresholds = prior_thresholds
    # compute threshold of each phone
    # I should create two list that like below:
    # ground truth: [0, 0, 1, 1, 0 ....]
    # predict: [0, 1, 0, 0]
    # call scikit learn toolkit (Recall, Precision, and F1-score)
    for phn_id in phone_gop_ann.keys():
        #print(len(phone_gop_ann[phn_id]))
        #print(len(phone_uttid[phn_id]))
        if phn_id not in thresholds:
            if use_threshold:
                print("[WARNING] using threshold, but we don't have threshold of this phone.")
            y_true, y_pred, best_threshold, max_f1_score = threshold_decision(phone_gop_ann[phn_id])
            thresholds[phn_id] = best_threshold
        else:
            y_true, y_pred, best_threshold, max_f1_score = threshold_decision(phone_gop_ann[phn_id], thresholds[phn_id])
            
        ground_truth += y_true
        prediction += y_pred
        for i in range(len(y_pred)):
            utt_id = phone_uttid[phn_id][i]
            uttid_results[utt_id].append([ground_truth[i], prediction[i]])
    return [ground_truth, prediction, uttid_results, thresholds]
    

if __name__ == "__main__":
    dev_GOP_mdl = GOPModel(dev_data_dir, dev_model_dir, dont_care_phone_id)
    [dev_phone_gop_ann, dev_phone_uttid] = dev_GOP_mdl.getPhoneGOPAnn()
    
    test_GOP_mdl = GOPModel(test_data_dir, test_model_dir, dont_care_phone_id)
    [phone_gop_ann, phone_uttid] = test_GOP_mdl.getPhoneGOPAnn()
    [ground_truth, prediction, uttid_results, thresholds] = conv_GOP2Pred_by_threshold(dev_phone_gop_ann, dev_phone_uttid)
    [ground_truth, prediction, uttid_results, thresholds] = conv_GOP2Pred_by_threshold(phone_gop_ann, phone_uttid, thresholds) 

    from sklearn.metrics import classification_report, confusion_matrix
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(classification_report(ground_truth, prediction, output_dict=True))
    tn, fp, fn, tp = confusion_matrix(ground_truth, prediction).ravel()
    print("tn, fp, fn, tp")
    print(tn, fp, fn, tp)
    print("Evaluation Done !")
