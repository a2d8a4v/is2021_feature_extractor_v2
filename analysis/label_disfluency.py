import json
import pickle
import sys


def jsonLoad(scores_json):
    with open(scores_json) as json_file:
        return json.load(json_file)


def pickleStore( savethings , filename ):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return


def pikleOpen( filename ):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p


def opentext( file, col_start ):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            f_ = []
            for w in l_[col_start:]:
                f_.append(checkDisfluency(w))
            s[l_[0]] = f_
    return s


def checkDisfluency(word):
    
    # Labels
    fluency    = "O"
    disfluency = "D"

    b = ["@breath", "@cough", "@laugh", "@bkg", "@ns", "@sil"]

    # Interregnum, and controllable issues by speakers
    d = ["@sil"]
    o = "@"

    if word in d:
        return disfluency
    else:
        if o in word and word not in b: 
            return disfluency
        return fluency


def printOut(data):
    for i,j in data.items():
        sys.stdout.write(i+'\t')
        # print("{}\t{}".format(i, " ".join(j)))
        sys.stdout.write(" ".join(j))
        sys.stdout.write('\n')


## START
file = "/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/dev/text.new"
data = opentext( file, 1 )

## Print data
printOut(data)

