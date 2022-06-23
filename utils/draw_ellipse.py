import os
import argparse

import numpy as np
from matplotlib.patches import Ellipse
from utilities import (
    pikleOpen
)


def fitEllipse(cont,method):

    x=cont[:,0]
    y=cont[:,1]

    x=x[:,None]
    y=y[:,None]

    D=np.hstack([x*x,x*y,y*y,x,y,np.ones(x.shape)])
    S=np.dot(D.T,D)
    C=np.zeros([6,6])
    C[0,2]=C[2,0]=2
    C[1,1]=-1
    E,V=np.linalg.eig(np.dot(np.linalg.inv(S),C))

    if method==1:
        n=np.argmax(np.abs(E))
    else:
        n=np.argmax(E)

    a=V[:,n]

    #-------------------Fit ellipse-------------------
    b,c,d,f,g,a=a[1]/2., a[2], a[3]/2., a[4]/2., a[5], a[0]
    num=b*b-a*c
    cx=(c*d-b*f)/num
    cy=(a*f-b*d)/num

    angle=0.5*np.arctan(2*b/(a-c))*180/np.pi
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    a=np.sqrt(abs(up/down1))
    b=np.sqrt(abs(up/down2))

    #---------------------Get path---------------------
    ell=Ellipse((cx,cy),a*2.,b*2.,angle)
    ell_coord=ell.get_verts()

    params=[cx,cy,a,b,angle]

    return params, ell_coord


def plotSaveConts(contour_list, vowel_name, output_dir_path):
    '''Plot a list of contours'''
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax2=fig.add_subplot(111)
    for ii,cii in enumerate(contour_list):
        x=cii[:,0]
        y=cii[:,1]
        ax2.plot(x,y,'-')
    # plt.show(block=False)
    save_file_path = os.path.join(output_dir_path, 'draw_ellipse_'+vowel_name+'.png')
    plt.savefig(save_file_path)


if __name__ == '__main__':

    # parsers
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_file_path', default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/cefr_train_tr/gigaspeech_20220525_prompt/all.json", type=str)
    parser.add_argument('--output_dir_path', default="./test.pkl", type=str)
    args = parser.parse_args()

    # variables
    input_file_path = args.input_file_path
    output_dir_path = args.output_dir_path
    vowel_formant_dict = pikleOpen(input_file_path)
    collect = []

    # compute the radius of circle and save plot image for each vowel
    for vowel, f1_f2_info in vowel_formant_dict.items():
        params, ell = fitEllipse(f1_f2_info, 2)
        plotConts(
            [f1_f2_info, ell],
            vowel,
            output_dir_path
        )
        collect.append(ell)

    # save plot image for all
    plotConts(
        collect,
        '_vowels',
        output_dir_path
    )
