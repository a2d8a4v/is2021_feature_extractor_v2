import os
import argparse

import numpy as np
from espnet.utils.cli_utils import strtobool
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


def plotSaveConts(contour_list, ellipse_list, color_list, vowel_name, output_dir_path, annotate_vowels=False, vowel_list=[]):
    '''Plot a list of contours'''
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax2=fig.add_subplot(111)

    ## plot for data
    if contour_list is not None:
        for i, cii in enumerate(contour_list):
            x=cii[:,0]
            y=cii[:,1]
            ax2.scatter(
                x,
                y,
                alpha=0 if annotate_vowels else 0.5
            )
            if annotate_vowels:
                for _x, _y in zip(x, y):
                    ax2.annotate(
                        vowel_name.lower(),
                        (_x, _y)
                    )

    ## plot for ellipse
    i = 0
    for ellipse in ellipse_list:
        x=ellipse[:,0]
        y=ellipse[:,1]
        ax2.plot(
            x,
            y,
            color_list if isinstance(color_list, str) else color_list[i],
            linestyle='-'
        )
        i += 1

    ax2.legend(vowel_list, loc ="lower right") 

    save_file_path = os.path.join(output_dir_path, 'draw_ellipse_'+vowel_name.upper()+'.png')
    plt.savefig(save_file_path)


if __name__ == '__main__':

    # parsers
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_file_path', default="/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/tlt-school/is2021_data-prep-all_baseline/data/cefr_train_tr/gigaspeech_20220525_prompt/all.json", type=str)
    parser.add_argument('--output_dir_path', default="./test.pkl", type=str)
    parser.add_argument('--combine_to_basic_vowels', default=False, type=strtobool)
    parser.add_argument('--annotate_vowels', default=False, type=strtobool)
    args = parser.parse_args()

    # variables
    input_file_path = args.input_file_path
    output_dir_path = args.output_dir_path
    combine_to_basic_vowels = args.combine_to_basic_vowels
    annotate_vowels = args.annotate_vowels
    vowel_formant_dict = pikleOpen(input_file_path)
    collect_ellipse = []
    vowel_list = []

    # combine
    if combine_to_basic_vowels:
        mapping_vowels = ['a','e','i','o','u']
        new_vowel_formant_dict = {}
        for vowel, vowel_infos in vowel_formant_dict.items():
            new_vowel = " ".join(vowel).split()[0].lower()
            if new_vowel in mapping_vowels:
                new_vowel_formant_dict.setdefault(new_vowel, []).extend(vowel_infos)
        vowel_formant_dict = new_vowel_formant_dict

    # color
    vowel_colors = [
        '#003f5c',
        '#58508d',
        '#bc5090',
        '#ff6361',
        '#ffa600',
        '#00ff00',
        '#feb300',
        '#ff0000',
        '#007900',
        '#ba00ff',
        '#535f2c',
        '#aac9a8',
        '#e9a2a5',
        '#d73068',
        '#ffb06d',
        '#fde398',
        '#563432'
    ]

    # compute the radius of circle and save plot image for each vowel
    for i, (vowel, f1_f2_info) in enumerate(vowel_formant_dict.items()):
        f1_f2_info = np.array(f1_f2_info)
        params, ell = fitEllipse(f1_f2_info, 2)
        plotSaveConts(
            [f1_f2_info],
            [ell],
            vowel_colors[i],
            vowel,
            output_dir_path,
            vowel_list=[vowel],
            annotate_vowels=annotate_vowels
        )
        collect_ellipse.append(ell)
        vowel_list.append(vowel)

    # save plot image for all
    plotSaveConts(
        None,
        collect_ellipse,
        vowel_colors,
        '_vowels',
        output_dir_path,
        vowel_list=vowel_list,
        annotate_vowels=annotate_vowels
    )
