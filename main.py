""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np

import ipdb # remove this line before final submission

def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    return args


def baseline_bgs(args):
    # find out the evaluation frames (assuming they will be at the end of the video)
    eval_frames = open(args.eval_frames).readlines()[0].split(' ')
    eval_frame_start = int(eval_frames[0])-1
    eval_frame_end = int(eval_frames[1])-1

    # complete video sequence
    filenames = os.listdir(args.inp_path)
    filenames.sort()

    # split the video sequence into train and dev set (as per eval frames given)
    train_data = filenames[0 : eval_frame_start] 
    dev_data = filenames[eval_frame_start : eval_frame_end]

    # read all the training frames
    imgs = []
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    
    # # approach (1) - take average of all train frames to find the reference image (background model)
    # imgs = np.array(imgs)
    # background_model = np.mean(imgs, axis = 0)
    # background_model = cv2.convertScaleAbs(background_model)

    # approach (2) - running average
    background_model = np.float32(imgs[0])
    for img in imgs[1:]:
        cv2.accumulateWeighted(img, background_model, 0.02)
    background_model = cv2.convertScaleAbs(background_model)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        pred_mask = cv2.absdiff(background_model, img)

        # for approach (2) - update the background model
        # background_model = np.float32(background_model)
        # cv2.accumulateWeighted(img, background_model, 0.02)
        # background_model = cv2.convertScaleAbs(background_model)
        # till here

        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
        pred_mask = cv2.threshold(pred_mask, 50, 255, cv2.THRESH_BINARY)[1] # first returned value is True
        pred_mask = cv2.GaussianBlur(pred_mask,(5,5),0)
        cv2.imwrite(os.path.join(args.out_path, "gt" + img_name[2:-3] + "png"), pred_mask)

def illumination_bgs(args):
    #TODO complete this function
    pass


def jitter_bgs(args):
    #TODO complete this function
    pass


def dynamic_bgs(args):
    #TODO complete this function
    pass


def ptz_bgs(args):
    #TODO: (Optional) complete this function
    pass


def main(args):
    if args.category not in "bijdp":
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)