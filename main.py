""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np

import ipdb #
import subprocess # remove this line before final submission

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

def train_dev_split(args):
    # find out the evaluation frames (assuming they will be at the end of the video)
    eval_frames = open(args.eval_frames).readlines()[0].split(' ')
    eval_frame_start = int(eval_frames[0])-1
    eval_frame_end = int(eval_frames[1])

    # complete video sequence
    filenames = os.listdir(args.inp_path)
    filenames.sort()

    # split the video sequence into train and dev set (as per eval frames given)
    train_data = filenames[0 : eval_frame_start] 
    dev_data = filenames[eval_frame_start : eval_frame_end]

    return train_data, dev_data

def baseline_bgs(args):
    train_data, dev_data = train_dev_split(args)

    # read all the training frames
    imgs = []
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        
    # background_model = cv2.convertScaleAbs(background_model)
    # background_model = cv2.createBackgroundSubtractorMOG2()
    # background_model = cv2.createBackgroundSubtractorKNN() 
    background_model = cv2.bgsegm.createBackgroundSubtractorGSOC()
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        _ = background_model.apply(img)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        # pred_mask = cv2.absdiff(background_model, img)
        # pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
        # pred_mask = cv2.threshold(pred_mask, 50, 255, cv2.THRESH_BINARY)[1] # first returned value is True
        pred_mask = background_model.apply(img)
        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

    # print(background_model)
    # subprocess.call("python eval.py -p COL780-A1-Data/baseline/predictions -g COL780-A1-Data/baseline/groundtruth", shell=True)

def illumination_bgs(args):
    #TODO complete this function
    # hard-coding for a bug fix : the masks are of different shape than the images
    path = args.eval_frames[:args.eval_frames.find("eval_frames.txt")]
    filenames = os.listdir(os.path.join(path, 'groundtruth'))
    img = cv2.imread(os.path.join(path, 'input', 'in000001.jpg'))
    # ipdb.set_trace()
    for filename in filenames:
        gt_mask = cv2.imread(os.path.join(path, 'groundtruth', filename))
        gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(path, 'groundtruth', filename), gt_mask)
    baseline_bgs(args)


def jitter_bgs(args):
    #TODO complete this function
    baseline_bgs(args)


def dynamic_bgs(args):
    #TODO complete this function
    baseline_bgs(args)

def ptz_bgs(args):
    #TODO: (Optional) complete this function

    # IMPORTANT - eval_frames.txt is not present in data but it is named as temporalROI.txt | for now I have stored a copy of it as eval_frames.txt
    baseline_bgs(args)


def main(args):
    if args.category not in "bijmp": # error in main.py -> earlier it was bijdp
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