""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np

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
    print("train frames:", train_data[0], "to", train_data[-1])
    print("eval frames:", dev_data[0], "to", dev_data[-1])

    return train_data, dev_data

def baseline_bgs(args):
    train_data, dev_data = train_dev_split(args)
    
    #Hyperparams
    history = 90
    varThreshold = 250
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    
    # read all the training frames
    imgs = []
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        _ = background_model.apply(img,learningRate=learningRate)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        # pred_mask = cv2.absdiff(background_model, img)
        # pred_mask = cv2.threshold(pred_mask, 50, 255, cv2.THRESH_BINARY)[1] # first returned value is True
        
        pred_mask = background_model.apply(img)
        # pred_mask = cv2.erode(pred_mask, kernel, iterations = 1)
        # pred_mask =  cv2.dilate(pred_mask, kernel, iterations = 1)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        pred_mask = cv2.medianBlur(pred_mask, 7)
        
        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

def hisEqulColor(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YCR_CB2BGR)
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((3,3), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 25)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm

def illumination_bgs(args):
    train_data, dev_data = train_dev_split(args)
    
    # hyper params
    history = 50
    varThreshold = 300
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    # read the training frames
    imgs = []
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) 
        imgs.append(img)

        img = hisEqulColor(img)
        _ = background_model.apply(img, learningRate=learningRate)
    
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        img = hisEqulColor(img)
        
        pred_mask = background_model.apply(img)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        pred_mask = cv2.medianBlur(pred_mask, 7)
        
        pred_img_name = "gt" + img_name[2:-3] + "png"
        pred_mask = cv2.resize(pred_mask, (320, 240))
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

def jitter_bgs(args):
    train_data, dev_data = train_dev_split(args)

    #Hyperparams
    history = 250
    varThreshold = 250
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    
    # read all the training frames
    imgs = []
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

        _ = background_model.apply(img,learningRate=learningRate)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        
        pred_mask = background_model.apply(img)
        pred_mask = cv2.erode(pred_mask, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        pred_mask = cv2.medianBlur(pred_mask, 7)
        
        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

def dynamic_bgs(args):
    train_data, dev_data = train_dev_split(args)

    # hyper params
    history = 250
    varThreshold = 250
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    # read the training frames
    imgs = []
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) 
        imgs.append(img)
        _ = background_model.apply(img,learningRate=learningRate)
    
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
        
    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        
        pred_mask = background_model.apply(img)
        pred_mask = cv2.erode(pred_mask, kernel)
        pred_mask = cv2.erode(pred_mask, kernel)
        pred_mask =  cv2.dilate(pred_mask,kernel)
        pred_mask =  cv2.dilate(pred_mask,kernel)

        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        pred_mask = cv2.medianBlur(pred_mask, 7)
        
        # pred_mask = ObtainForeground(pred_mask)
        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

def ptz_bgs(args):
    # IMPORTANT - eval_frames.txt is not present in data but it is named as temporalROI.txt | This is handled by giving correct path in the arguments
    train_data, dev_data = train_dev_split(args)

    #Hyperparams
    history = 200
    varThreshold = 250
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    
    # read all the training frames
    imgs = []
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        _ = background_model.apply(img,learningRate=learningRate)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        
        pred_mask = background_model.apply(img)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        pred_mask = cv2.medianBlur(pred_mask, 7)

        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

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