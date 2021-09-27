""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np

# import ipdb #
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
    print("train frames names:", train_data[0], train_data[-1])
    print("eval frames names:", dev_data[0], dev_data[-1])

    return train_data, dev_data

def ObtainForeground(img):
    # th, im_th = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV);
    im_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = img | im_floodfill_inv
    return im_out

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
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        
    # background_model = cv2.convertScaleAbs(background_model)
    # background_model = cv2.createBackgroundSubtractorMOG2(history = history, varThreshold = varThreshold )
    # background_model = cv2.createBackgroundSubtractorGMG()
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    # background_model = cv2.bgsegm.createBackgroundSubtractorGSOC()
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        _ = background_model.apply(img,learningRate=learningRate)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        # pred_mask = cv2.absdiff(background_model, img)
        # pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
        # pred_mask = cv2.threshold(pred_mask, 50, 255, cv2.THRESH_BINARY)[1] # first returned value is True
        
        pred_mask1 = background_model.apply(img)
        pred_mask2 = cv2.erode(pred_mask1,kernel,iterations = 1)
        pred_mask3 =  cv2.dilate(pred_mask2,kernel,iterations = 1)
        pred_mask4 = cv2.morphologyEx(pred_mask3, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask4, cv2.MORPH_CLOSE, kernel2)
        
        # pred_mask = ObtainForeground(pred_mask)
        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

    log_file = open('Results.txt',"a")
    log_info = []
    log_info.append(f"History: {history}\n")
    log_info.append(f"VarThreshold: {varThreshold}\n")
    log_info.append(f"Learning Rate: {learningRate}\n")
    log_file.writelines(log_info)
    
    subprocess.call("python .\\eval.py --pred_path COL780-A1-Data\\baseline\\predictions --gt_path COL780-A1-Data\\baseline\\groundtruth", shell=True)

# def hisEqulColor(img):
#     ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
#     channels=cv2.split(ycrcb)
#     # print len(channels)
#     cv2.equalizeHist(channels[0],channels[0])
#     cv2.merge(channels,ycrcb)
#     cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
#     return img
# def hisEqulColor(img):
#     alpha = 1.95 # Contrast control (1.0-3.0)
#     beta = 0 # Brightness control (0-100)
#     manual_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
#     return manual_result
def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

# Automatic brightness and contrast optimization with optional histogram clipping
# def hisEqulColor(image, clip_hist_percent=25):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Calculate grayscale histogram
#     hist = cv2.calcHist([gray],[0],None,[256],[0,256])
#     hist_size = len(hist)

#     # Calculate cumulative distribution from the histogram
#     accumulator = []
#     accumulator.append(float(hist[0]))
#     for index in range(1, hist_size):
#         accumulator.append(accumulator[index -1] + float(hist[index]))

#     # Locate points to clip
#     maximum = accumulator[-1]
#     clip_hist_percent *= (maximum/100.0)
#     clip_hist_percent /= 2.0

#     # Locate left cut
#     minimum_gray = 0
#     while accumulator[minimum_gray] < clip_hist_percent:
#         minimum_gray += 1

#     # Locate right cut
#     maximum_gray = hist_size -1
#     while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
#         maximum_gray -= 1

#     # Calculate alpha and beta values
#     alpha = 255 / (maximum_gray - minimum_gray)
#     beta = -minimum_gray * alpha

#     '''
#     # Calculate new histogram with desired range and show histogram 
#     new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
#     plt.plot(hist)
#     plt.plot(new_hist)
#     plt.xlim([0,256])
#     plt.show()
#     '''

#     auto_result = convertScale(image, alpha=alpha, beta=beta)
#     return auto_result


# def hisEqulColor(img):
    
#     b, g, r = cv2.split(img)
#     h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
#     h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
#     h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
#     # calculate cdf
#     cdf_b = np.cumsum(h_b)
#     cdf_g = np.cumsum(h_g)
#     cdf_r = np.cumsum(h_r)

#     # mask all pixels with value=0 and replace it with mean of the pixel values
#     cdf_m_b = np.ma.masked_equal(cdf_b, 0)
#     cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
#     cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

#     cdf_m_g = np.ma.masked_equal(cdf_g, 0)
#     cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
#     cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')


#     cdf_m_r = np.ma.masked_equal(cdf_r, 0)
#     cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
#     cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')
#     # merge the images in the three channels
#     img_b = cdf_final_b[b]
#     img_g = cdf_final_g[g]
#     img_r = cdf_final_r[r]

#     img_out = cv2.merge((img_b, img_g, img_r))
#     # validation
#     equ_b = cv2.equalizeHist(b)
#     equ_g = cv2.equalizeHist(g)
#     equ_r = cv2.equalizeHist(r)
#     equ = cv2.merge((equ_b, equ_g, equ_r))
#     # print(equ)
#     # cv2.imwrite('output_name.png', equ)
#     return img_out
    
    
    return img
def hisEqulColor(img):
    #THIS WORKS WONDERS BEST SCORE TILL NOW 0.47
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

    #Hyperparams
    history = 50
    varThreshold = 300
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    # read all the training frames
    imgs = []
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        imgs.append(img)
        
    # background_model = cv2.createBackgroundSubtractorMOG2(history = history, varThreshold = varThreshold ,detectShadows=False)
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # img = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 249, 5)
        img = hisEqulColor(img)
        # img = hisEqulColor(img)
        _ = background_model.apply(img,learningRate=learningRate)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        img = hisEqulColor(img)
        # img = hisEqulColor(img)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # img = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 249, 5)
        
        pred_mask = background_model.apply(img)

        pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)
        pred_mask =  cv2.dilate(pred_mask,kernel,iterations = 1)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        
        # pred_mask = ObtainForeground(pred_mask)
        pred_img_name = "gt" + img_name[2:-3] + "png"
        pred_mask = cv2.resize(pred_mask, (320, 240))
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

    
    
    subprocess.call("python eval.py --pred_path COL780-A1-Data\\ptz\\predictions --gt_path COL780-A1-Data\\ptz\\groundtruth", shell=True)



def jitter_bgs(args):
    train_data, dev_data = train_dev_split(args)

    #Hyperparams
    history = 250
    varThreshold = 250
    learningRate = -1
    
    # read all the training frames
    imgs = []
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    # background_model = cv2.createBackgroundSubtractorMOG2(history = history, varThreshold = varThreshold )
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        _ = background_model.apply(img,learningRate=learningRate)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # hyper_params
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        
        pred_mask = background_model.apply(img)
        pred_mask = cv2.erode(pred_mask, kernel)
        # pred_mask =  cv2.dilate(pred_mask,kernel,iterations = 1)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        
        # pred_mask = ObtainForeground(pred_mask)
        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

def dynamic_bgs(args):
    train_data, dev_data = train_dev_split(args)

    #Hyperparams
    history = 250
    varThreshold = 250
    learningRate = -1
    
    # read all the training frames
    imgs = []
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    # background_model = cv2.createBackgroundSubtractorMOG2(history = history, varThreshold = varThreshold )
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        _ = background_model.apply(img,learningRate=learningRate)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # hyper_params
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        
        pred_mask = background_model.apply(img)
        pred_mask = cv2.erode(pred_mask, kernel)
        # pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        # pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        pred_mask = cv2.erode(pred_mask, kernel)
        pred_mask =  cv2.dilate(pred_mask,kernel)
        pred_mask =  cv2.dilate(pred_mask,kernel)
        # pred_mask =  cv2.dilate(pred_mask,kernel)

        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel2)
        
        # pred_mask = ObtainForeground(pred_mask)
        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

    subprocess.call("python eval.py --pred_path COL780-A1-Data\\moving_bg\\predictions --gt_path COL780-A1-Data\\moving_bg\\groundtruth", shell=True)

def ptz_bgs(args):
    #TODO: (Optional) complete this function

    # IMPORTANT - eval_frames.txt is not present in data but it is named as temporalROI.txt | for now I have stored a copy of it as eval_frames.txt
    train_data, dev_data = train_dev_split(args)

    #Hyperparams
    history = 200
    varThreshold = 250
    learningRate = -1
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    # read all the training frames
    imgs = []
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name)) # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        
    # background_model = cv2.convertScaleAbs(background_model)
    # background_model = cv2.createBackgroundSubtractorMOG2(history = history, varThreshold = varThreshold )
    # background_model = cv2.createBackgroundSubtractorGMG()
    background_model = cv2.createBackgroundSubtractorKNN(history = history, dist2Threshold = varThreshold,detectShadows=False) 
    # background_model = cv2.bgsegm.createBackgroundSubtractorGSOC()
    for img_name in train_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        _ = background_model.apply(img,learningRate=learningRate)

    # check whether the path to write predictions over dev set exists or not
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # predict foreground over dev frames
    for img_name in dev_data:
        img = cv2.imread(os.path.join(args.inp_path, img_name))
        # pred_mask = cv2.absdiff(background_model, img)
        # pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
        # pred_mask = cv2.threshold(pred_mask, 50, 255, cv2.THRESH_BINARY)[1] # first returned value is True
        
        pred_mask1 = background_model.apply(img)
        pred_mask2 = cv2.erode(pred_mask1,kernel,iterations = 1)
        pred_mask3 =  cv2.dilate(pred_mask2,kernel,iterations = 1)
        pred_mask4 = cv2.morphologyEx(pred_mask3, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask4, cv2.MORPH_CLOSE, kernel2)
        
        # pred_mask = ObtainForeground(pred_mask)
        pred_img_name = "gt" + img_name[2:-3] + "png"
        cv2.imwrite(os.path.join(args.out_path, pred_img_name), pred_mask)

    log_file = open('Results.txt',"a")
    log_info = []
    log_info.append(f"History: {history}\n")
    log_info.append(f"VarThreshold: {varThreshold}\n")
    log_info.append(f"Learning Rate: {learningRate}\n")
    log_file.writelines(log_info)
    
    subprocess.call("python .\\eval.py --pred_path COL780-A1-Data\\ptz\\predictions --gt_path COL780-A1-Data\\ptz\\groundtruth", shell=True)

    # baseline_bgs(
    # )

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

