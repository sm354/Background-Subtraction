import os
import cv2
import argparse
import numpy as np

import ipdb # remove this line before final submission

def parse_args():
    parser = argparse.ArgumentParser(description='make video using image and corresponding masks')
    parser.add_argument('--imgs_path', type=str)
    parser.add_argument('--masks_path', type=str)
    parser.add_argument('--video_path', type=str)
    args = parser.parse_args()
    return args

def makevideo(imgs_path, masks_path, video_path):
    filenames = os.listdir(masks_path)
    filenames.sort()
    video_frames = []

    for filename in filenames[470:]:
        # ipdb.set_trace()
        img =  cv2.imread(os.path.join(imgs_path, 'in'+filename[2:-3]+'jpg'))
        mask = cv2.imread(os.path.join(masks_path, filename))

        h, w, ch = img.shape
        size = (w, h)

        img = np.float32(img)
        mask = np.float32(mask)

        mask = mask / 255.
        mask = mask + ((mask == 0.) * 0.1)

        video_frame = img * mask
        video_frame = cv2.convertScaleAbs(video_frame)
        video_frames.append(video_frame)

    video_output = cv2.VideoWriter(video_path + ".mp4", cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for video_frame in video_frames:
        video_output.write(video_frame)
    video_output.release()

def main(args):
    makevideo(args.imgs_path, args.masks_path, args.video_path)
    print("video saved at %s"%(args.video_path))

if __name__ == "__main__":
    args = parse_args()
    main(args)