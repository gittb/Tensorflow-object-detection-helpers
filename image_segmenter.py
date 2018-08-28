import numpy as np
import cv2
import os
import time

#program will take a file of images and segment them based on parameters below
#segments of images will be placed within the inital image directory in the folder "image_segments"

#config
imagedir = 'images_to_chop' #change this to location of folder containing images
num_rows = 4    #number of rows of segments
num_cols = 4    #number of columns of segments

#creates holding folder
filelinks = [i for i in os.listdir("images_to_chop")]
if not os.path.exists(imagedir + '/image_segments'):
    os.makedirs(imagedir + '/image_segments')

#chopping loops
for flink in filelinks:
    img = cv2.imread('images_to_chop/' + flink)
    y, x, _ = img.shape
    print("[+] Loading image:", flink, "with size:", "X:", x, "Y:", y)
    c = 0
    for i in range(1,num_rows + 1):
        for o in range(1,num_cols + 1):
            x1 = int(x/num_rows) * (i - 1)
            x2 = int(x/num_rows) * i
            y1 = int(y/num_cols) * (o - 1)
            y2 = int(y/num_cols) * o
            crop = img[y1:y2, x1:x2]
            fname = flink.split('.')[0]
            ftype = '.' + flink.split('.')[1]
            cv2.imwrite(imagedir + '/image_segments/' + fname + str(o) + '_' + str(i) + '_' + ftype, crop)
            c += 1
            print("--- slice:", str(c), "out of:", str(num_cols * num_rows), fname + '_' + str(o) + '_' + str(i))