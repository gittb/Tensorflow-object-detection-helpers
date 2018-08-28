import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from utils import label_map_util
from utils import visualization_utils as vis_util

#This script is the complmentary script to the intial image segmenter. This script will both run predictions on all the segments then combine and save them.

# Some of the code here is taken from Author: Evan Juras's tutorial on Object Detection in Tensorflow. (Great example and introduction into object detection)
# https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
# I have adapted and reorganized some of it for my own readablity

#---------- Config ------------
segments_path = 'images_park/test/image_segments'
MODEL_NAME = 'inference_graph'
frozen_graph_file = 'frozen_inference_graph.pb'
label_path = 'training_park/labelmap.pbtxt'
save_dir = 'assembled_segments'
min_display_sort = 0.90


#---------- Classes ------------
class Segment:
    def __init__(self, link, row, col, boxes, scores, classes, num_boxes, image):
        self.link = link
        self.row = row
        self.col = col
        self.boxes = np.squeeze(boxes)
        self.scores = np.squeeze(scores)
        self.classes = np.squeeze(classes).astype(np.int32)
        self.num_boxes = num_boxes
        self.image = image


#---------- Init paths ------------
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

current_path = os.getcwd()
NUM_CLASSES = 7
PATH_TO_CKPT = os.path.join(current_path,MODEL_NAME, frozen_graph_file)
PATH_TO_LABELS = os.path.join(current_path, label_path)
PATH_TO_IMAGE = os.path.join(current_path,segments_path)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#---------- Init TF graph and session ------------
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


#---------- Functions ------------
def run_image(image_dir):
    seg_image = cv2.imread(segments_path + '/' + image_dir)
    print(seg_image.shape)
    y, x, _ = seg_image.shape
    image_expanded = np.expand_dims(seg_image, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    return boxes, scores, classes, num, seg_image, x, y


def manipulate(fparams):
    boxes = []
    scores = []
    classes = []
    seg_x = fparams['x']
    seg_y = fparams['y']
    cols = fparams['cols']
    rows = fparams['rows']
    full_image = np.zeros((seg_y * cols, seg_x * rows, 3), np.uint8)

    for seg in fparams['segments']:
        for num in range(0,300):
            if seg.scores[num] > min_display_sort:
                seg.boxes[num][0] = (seg.boxes[num][0] / cols) + ((seg.col - 1) * .25)
                seg.boxes[num][1] = (seg.boxes[num][1] / rows) + ((seg.row - 1) * .25)
                seg.boxes[num][2] = (seg.boxes[num][2] / cols) + ((seg.col - 1) * .25)
                seg.boxes[num][3] = (seg.boxes[num][3] / rows) + ((seg.row - 1) * .25)
                boxes.append(seg.boxes[num])
                classes.append(seg.classes[num])
                scores.append(seg.scores[num])

        x1 = seg_x * (seg.row - 1)
        x2 = seg_x * seg.row
        y1 = seg_y * (seg.col - 1)
        y2 = seg_y * seg.col
        full_image[y1:y2, x1:x2] = seg.image

    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    render(boxes, scores, classes, full_image, fparams['base_name'])

def render(boxes, scores, classes, full_image, base_name):
    vis_util.visualize_boxes_and_labels_on_image_array(
        full_image,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=15,
        max_boxes_to_draw = 300,
        min_score_thresh=0.0)
    cv2.imwrite(save_dir + '/' + base_name + ".jpg", full_image)
    print("[+] Assembled and saved:", base_name)
    


#---------- File Handler ------------
filelinks = [i for i in os.listdir(segments_path)]
fparams = {'base_name' : ''}
c = 0

for flink in filelinks:
    col_num = int(flink.split('_serial_')[1].split('_')[0])
    row_num = int(flink.split('_serial_')[1].split('_')[1].split('.')[0])
    
    if fparams['base_name'] != flink.split('_serial_')[0]:
        if fparams['base_name'] != '':
            manipulate(fparams)
        
        fparams = {}
        fparams['base_name'] = flink.split('_serial_')[0]
        boxes, scores, classes, num, seg_image, fparams['x'], fparams['y'] = run_image(flink)
        fparams['segments'] = [Segment(flink, row_num, col_num, boxes, scores, classes, num, seg_image)]

    else:
        boxes, scores, classes, num, seg_image, fparams['x'], fparams['y'] = run_image(flink)
        fparams['segments'].append(Segment(flink, row_num, col_num, boxes, scores, classes, num, seg_image))
        fparams['cols'] = int(col_num)
        fparams['rows'] = int(row_num)


sess.close()