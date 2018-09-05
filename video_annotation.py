import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from utils import label_map_util
from utils import visualization_utils as vis_util

#quiets all the tensorflow debug prints
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# This script takes a video file and annotates it frame by frame assembling into a new video file.
# Decently ram friendly

# Small segments of this code are taken from Evan Juras's tutorial on Object Detection in Tensorflow. (Great example and introduction into object detection)
# https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

#---------- Config ------------
video_path = 'converted_super.mp4'
MODEL_NAME = 'inference_graph'
frozen_graph_file = 'frozen_inference_graph.pb'
label_path = 'training/labelmap.pbtxt'
display_threshold = 0.75
NUM_CLASSES = 3

#---------- Init paths ------------

current_path = os.getcwd()
PATH_TO_CKPT = os.path.join(current_path,MODEL_NAME, frozen_graph_file)
PATH_TO_LABELS = os.path.join(current_path, label_path)
PATH_TO_IMAGE = os.path.join(current_path,video_path)
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
print("[+] Frozen Graph Loaded")

#---------- Functions ------------
def run_image(frame):
    y, x, _ = frame.shape
    image_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    return render(boxes, scores, classes, frame)

def render(boxes, scores, classes, full_image):
    vis_util.visualize_boxes_and_labels_on_image_array(
        full_image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        max_boxes_to_draw = 300,
        min_score_thresh=display_threshold)
    return full_image


#---------- Frame Chop, annotate, reassemble ------------

#load videos
video_in = cv2.VideoCapture(video_path)
video_out = cv2.VideoWriter('Annotated_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'),30,(1920,1080))

#process and write frames
print("[#] Annotating... \n")
frame_count = 0
success, frame = video_in.read()
while success:
    video_out.write(run_image(frame))
    frame_count += 1
    print("Frames Rendered: " + str(frame_count), end='\r')
    success, frame = video_in.read()

sess.close()
print('\n[-] Sess Closed')

video_out.release()
print('[-] Video Released')