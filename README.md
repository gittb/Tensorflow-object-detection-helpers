# Tensorflow-object-detection-helpers
scripts that will assist in processing of images

--------------------------
Image segmenter:
This program is designed to be used to when images are larger than the input of pretrained object detection models and you would like to segment the larger image to preserver quality.

Image assembler:
Will take the segments created with the image segmentor above, annotate, reassemble, and save.

Video Annotation:
Accepts input video (codex must be accepted by opencv) then annotates frame by frame rewritting to new video file.

Within each program there is a configuration section which you will need to put in your own params to make it work accordingly.


---------------------------
All programs designed to work with Object Detection API from tensorflow
