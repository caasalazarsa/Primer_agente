#!/bin/bash
export GLOG_minloglevel=1
for file in INPUT2/*; do
     python Scripts/image_demo.py --model Example_Models/segnet_model_driving_webdemo.prototxt --weights Example_Models/segnet_weights_driving_webdemo.caffemodel --colours Scripts/camvid12.png --input "$file"
done
