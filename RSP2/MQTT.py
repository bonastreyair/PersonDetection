ç#!/usr&bin/env python3
# USAGE
# python3 deep_learning_object_detection.py --image images/example_01.jpg \
#   --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import time
import numpy as np
import argparse
import cv2
import picamera

import RPi.GPIO as GPIO
import time
import paho.mqtt.client as mqtt

PIR_PIN = 17
PORT_MQTT = 1883

BROKER = "10.12.13.62"
TOPIC_PIR = "RSP2/PIR/Presencia"
TOPIC_PER = "RSP2/CAM/Personas"

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

#ap.add_argument("-i", "--image", required=True,
#    help="path to input image")
ap.add_argument("-p", "--prototxt", required=False, default='MobileNetSSD_deploy.prototxt.txt',
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False, default='MobileNetSSD_deploy.caffemodel',
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it

mqttc = mqtt.Client()
mqttc.connect(BROKER, PORT_MQTT, 60)

def make_photo():
    
    width = 1280
    height = 1280

    with picamera.PiCamera() as picam:
        picam.resolution = (width, height)
        picam.start_preview()
        time.sleep(1)
        image = np.empty((height, width, 3), dtype=np.uint8)
        picam.capture(image, 'bgr')
        image = cv2.flip(image, -1)
    
    return image
    
    
def count_people(image):
    person = 0
    # (note: normalization is done via the authors of the MobileNet SSD implementation)
    #image = cv2.imread(args["image"])
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and predictions
    print("[INFO] computing object detections...")
    start_time = time.time()
    net.setInput(blob)
    detections = net.forward()
    end_time = time.time()
    print(f"[INFO] computing time {(end_time-start_time)*1000:3.0f}ms")
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            if CLASSES[idx] == 'person':
                person += 1
                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    print("[INFO] Ha detectado " + str(person) + " personas")

    return person  

while True:
    mqttc.publish(TOPIC_PIR, GPIO.input(PIR_PIN))    
    
    if GPIO.input(PIR_PIN):
        
        for i in range(4): # takes 4 photos
            image = make_photo()
            
            people = count_people(image)
            
            image_to_save = image.copy()
            
        cv2.imwrite("image.jpg", image_to_save)

        mqttc.publish(TOPIC_PER, max_people)
        print("[INFO] Finalmente hay " + str(max_people) + " personas")
        time.sleep(10) # sleeps 10 seconds
        continue
    
    if GPIO.wait_for_edge(PIR_PIN, GPIO.RISING, timeout=10000):
        print('Movement detected!')
    else:
        print('No movement..Comprovando personas..')
        image = make_photo()
        people = count_people(image)
        image_to_save = image.copy()
        
        cv2.imwrite("image.jpg", image_to_save)

        mqttc.publish(TOPIC_PER, people)
        print("[INFO] Finalmente hay " + str(people) + " personas")
