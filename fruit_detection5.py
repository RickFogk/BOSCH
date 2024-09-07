import cv2
import numpy as np
import pyttsx3
import depthai as dai
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load YOLOv4 Tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load the COCO class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Only detect specific fruits
FRUITS = ["banana", "apple", "orange"]

# Function to detect if fruit is rotten based on color
def is_rotten(image, class_name):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if class_name == "banana":
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])
    elif class_name == "apple" or class_name == "orange":
        lower_brown = np.array([0, 0, 0])
        upper_brown = np.array([30, 255, 100])
    mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
    percentage = np.sum(mask > 0) / mask.size
    return percentage > 0.1  # Rotten if more than 10% of the fruit has brown spots

# Initialize Oak-D Lite camera with RGB and depth streams
pipeline = dai.Pipeline()

# Set resolution for the MonoCameras to 480_P to suppress the warning
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(416, 416)
camRgb.setInterleaved(False)

xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")

camRgb.preview.link(xoutRgb.input)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)

def get_distance(depth_frame, bbox):
    x, y, w, h = bbox
    depth_roi = depth_frame[y:y+h, x:x+w]
    valid_depths = depth_roi[depth_roi > 0]
    if valid_depths.size > 0:
        return np.median(valid_depths)
    else:
        return None

with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.get()
        inDepth = qDepth.get()

        frame = inRgb.getCvFrame()
        depth_frame = inDepth.getFrame()

        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        detections = net.forward(output_layers)

        # Analyze detections
        height, width = frame.shape[:2]
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] in FRUITS:
                    center_x, center_y, w, h = obj[0:4] * np.array([width, height, width, height])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Adjust bounding box to stay within frame boundaries
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)

                    # Extract the detected fruit's image
                    fruit_img = frame[y:y+int(h), x:x+int(w)]

                    # Check if the fruit_img is valid (non-empty) before further processing
                    if fruit_img is not None and fruit_img.size > 0:
                        # Check if the fruit is rotten
                        if is_rotten(fruit_img, classes[class_id]):
                            fruit_status = f"{classes[class_id]} is rotten"
                        else:
                            fruit_status = f"{classes[class_id]} is fresh"

                        # Estimate the distance to the fruit
                        distance = get_distance(depth_frame, (x, y, int(w), int(h)))
                        if distance is not None and distance > 0:
                            distance_text = f" at {distance/1000:.2f} meters"
                        else:
                            distance_text = ""

                        # Announce the fruit name, status, and distance
                        engine.say(f"{fruit_status}{distance_text}")
                        engine.runAndWait()

                        # Draw bounding box and label on the frame
                        cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{fruit_status}{distance_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        print(f"Skipped processing due to invalid ROI for {classes[class_id]}")

        # Show the frame
        cv2.imshow("Fruit Detection", frame)

        # Press 'q' to terminate the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
