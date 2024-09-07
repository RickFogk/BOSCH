import cv2
import numpy as np
import depthai as dai
import os
import time

# Change this to your darknet directory
darknet_path = "/home/rickfogk/darknet"
os.chdir(darknet_path)

# Verify coco.names file
coco_names_path = os.path.join(darknet_path, "data", "coco.names")
if not os.path.exists(coco_names_path):
    raise FileNotFoundError(f"coco.names not found at {coco_names_path}")

# Load YOLOv4-tiny
weights_path = os.path.join(darknet_path, "yolov4-tiny.weights")
cfg_path = os.path.join(darknet_path, "cfg", "yolov4-tiny.cfg")

if not os.path.exists(weights_path) or not os.path.exists(cfg_path):
    raise FileNotFoundError(f"YOLOv4-tiny weights or cfg file not found")

net = cv2.dnn.readNet(weights_path, cfg_path)

# Check CUDA availability
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using CUDA")
else:
    print("CUDA not available, using CPU")

# Load class names
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

print(f"Loaded {len(classes)} classes")

# Specify the classes we're interested in
target_classes = ["banana", "apple", "orange"]

# Verify target classes are in the loaded classes
for target in target_classes:
    if target not in classes:
        print(f"Warning: {target} not in coco.names")

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Create pipeline for OAK-D Lite
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.createColorCamera()
xoutRgb = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(640, 480)  # Increased resolution
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Linking
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras:', device.getConnectedCameras())
   
    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.get()  # Blocking call, will wait until a new data has arrived
        frame = inRgb.getCvFrame()
       
        # Preprocessing
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

        height, width = frame.shape[:2]

        # Detect objects
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        outs = net.forward(output_layers)
        end = time.time()

        # Information to display on frame
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:  # Lower threshold for debugging
                    print(f"Detected {classes[class_id]} with confidence {confidence:.2f}")
                if confidence > 0.3 and classes[class_id] in target_classes:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

        # Draw bounding boxes and labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                if label in target_classes:
                    color = (0, 255, 0)  # Green for target fruits
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # Improved label positioning and appearance
                    label_size, base_line = cv2.getTextSize(f"{label} {confidences[i]:.2f}", font, 0.5, 1)
                    y = max(y, label_size[1])
                    cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y + base_line - 10), color, cv2.FILLED)
                    cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y - 7), font, 0.5, (0, 0, 0), 1)

        # Display FPS
        fps = f"FPS: {1 / (end - start):.2f}"
        cv2.putText(frame, fps, (10, 30), font, 1, (0, 0, 255), 2)

        # Display output in a larger window
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 1280, 960)
        cv2.imshow("Object Detection", frame)

        # Display preprocessed image
        cv2.namedWindow("Preprocessed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preprocessed", 640, 480)
        cv2.imshow("Preprocessed", cv2.resize(blob[0].transpose(1,2,0), (640, 480)))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
