import cv2
import numpy as np
import depthai as dai
import rospy
from sensor_msgs.msg import LaserScan
import time
from threading import Thread
from queue import Queue
import cv2.aruco as aruco
from gtts import gTTS
import os
import pygame

print(f"OpenCV version: {cv2.__version__}")

# Load YOLOv4 Tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load the COCO class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Only detect specific fruits
FRUITS = ["banana", "apple", "orange"]

# Obstacle detection threshold (adjusted for less sensitivity)
OBSTACLE_DISTANCE_THRESHOLD = 2.0  # 2.0 meters

# Global variables
lidar_data = []
speech_queue = Queue()
last_speech_time = 0
SPEECH_COOLDOWN = 5  # 5 seconds
last_detections = []  # Store last detections

# ArUco setup
aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters_create()

# Define a mapping of ArUco IDs to fruit stands
FRUIT_STAND_MAPPING = {
    12: "banana",  # ArUco marker with ID 12 represents the banana stand
    21: "apple"    # ArUco marker with ID 21 represents the apple stand
}

# Initialize pygame for audio playback
pygame.mixer.init()

def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    filename = "temp_speech.mp3"
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    os.remove(filename)

def lidar_callback(scan_data):
    global lidar_data
    ranges = np.array(scan_data.ranges)
    ranges = ranges[~np.isinf(ranges)]  # Remove infinite values
    lidar_data = ranges

def get_distance(depth_frame, bbox):
    x, y, w, h = bbox
    depth_roi = depth_frame[y:y+h, x:x+w]
    valid_depths = depth_roi[depth_roi > 0]
    if valid_depths.size > 0:
        return np.median(valid_depths)
    else:
        return None

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

def visualize_lidar(lidar_data, window_name="LIDAR Visualization"):
    lidar_frame = np.zeros((500, 500, 3), dtype=np.uint8)
    center = (250, 250)
    max_distance = OBSTACLE_DISTANCE_THRESHOLD * 500  # Scale factor for visualization

    for i, distance in enumerate(lidar_data):
        angle = np.radians(i * 360.0 / len(lidar_data))
        if distance > 0 and distance < OBSTACLE_DISTANCE_THRESHOLD:
            x = int(center[0] + np.cos(angle) * distance * max_distance)
            y = int(center[1] + np.sin(angle) * distance * max_distance)
            cv2.circle(lidar_frame, (x, y), 3, (0, 0, 255), -1)

    cv2.circle(lidar_frame, center, int(max_distance), (255, 0, 0), 1)
    cv2.imshow(window_name, lidar_frame)

def speech_worker():
    global last_speech_time
    while True:
        message = speech_queue.get()
        current_time = time.time()
        if current_time - last_speech_time >= SPEECH_COOLDOWN:
            text_to_speech(message)
            last_speech_time = current_time
        speech_queue.task_done()

def detect_aruco_markers(frame, depth_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is not None:
        for i, corner in zip(ids, corners):
            if i[0] in FRUIT_STAND_MAPPING:
                center = np.mean(corner[0], axis=0).astype(int)
                stand_type = FRUIT_STAND_MAPPING[i[0]]
                distance = get_distance(depth_frame, (center[0]-25, center[1]-25, 50, 50))
                if distance:
                    direction = "right" if center[0] > frame.shape[1] / 2 else "left"
                    message = f"{stand_type} stand detected {distance/1000:.2f} meters ahead on the {direction} side"
                    speech_queue.put(message)
                    cv2.putText(frame, f"{stand_type} stand", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                aruco.drawDetectedMarkers(frame, [corner], [i])
    
    return frame

def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def main():
    # Initialize the ROS node
    rospy.init_node('lidar_fruit_detection', anonymous=True)
    rospy.Subscriber("/scan", LaserScan, lidar_callback)

    # Start speech worker thread
    speech_thread = Thread(target=speech_worker, daemon=True)
    speech_thread.start()

    # Initialize Oak-D Lite camera with RGB and depth streams
    pipeline = dai.Pipeline()

    # Set up camera pipeline
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

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        frame_count = 0
        start_time = time.time()
        while not rospy.is_shutdown():
            inRgb = qRgb.get()
            inDepth = qDepth.get()

            frame = inRgb.getCvFrame()
            depth_frame = inDepth.getFrame()

            # Detect ArUco markers
            frame = detect_aruco_markers(frame, depth_frame)

            # Perform YOLO inference every 5 frames
            if frame_count % 5 == 0:
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                output_layers = get_output_layers(net)
                detections = net.forward(output_layers)

                current_detections = []
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

                            x = max(0, x)
                            y = max(0, y)
                            w = min(w, width - x)
                            h = min(h, height - y)

                            current_detections.append((x, y, int(w), int(h), classes[class_id]))

                last_detections = current_detections

            # Process and display detections
            for det in last_detections:
                x, y, w, h, class_name = det
                fruit_img = frame[y:y+h, x:x+w]

                if fruit_img is not None and fruit_img.size > 0:
                    fruit_status = f"{class_name} is {'rotten' if is_rotten(fruit_img, class_name) else 'fresh'}"
                    distance = get_distance(depth_frame, (x, y, w, h))
                    distance_text = f" at {distance/1000:.2f} meters" if distance is not None and distance > 0 else ""

                    message = f"{fruit_status}{distance_text}"
                    speech_queue.put(message)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    print(f"Skipped processing due to invalid ROI for {class_name}")

            cv2.imshow("Fruit Detection", frame)

            # Visualize LIDAR data every 10 frames
            if frame_count % 10 == 0 and len(lidar_data) > 0:
                visualize_lidar(lidar_data, "LIDAR Visualization")

            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                print(f"FPS: {fps:.2f}")
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()