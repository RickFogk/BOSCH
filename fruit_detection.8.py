import cv2
import numpy as np
import depthai as dai
import rospy
from sensor_msgs.msg import LaserScan
import time
from threading import Thread
from queue import Queue
from gtts import gTTS
import os
import pygame
import apriltag

print(f"OpenCV version: {cv2.__version__}")

# Load YOLOv4 Tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load the COCO class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Detect specific fruits and people
FRUITS = ["banana", "apple", "orange"]
DETECTION_CLASSES = FRUITS + ["person"]

# Obstacle detection parameters
OBSTACLE_DISTANCE_THRESHOLD = 0.5  # 0.5 meters
DETECTION_ANGLE_RANGE = 90  # Total degrees to consider (45 degrees each side)

# Global variables
lidar_data = []
speech_queue = Queue()
last_speech_time = 0
SPEECH_COOLDOWN = 3  # 3 seconds to allow more frequent alerts
last_detections = []  # Store last detections
last_obstacle_distance = None  # Track the last obstacle distance for alert control
last_person_distance = None  # Track last detected person distance
last_person_alert_time = 0  # Time of last person alert
last_fruit_alert_time = 0  # Time of last fruit alert
last_fruit_detection_time = {}  # Track last alert time per fruit type

# Define a mapping of AprilTag IDs to fruit stands
FRUIT_STAND_MAPPING = {
    12: "banana",  # AprilTag with ID 12 represents the banana stand
    21: "apple"    # AprilTag with ID 21 represents the apple stand
}

# Initialize pygame for audio playback
pygame.mixer.init()

# Mapping variables
MAP_SIZE = 1000  # Size of the map in pixels
MAP_RESOLUTION = 0.05  # Each pixel represents 5cm
occupancy_grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
robot_position = (MAP_SIZE // 2, MAP_SIZE // 2)  # Start the robot at the center of the map

def text_to_speech(text):
    """Convert text to speech using gTTS and play it using pygame."""
    tts = gTTS(text=text, lang='en', slow=False)
    filename = "temp_speech.mp3"
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    os.remove(filename)

def lidar_callback(scan_data):
    """Callback function for receiving LIDAR data."""
    global lidar_data
    lidar_data = scan_data.ranges

def get_distance(depth_frame, bbox):
    """Calculate the median distance from the depth frame within a bounding box."""
    x, y, w, h = bbox
    depth_roi = depth_frame[y:y+h, x:x+w]
    valid_depths = depth_roi[depth_roi > 0]
    if valid_depths.size > 0:
        return np.median(valid_depths)
    else:
        return None

def is_rotten(image, class_name):
    """Determine if a fruit is rotten based on its color."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if class_name == "banana":
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])
    elif class_name == "apple" or class_name == "orange":
        lower_brown = np.array([0, 0, 0])
        upper_brown = np.array([30, 255, 100])
    else:
        return False

    mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
    percentage = np.sum(mask > 0) / mask.size
    return percentage > 0.1  # Rotten if more than 10% of the fruit has brown spots

def detect_obstacle(lidar_data):
    """Detect obstacles based on LIDAR data within a limited field of view."""
    if not lidar_data:
        return False, None, None

    num_points = len(lidar_data)
    center_index = num_points // 2
    points_per_degree = num_points / 360

    # Calculate start and end indices for the limited field of view (45 degrees each side)
    start_index = int(center_index - (DETECTION_ANGLE_RANGE / 2) * points_per_degree)
    end_index = int(center_index + (DETECTION_ANGLE_RANGE / 2) * points_per_degree)

    # Ensure indices are within bounds
    start_index = max(0, start_index)
    end_index = min(num_points - 1, end_index)

    # Get ranges within the field of view
    limited_ranges = lidar_data[start_index:end_index]

    # Find minimum distance in the limited field of view
    min_distance = min(limited_ranges) if limited_ranges else float('inf')

    # Determine if an obstacle is detected within the limited field of view
    if min_distance < OBSTACLE_DISTANCE_THRESHOLD:
        if start_index <= num_points // 2:
            return True, min_distance, "right"
        else:
            return True, min_distance, "left"
    
    return False, None, None

def visualize_lidar(lidar_data, window_name="LIDAR Visualization"):
    """Visualize LIDAR data on a standard window size."""
    if not lidar_data:
        return

    lidar_frame = np.zeros((500, 500, 3), dtype=np.uint8)  # Revert to original window size
    center = (250, 250)
    max_distance = OBSTACLE_DISTANCE_THRESHOLD * 500  # Scale factor for visualization

    num_points = len(lidar_data)
    for i, distance in enumerate(lidar_data):
        angle = i * 2 * np.pi / num_points
        if distance > 0 and distance < OBSTACLE_DISTANCE_THRESHOLD:
            x = int(center[0] + np.cos(angle) * distance * max_distance)
            y = int(center[1] + np.sin(angle) * distance * max_distance)
            cv2.circle(lidar_frame, (x, y), 3, (0, 0, 255), -1)

    # Visualize the detection range
    cv2.ellipse(lidar_frame, center, (int(max_distance), int(max_distance)), 0, 
                -45, 45, (0, 255, 0), 2)  # Limit visualization to 90 degrees

    cv2.circle(lidar_frame, center, int(max_distance), (255, 0, 0), 1)
    cv2.imshow(window_name, lidar_frame)

def speech_worker():
    """Background thread to handle text-to-speech in a synchronized manner."""
    global last_speech_time
    while True:
        message = speech_queue.get()
        current_time = time.time()
        if current_time - last_speech_time >= SPEECH_COOLDOWN:
            text_to_speech(message)
            last_speech_time = current_time
        speech_queue.task_done()

def detect_apriltags(frame, depth_frame, tag_detector):
    """Detect AprilTags and determine the distance to them."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = tag_detector.detect(gray)

    for tag in tags:
        tag_id = tag.tag_id
        if tag_id in FRUIT_STAND_MAPPING:
            center = np.mean(tag.corners, axis=0).astype(int)
            stand_type = FRUIT_STAND_MAPPING[tag_id]
            distance = get_distance(depth_frame, (center[0] - 25, center[1] - 25, 50, 50))
            if distance:
                direction = "right" if center[0] > frame.shape[1] / 2 else "left"
                message = f"{stand_type} stand detected {distance / 1000:.2f} meters ahead on the {direction} side"
                speech_queue.put(message)
                cv2.putText(frame, f"{stand_type} stand", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for corner in tag.corners:
                cv2.circle(frame, tuple(corner), 5, (0, 255, 0), -1)
            cv2.polylines(frame, [tag.corners.astype(int)], True, (0, 255, 0), 2)

    return frame

def get_output_layers(net):
    """Get the output layers of the YOLO model."""
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def update_map(lidar_data):
    """Update the occupancy grid map based on LIDAR data."""
    global occupancy_grid, robot_position
    
    if not lidar_data:
        return

    # Clear the area around the robot
    cv2.circle(occupancy_grid, robot_position, 10, 0, -1)

    for i, distance in enumerate(lidar_data):
        if distance > 0 and distance < 10:  # Limit the max range to 10 meters
            angle = i * 2 * np.pi / len(lidar_data)
            x = int(robot_position[0] + distance * np.cos(angle) / MAP_RESOLUTION)
            y = int(robot_position[1] + distance * np.sin(angle) / MAP_RESOLUTION)
            
            # Ensure the point is within the map bounds
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                occupancy_grid[y, x] = 255  # Mark as occupied

def visualize_map():
    """Visualize the occupancy grid map."""
    map_vis = cv2.cvtColor(occupancy_grid, cv2.COLOR_GRAY2BGR)
    # Draw robot position
    cv2.circle(map_vis, robot_position, 5, (0, 0, 255), -1)
    cv2.imshow("Map", map_vis)

def main():
    """Main function to run the fruit, person, and obstacle detection system."""
    global last_obstacle_distance, last_person_distance, last_fruit_alert_time, last_person_alert_time
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

    # Initialize AprilTag detector
    tag_detector = apriltag.Detector()

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

            # Detect obstacles
            obstacle_detected, obstacle_distance, deviation_direction = detect_obstacle(lidar_data)
            if obstacle_detected:
                # Only alert if the difference in distance is greater than 0.5 meters
                if last_obstacle_distance is None or abs(obstacle_distance - last_obstacle_distance) > 0.5:
                    message = f"Obstacle detected {obstacle_distance:.2f} meters ahead. Deviate {deviation_direction}."
                    speech_queue.put(message)
                    last_obstacle_distance = obstacle_distance

            # Update and visualize the map
            update_map(lidar_data)
            visualize_map()

            # Detect AprilTags
            frame = detect_apriltags(frame, depth_frame, tag_detector)

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
                        if confidence > 0.6 and classes[class_id] in DETECTION_CLASSES:  # Include "person" class
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
                    current_time = time.time()
                    if class_name in FRUITS:  # Handle fruit detection
                        fruit_status = f"{class_name} is {'rotten' if is_rotten(fruit_img, class_name) else 'fresh'}"
                        distance = get_distance(depth_frame, (x, y, w, h))
                        distance_text = f" at {distance/1000:.2f} meters" if distance is not None and distance > 0 else ""

                        if current_time - last_fruit_alert_time > 10:
                            message = f"{fruit_status}{distance_text}"
                            speech_queue.put(message)
                            last_fruit_alert_time = current_time

                        color = (0, 255, 0) if 'fresh' in fruit_status else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    elif class_name == "person":  # Handle person detection
                        person_distance = get_distance(depth_frame, (x, y, w, h))
                        if person_distance and (current_time - last_person_alert_time > 10):
                            person_message = f"Person detected at {person_distance/1000:.2f} meters ahead."
                            speech_queue.put(person_message)
                            last_person_distance = person_distance
                            last_person_alert_time = current_time
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, f"Person at {person_distance/1000:.2f} meters", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    print(f"Skipped processing due to invalid ROI for {class_name}")

            cv2.imshow("Fruit and Person Detection", frame)

            # Visualize LIDAR data every frame
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