import cv2
import numpy as np
import depthai as dai
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import time
from threading import Thread
from queue import Queue
from gtts import gTTS
import os
import pygame
import apriltag

print(f"OpenCV version: {cv2.__version__}")
print(f"DepthAI version: {dai.__version__}")

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
occupancy_grid = None
speech_queue = Queue()
last_speech_time = 0
SPEECH_COOLDOWN = 3  # 3 seconds to allow more frequent alerts
last_detections = []  # Store last detections
last_obstacle_distance = None  # Track the last obstacle distance for alert control
last_person_distance = None  # Track last detected person distance
last_person_alert_time = 0  # Time of last person alert
last_fruit_alert_time = 0  # Time of last fruit alert
last_fruit_detection_time = {}  # Track last alert time per fruit type

# Initialize pygame for audio playback
pygame.mixer.init()

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

def occupancy_grid_callback(grid_msg):
    """Callback function for receiving occupancy grid data."""
    global occupancy_grid
    occupancy_grid = np.array(grid_msg.data).reshape((grid_msg.info.height, grid_msg.info.width))

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

    start_index = int(center_index - (DETECTION_ANGLE_RANGE / 2) * points_per_degree)
    end_index = int(center_index + (DETECTION_ANGLE_RANGE / 2) * points_per_degree)

    start_index = max(0, start_index)
    end_index = min(num_points - 1, end_index)

    limited_ranges = lidar_data[start_index:end_index]

    min_distance = min(limited_ranges) if limited_ranges else float('inf')

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

    lidar_frame = np.zeros((500, 500, 3), dtype=np.uint8)
    center = (250, 250)
    max_distance = OBSTACLE_DISTANCE_THRESHOLD * 500

    num_points = len(lidar_data)
    for i, distance in enumerate(lidar_data):
        angle = i * 2 * np.pi / num_points
        if distance > 0 and distance < OBSTACLE_DISTANCE_THRESHOLD:
            x = int(center[0] + np.cos(angle) * distance * max_distance)
            y = int(center[1] + np.sin(angle) * distance * max_distance)
            cv2.circle(lidar_frame, (x, y), 3, (0, 0, 255), -1)

    cv2.ellipse(lidar_frame, center, (int(max_distance), int(max_distance)), 0, 
                -45, 45, (0, 255, 0), 2)

    cv2.circle(lidar_frame, center, int(max_distance), (255, 0, 0), 1)
    cv2.imshow(window_name, lidar_frame)

def visualize_slam_map(occupancy_grid, window_name="SLAM Map"):
    """Visualize the SLAM map."""
    if occupancy_grid is None:
        return
    
    # Convert occupancy grid to a grayscale image for display
    map_img = (occupancy_grid + 1) * 127  # Convert [-1, 100] to [0, 255]
    map_img = np.uint8(map_img)
    
    # Resize the image for better display
    map_img_resized = cv2.resize(map_img, (500, 500), interpolation=cv2.INTER_NEAREST)
    
    # Show the map
    cv2.imshow(window_name, map_img_resized)

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

def get_output_layers(net):
    """Get the output layers of the YOLO model."""
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def main():
    """Main function to run the fruit, person, and obstacle detection system."""
    global last_obstacle_distance, last_person_distance, last_fruit_alert_time, last_person_alert_time

    try:
        print("Initializing ROS node...")
        rospy.init_node('lidar_fruit_detection', anonymous=True)
        rospy.Subscriber("/scan", LaserScan, lidar_callback)
        rospy.Subscriber("/map", OccupancyGrid, occupancy_grid_callback)
        print("ROS node initialized and subscribed to topics.")

        print("Starting speech worker thread...")
        speech_thread = Thread(target=speech_worker, daemon=True)
        speech_thread.start()
        print("Speech worker thread started.")

        print("Initializing Oak-D Lite camera...")
        pipeline = dai.Pipeline()

        # Set up camera pipeline
        camRgb = pipeline.createColorCamera()
        camRgb.setPreviewSize(416, 416)
        camRgb.setInterleaved(False)

        monoLeft = pipeline.createMonoCamera()
        monoRight = pipeline.createMonoCamera()
        stereo = pipeline.createStereoDepth()

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        # IMU setup
        imu = pipeline.createIMU()
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 400)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)

        xoutRgb = pipeline.createXLinkOut()
        xoutDepth = pipeline.createXLinkOut()
        xoutImu = pipeline.createXLinkOut()

        xoutRgb.setStreamName("rgb")
        xoutDepth.setStreamName("depth")
        xoutImu.setStreamName("imu")

        camRgb.preview.link(xoutRgb.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        stereo.depth.link(xoutDepth.input)
        imu.out.link(xoutImu.input)

        print("Camera pipeline set up.")

        print("Initializing AprilTag detector...")
        tag_detector = apriltag.Detector()
        print("AprilTag detector initialized.")

        print("Starting main loop...")
        with dai.Device(pipeline) as device:
            print("Connected to device.")
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            qImu = device.getOutputQueue(name="imu", maxSize=50, blocking=False)

            frame_count = 0
            start_time = time.time()
            while not rospy.is_shutdown():
                inRgb = qRgb.get()
                inDepth = qDepth.get()
                inImu = qImu.get()

                frame = inRgb.getCvFrame()
                depth_frame = inDepth.getFrame()

                # Process IMU data
                imuData = inImu.getData()
                for imuPacket in imuData:
                    acceleroValues = imuPacket.acceleroMeter
                    gyroValues = imuPacket.gyroscope
                    
                    imu_frame = np.zeros((200, 300, 3), dtype=np.uint8)
                    cv2.putText(imu_frame, f"Accel: {acceleroValues.x:.2f}, {acceleroValues.y:.2f}, {acceleroValues.z:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imu_frame, f"Gyro: {gyroValues.x:.2f}, {gyroValues.y:.2f}, {gyroValues.z:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow("IMU Data", imu_frame)

                # Detect obstacles using LIDAR
                obstacle_detected, obstacle_distance, deviation_direction = detect_obstacle(lidar_data)
                if obstacle_detected:
                    if last_obstacle_distance is None or abs(obstacle_distance - last_obstacle_distance) > 0.5:
                        message = f"Obstacle detected {obstacle_distance:.2f} meters ahead. Deviate {deviation_direction}."
                        speech_queue.put(message)
                        last_obstacle_distance = obstacle_distance

                # Visualizations
                visualize_lidar(lidar_data, "LIDAR Visualization")
                if occupancy_grid is not None:
                    visualize_slam_map(occupancy_grid, "SLAM Map")

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow("Stereo Depth", depth_colormap)

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
                            if confidence > 0.6 and classes[class_id] in DETECTION_CLASSES:
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
                            cv2.putText(frame, fruit_status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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

                frame_count += 1
                if frame_count % 30 == 0:
                    end_time = time.time()
                    fps = 30 / (end_time - start_time)
                    print(f"FPS: {fps:.2f}")
                    start_time = time.time()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
