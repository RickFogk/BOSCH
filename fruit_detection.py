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
person_approaching_stand = False  # Flag to indicate if a person is approaching the fruit stand

def initialize_pygame():
    try:
        pygame.init()
        pygame.mixer.init()
        print("Pygame initialized successfully.")
    except pygame.error:
        print("Failed to initialize Pygame. Audio feedback will be disabled.")

def text_to_speech(text):
    """Convert text to speech using gTTS and play it using pygame."""
    try:
        # Replace 'apple' with 'maçã' for TTS
        text = text.replace('apple', 'maçã')
        tts = gTTS(text=text, lang='pt', slow=False)
        filename = "temp_speech.mp3"
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove(filename)
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")

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

def classify_fruit_state(image, class_name):
    """Determine the state of a fruit based on its color and texture."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if class_name == "banana":
        # Define color ranges for banana states
        unripe_range = ([20, 100, 100], [30, 255, 255])  # Green
        good_range = ([22, 60, 200], [60, 255, 255])  # Yellow
        ripe_range = ([10, 100, 100], [20, 255, 200])  # Orange-yellow
        rotten_range = ([0, 100, 20], [10, 255, 100])  # Brown
    elif class_name == "apple":
        # Define color ranges for apple states
        unripe_range = ([35, 100, 100], [85, 255, 255])  # Green
        good_range = ([0, 100, 100], [10, 255, 255])  # Red
        ripe_range = ([170, 100, 100], [180, 255, 255])  # Deep Red
        rotten_range = ([20, 100, 20], [30, 255, 100])  # Brown
    elif class_name == "orange":
        # Define color ranges for orange states
        unripe_range = ([35, 100, 100], [85, 255, 255])  # Green
        good_range = ([10, 100, 100], [25, 255, 255])  # Orange
        ripe_range = ([5, 100, 100], [15, 255, 200])  # Deep Orange
        rotten_range = ([0, 0, 0], [180, 255, 30])  # Dark spots (potential fungus)

    # Calculate percentage of pixels in each state
    unripe_mask = cv2.inRange(hsv_image, np.array(unripe_range[0]), np.array(unripe_range[1]))
    good_mask = cv2.inRange(hsv_image, np.array(good_range[0]), np.array(good_range[1]))
    ripe_mask = cv2.inRange(hsv_image, np.array(ripe_range[0]), np.array(ripe_range[1]))
    rotten_mask = cv2.inRange(hsv_image, np.array(rotten_range[0]), np.array(rotten_range[1]))

    total_pixels = image.shape[0] * image.shape[1]
    unripe_percentage = np.sum(unripe_mask > 0) / total_pixels
    good_percentage = np.sum(good_mask > 0) / total_pixels
    ripe_percentage = np.sum(ripe_mask > 0) / total_pixels
    rotten_percentage = np.sum(rotten_mask > 0) / total_pixels

    # Determine the fruit state based on the highest percentage
    max_percentage = max(unripe_percentage, good_percentage, ripe_percentage, rotten_percentage)
    if max_percentage == unripe_percentage:
        return "verde"
    elif max_percentage == good_percentage:
        return "boa"
    elif max_percentage == ripe_percentage:
        return "madura"
    else:
        return "podre"

def detect_obstacle(lidar_data, depth_frame):
    """Detect obstacles based on LIDAR data and depth frame within a limited field of view."""
    if not lidar_data or depth_frame is None:
        return False, None, None

    # Process LIDAR data
    num_points = len(lidar_data)
    center_index = num_points // 2
    points_per_degree = num_points / 360

    start_index = int(center_index - (DETECTION_ANGLE_RANGE / 2) * points_per_degree)
    end_index = int(center_index + (DETECTION_ANGLE_RANGE / 2) * points_per_degree)

    start_index = max(0, start_index)
    end_index = min(num_points - 1, end_index)

    limited_ranges = lidar_data[start_index:end_index]

    lidar_min_distance = min(limited_ranges) if limited_ranges else float('inf')

    # Process depth frame
    depth_roi = depth_frame[depth_frame.shape[0]//3:2*depth_frame.shape[0]//3, :]
    valid_depths = depth_roi[depth_roi > 0]
    depth_min_distance = np.min(valid_depths) / 1000 if valid_depths.size > 0 else float('inf')  # Convert to meters

    # Combine LIDAR and depth data
    min_distance = min(lidar_min_distance, depth_min_distance)

    if min_distance < OBSTACLE_DISTANCE_THRESHOLD:
        if lidar_min_distance < depth_min_distance:
            # LIDAR detected the closer obstacle
            if start_index <= num_points // 2:
                return True, min_distance, "direita"
            else:
                return True, min_distance, "esquerda"
        else:
            # Depth camera detected the closer obstacle
            obstacle_x = np.argmin(depth_roi)
            if obstacle_x < depth_roi.shape[1] // 2:
                return True, min_distance, "direita"
            else:
                return True, min_distance, "esquerda"
    
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
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def main():
    """Main function to run the fruit, person, and obstacle detection system."""
    global last_obstacle_distance, last_person_distance, last_fruit_alert_time, last_person_alert_time, person_approaching_stand

    try:
        print("Initializing ROS node...")
        rospy.init_node('lidar_fruit_detection', anonymous=True)
        rospy.Subscriber("/scan", LaserScan, lidar_callback)
        rospy.Subscriber("/map", OccupancyGrid, occupancy_grid_callback)
        print("ROS node initialized and subscribed to topics.")

        print("Initializing Pygame...")
        initialize_pygame()

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

                # Detect obstacles using combined LIDAR and depth data
                obstacle_detected, obstacle_distance, deviation_direction = detect_obstacle(lidar_data, depth_frame)
                if obstacle_detected and not person_approaching_stand:
                    if last_obstacle_distance is None or abs(obstacle_distance - last_obstacle_distance) > 0.5:
                        message = f"Obstáculo detectado a {obstacle_distance:.2f} metros à frente. Desvie para a {deviation_direction}."
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
                    try:
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
                    except Exception as e:
                        print(f"Error during YOLO inference: {str(e)}")

                # Process and display detections
                person_detected = False
                for det in last_detections:
                    x, y, w, h, class_name = det
                    roi = frame[y:y+h, x:x+w]

                    if roi is not None and roi.size > 0:
                        current_time = time.time()
                        if class_name in FRUITS:  # Handle fruit detection
                            fruit_state = classify_fruit_state(roi, class_name)
                            distance = get_distance(depth_frame, (x, y, w, h))
                            distance_text = f" a {distance/1000:.2f} metros" if distance is not None and distance > 0 else ""

                            if current_time - last_fruit_alert_time > 10:
                                message = f"{class_name} {fruit_state}{distance_text}"
                                speech_queue.put(message)
                                last_fruit_alert_time = current_time

                            color = (0, 255, 0) if fruit_state in ["boa", "madura"] else (0, 0, 255)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, f"{class_name} {fruit_state}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        elif class_name == "person":  # Handle person detection
                            person_detected = True
                            person_distance = get_distance(depth_frame, (x, y, w, h))
                            if person_distance and (current_time - last_person_alert_time > 10):
                                person_message = f"Pessoa detectada a {person_distance/1000:.2f} metros à frente."
                                speech_queue.put(person_message)
                                last_person_distance = person_distance
                                last_person_alert_time = current_time
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(frame, f"Pessoa a {person_distance/1000:.2f} metros", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        print(f"Skipped processing due to invalid ROI for {class_name}")

                # Update person_approaching_stand flag
                if person_detected and last_person_distance is not None and last_person_distance < 2000:  # 2 meters
                    person_approaching_stand = True
                else:
                    person_approaching_stand = False

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