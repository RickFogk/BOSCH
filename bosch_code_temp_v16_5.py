import cv2
import numpy as np
import depthai as dai
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Vector3
import time
from threading import Thread, Lock, Event
from queue import Queue
from gtts import gTTS
import os
import pygame
import logging
import csv
from datetime import datetime
from adafruit_servokit import ServoKit
import math
import sys
import speech_recognition as sr
from pynput import keyboard

# ===========================
# Configuration
# ===========================

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
lidar_data = []
occupancy_grid = None
acceleration_data = Vector3()
cart_velocity = Vector3()
speech_queue = Queue()
last_speech_time = 0
last_navigation_update_time = 0
initial_obstacle_check_done = False
running = True
sleep_mode = True
last_fruit_alert_time = 0
last_fruit_detection_time = 0
sweeping = False
fine_tuning = False
frames_without_detection = 0
current_pan = 30
current_tilt = 10
fruit_last_seen_time = None
fruit_state_analyzed = False
fruit_centered = False
fruit_processed = False
selected_fruit_index = None
last_reported_distance = None
last_reported_angle = None
fruit_position = None

# Constants
SPEECH_COOLDOWN = 3
OBSTACLE_DISTANCE_THRESHOLD = 1.0  # meters
OBSTACLE_ANGLE_THRESHOLD = 30  # degrees
NAVIGATION_UPDATE_INTERVAL = 2  # seconds
MOVEMENT_THRESHOLD = 0.05  # m/s, threshold to consider the cart moving
PAN_DEFAULT, TILT_DEFAULT = 30, 10
PAN_MIN, PAN_MAX = 0, 90
TILT_MIN, TILT_MAX = 10, 90
SWEEP_PAN_STEP = 5
SWEEP_TILT_STEP = 5
SWEEP_DELAY = 0.01
FINE_TUNE_STEP = 1
FINE_TUNE_DELAY = 0.05
CONFIDENCE_THRESHOLD = 0.3
FRUIT_LOSS_TIMEOUT = 10

# YOLO Configuration Files
YOLO_WEIGHTS = "yolov4-tiny.weights"
YOLO_CONFIG = "yolov4-tiny.cfg"
COCO_NAMES = "coco.names"

# Fruits of Interest
FRUITS = ["banana", "apple"]
DETECTION_CLASSES = FRUITS.copy()

# Class display names mapping
CLASS_DISPLAY_NAMES = {
    "banana": "banana",
    "apple": "maçã",
}

# Servo configuration
try:
    kit = ServoKit(channels=16)
    tilt_servo, pan_servo = 0, 1
    logging.info("ServoKit initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize ServoKit: {e}")
    sys.exit(1)

# Load YOLOv4 Tiny model
try:
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    logging.info("YOLO model loaded successfully with CUDA backend.")
except Exception as e:
    logging.error(f"Failed to load YOLO model with CUDA backend: {e}")
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logging.warning("Falling back to CPU backend for YOLO model.")
    except Exception as cpu_e:
        logging.error(f"Failed to set CPU backend for YOLO model: {cpu_e}")
        sys.exit(1)

# Load the COCO class names
try:
    with open(COCO_NAMES, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    logging.info("COCO class names loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load COCO class names: {e}")
    sys.exit(1)

# ===========================
# Functions
# ===========================

def initialize_pygame():
    try:
        pygame.init()
        pygame.mixer.init()
        logging.info("Pygame initialized successfully.")
    except pygame.error as e:
        logging.error(f"Failed to initialize Pygame: {e}")
        sys.exit(1)

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='pt', slow=False)
        filename = "temp_speech.mp3"
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove(filename)
    except Exception as e:
        logging.error(f"Error in text-to-speech: {e}")

def lidar_callback(scan_data):
    global lidar_data
    lidar_data = scan_data.ranges

def occupancy_grid_callback(grid_msg):
    global occupancy_grid
    occupancy_grid = np.array(grid_msg.data).reshape((grid_msg.info.height, grid_msg.info.width))

def acceleration_callback(msg):
    global acceleration_data, cart_velocity
    acceleration_data = msg
    # Integrate acceleration to get velocity (simplified)
    cart_velocity.x += msg.x * 0.1  # Assuming 10 Hz update rate
    cart_velocity.y += msg.y * 0.1
    cart_velocity.z += msg.z * 0.1

def is_cart_moving():
    velocity_magnitude = math.sqrt(cart_velocity.x**2 + cart_velocity.y**2 + cart_velocity.z**2)
    return velocity_magnitude > MOVEMENT_THRESHOLD

def detect_obstacles(lidar_data):
    global initial_obstacle_check_done

    if not lidar_data:
        return False, None, None

    try:
        # Find the closest obstacle
        min_distance = min(lidar_data)
        min_index = lidar_data.index(min_distance)
        angle = (min_index - len(lidar_data) // 2) * (360 / len(lidar_data))

        obstacle_detected = min_distance < OBSTACLE_DISTANCE_THRESHOLD

        if obstacle_detected and not initial_obstacle_check_done:
            initial_obstacle_check_done = True
            return True, min_distance, angle

        if is_cart_moving() and obstacle_detected:
            return True, min_distance, angle

        return False, None, None

    except Exception as e:
        logging.error(f"Error detecting obstacles: {e}")
        return False, None, None

def calculate_navigation_instruction(lidar_data):
    if not lidar_data:
        return "Não foi possível obter dados do LiDAR."

    try:
        left_sector = lidar_data[:len(lidar_data)//3]
        center_sector = lidar_data[len(lidar_data)//3:2*len(lidar_data)//3]
        right_sector = lidar_data[2*len(lidar_data)//3:]

        left_distance = min(left_sector)
        center_distance = min(center_sector)
        right_distance = min(right_sector)

        if center_distance > OBSTACLE_DISTANCE_THRESHOLD:
            return "Caminho à frente livre, continue em frente."
        elif left_distance > right_distance:
            return "Obstáculo à frente, vire levemente à esquerda."
        else:
            return "Obstáculo à frente, vire levemente à direita."

    except Exception as e:
        logging.error(f"Error calculating navigation instruction: {e}")
        return "Erro ao calcular instruções de navegação."

def speech_worker():
    global last_speech_time
    while running:
        try:
            message = speech_queue.get()
            current_time = time.time()
            if current_time - last_speech_time >= SPEECH_COOLDOWN:
                text_to_speech(message)
                last_speech_time = current_time
            speech_queue.task_done()
        except Exception as e:
            logging.error(f"Error in speech worker: {e}")

def navigation_worker():
    global last_navigation_update_time
    while running:
        current_time = time.time()
        if is_cart_moving() and current_time - last_navigation_update_time >= NAVIGATION_UPDATE_INTERVAL:
            instruction = calculate_navigation_instruction(lidar_data)
            speech_queue.put(instruction)
            last_navigation_update_time = current_time
        time.sleep(0.1)

def move_servo(servo, angle):
    try:
        if servo == pan_servo:
            angle = max(PAN_MIN, min(PAN_MAX, angle))
        elif servo == tilt_servo:
            angle = max(TILT_MIN, min(TILT_MAX, angle))
        kit.servo[servo].angle = angle
        logging.debug(f"Moved servo {servo} to angle {angle}")
    except Exception as e:
        logging.error(f"Error moving servo: {e}")

def sweep_camera():
    global sweeping, current_pan, current_tilt
    if not sweeping:
        return

    new_pan = current_pan + SWEEP_PAN_STEP
    if new_pan > PAN_MAX:
        new_pan = PAN_MIN
        new_tilt = current_tilt + SWEEP_TILT_STEP
        if new_tilt > TILT_MAX:
            new_tilt = TILT_MIN
        move_servo(tilt_servo, new_tilt)
        current_tilt = new_tilt

    move_servo(pan_servo, new_pan)
    current_pan = new_pan
    time.sleep(SWEEP_DELAY)

def fine_tune_camera(target_x, target_y, frame_width, frame_height):
    global current_pan, current_tilt

    center_x, center_y = frame_width // 2, frame_height // 2
    
    if abs(target_x - center_x) > 10:
        pan_dir = 1 if target_x > center_x else -1
        new_pan = current_pan + (FINE_TUNE_STEP * pan_dir)
        new_pan = max(PAN_MIN, min(PAN_MAX, new_pan))
        if new_pan != current_pan:
            move_servo(pan_servo, new_pan)
            current_pan = new_pan

    if abs(target_y - center_y) > 10:
        tilt_dir = 1 if target_y > center_y else -1
        new_tilt = current_tilt + (FINE_TUNE_STEP * tilt_dir)
        new_tilt = max(TILT_MIN, min(TILT_MAX, new_tilt))
        if new_tilt != current_tilt:
            move_servo(tilt_servo, new_tilt)
            current_tilt = new_tilt

    time.sleep(FINE_TUNE_DELAY)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def detect_fruits(frame):
    try:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE_THRESHOLD and classes[class_id] in DETECTION_CLASSES:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append(boxes[i] + [classes[class_ids[i]], confidences[i]])

        return detections
    except Exception as e:
        logging.error(f"Error during fruit detection: {e}")
        return []

def classify_fruit_quality(image, class_name):
    # Placeholder function - implement actual quality classification logic here
    return "bom_estado", (0, 255, 0)

def handle_fruit_detection(frame, depth_frame, detections):
    global last_fruit_alert_time, current_pan, current_tilt, sweeping, fine_tuning, frames_without_detection
    global last_fruit_detection_time, fruit_last_seen_time, fruit_state_analyzed, fruit_centered, fruit_processed
    global selected_fruit_index, last_reported_distance, last_reported_angle, fruit_position

    current_time = time.time()
    fruit_detected = False

    try:
        if detections:
            fruit_detected = True
            frames_without_detection = 0
            last_fruit_detection_time = current_time
            fruit_last_seen_time = current_time

            fruit_info_list = []
            for idx, detection in enumerate(detections):
                x, y, w, h, class_name, confidence = detection
                fruit_roi = frame[y:y+h, x:x+w]
                fruit_state, _ = classify_fruit_quality(fruit_roi, class_name)
                fruit_info_list.append({
                    "index": idx,
                    "bbox": (x, y, w, h),
                    "class_name": class_name,
                    "confidence": confidence,
                    "state": fruit_state
                })

            selected_fruit = fruit_info_list[0]  # Select the first fruit for simplicity
            selected_fruit_index = selected_fruit["index"]
            x, y, w, h = selected_fruit["bbox"]
            class_name = selected_fruit["class_name"]
            confidence = selected_fruit["confidence"]
            fruit_state = selected_fruit["state"]
            center_x = x + w // 2
            center_y = y + h // 2

            if sweeping:
                sweeping = False
                logging.info(f"{class_name} detected, centering camera.")

            fine_tune_camera(center_x, center_y, frame.shape[1], frame.shape[0])

            frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
            if abs(center_x - frame_center_x) <= 10 and abs(center_y - frame_center_y) <= 10:
                fruit_centered = True
            else:
                fruit_centered = False

            if fruit_centered:
                distance = get_distance(depth_frame, (x, y, w, h))
                if distance is not None:
                    distance_cm = distance / 10
                    angle = calculate_fruit_angle(center_x, frame.shape[1])

                    if (last_reported_distance is None or abs(distance_cm - last_reported_distance) > 20 or
                        last_reported_angle is None or abs(angle - last_reported_angle) > 15):

                        display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name)
                        direction = "direita" if angle > 0 else "esquerda"
                        message = f"{display_name} a {distance_cm:.1f} centímetros, {abs(angle):.0f} graus à {direction}."
                        
                        if current_time - last_fruit_alert_time > SPEECH_COOLDOWN:
                            speech_queue.put(message)
                            last_fruit_alert_time = current_time
                            last_reported_distance = distance_cm
                            last_reported_angle = angle

                    if not fruit_processed:
                        display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name)
                        state_message = f"A {display_name} está em {fruit_state}."
                        speech_queue.put(f"Analisando o estado da {display_name}.")
                        speech_queue.put(state_message)
                        fruit_state_analyzed = True
                        fruit_processed = True

            # Update fruit_position for potential fields navigation
            fruit_position = np.array([center_x - frame.shape[1] / 2, center_y - frame.shape[0] / 2])

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{class_name} ({fruit_state}): {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            frames_without_detection += 1
            if fruit_last_seen_time and (current_time - fruit_last_seen_time) > FRUIT_LOSS_TIMEOUT:
                if not sweeping:
                    logging.warning("Fruit lost, restarting sweep.")
                    sweeping = True
                    frames_without_detection = 0
                    fruit_centered = False
                    fruit_state_analyzed = False
                    fruit_processed = False
                    selected_fruit_index = None
                    last_reported_distance = None
                    last_reported_angle = None
                    fruit_position = None

        return frame

    except Exception as e:
        logging.error(f"Error handling fruit detection: {e}")
        return frame

def get_distance(depth_frame, bbox):
    x, y, w, h = bbox
    roi = depth_frame[y:y+h, x:x+w]
    valid_depths = roi[roi > 0]
    if len(valid_depths) > 0:
        return np.median(valid_depths)
    return None

def calculate_fruit_angle(center_x, frame_width):
    field_of_view = 62  # Adjust based on your camera's FOV
    angle_per_pixel = field_of_view / frame_width
    pixel_offset = center_x - (frame_width / 2)
    angle = pixel_offset * angle_per_pixel
    return angle

def main():
    global running, lidar_data, fruit_position

    # Initialize ROS node
    rospy.init_node('blind_guidance_system')
    rospy.Subscriber('/scan', LaserScan, lidar_callback)
    rospy.Subscriber('/imu/data', Vector3, acceleration_callback)

    # Initialize camera
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.createColorCamera()
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # Start threads
    speech_thread = Thread(target=speech_worker)
    speech_thread.start()

    navigation_thread = Thread(target=navigation_worker)
    navigation_thread.start()

    # Initialize Pygame for audio
    initialize_pygame()

    # Connect to the device and start the pipeline
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        speech_queue.put("Sistema iniciado. Pronto para guiar.")

        try:
            while running and not rospy.is_shutdown():
                in_rgb = q_rgb.get()
                frame = in_rgb.getCvFrame()

                # Perform fruit detection
                detections = detect_fruits(frame)
                frame = handle_fruit_detection(frame, frame, detections)  # Using RGB frame as depth for simplicity

                # Obstacle detection and navigation
                obstacle_detected, distance, angle = detect_obstacles(lidar_data)
                if obstacle_detected:
                    if not initial_obstacle_check_done:
                        message = f"Verificação do sistema: Obstáculo detectado a {distance:.2f} metros, ângulo {angle:.0f} graus."
                        speech_queue.put(message)
                    elif is_cart_moving():
                        message = f"Atenção: Obstáculo a {distance:.2f} metros, ângulo {angle:.0f} graus."
                        speech_queue.put(message)

                # Camera movement
                if sweeping:
                    sweep_camera()
                elif fruit_position is not None:
                    fine_tune_camera(fruit_position[0] + frame.shape[1] // 2, 
                                     fruit_position[1] + frame.shape[0] // 2, 
                                     frame.shape[1], frame.shape[0])

                cv2.imshow("Fruit Detection", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Shutting down.")
        finally:
            running = False
            speech_thread.join()
            navigation_thread.join()
            cv2.destroyAllWindows()
            logging.info("System shutdown complete.")

if __name__ == '__main__':
    main()
