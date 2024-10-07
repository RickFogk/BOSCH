import cv2
import numpy as np
import depthai as dai
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import time
from threading import Thread, Lock, Event
from queue import Queue
from gtts import gTTS
import os
import pygame
import logging
import math
import sys
import speech_recognition as sr
from pynput import keyboard

# Configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
lidar_data = []
speech_queue = Queue()
last_speech_time = 0
SPEECH_COOLDOWN = 3
running = True
sleep_mode = True
user_choice = None
user_choice_lock = Lock()
choice_event = Event()
fruit_position = None

# Constants
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
OBSTACLE_DISTANCE_THRESHOLD = 0.5  # meters
DETECTION_ANGLE_RANGE = 360  # degrees, for full LIDAR range
ATTRACTIVE_FORCE_GAIN = 1.0
REPULSIVE_FORCE_GAIN = 100.0
OBSTACLE_INFLUENCE_DISTANCE = 1.0  # meters
MAX_LINEAR_SPEED = 0.5  # m/s
MAX_ANGULAR_SPEED = 1.0  # rad/s

# YOLO Configuration Files
YOLO_WEIGHTS = "yolov4-tiny.weights"
YOLO_CONFIG = "yolov4-tiny.cfg"
COCO_NAMES = "coco.names"

# Fruits of Interest
FRUITS = ["banana", "apple"]
DETECTION_CLASSES = FRUITS.copy()

# Activation phrases
ACTIVATION_PHRASES = [
    "olá smartface", "ola smartface",
    "olá smart face", "ola smart face",
    "hello smartface", "hello smart face",
]

# Initialize YOLOv4-tiny
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
with open(COCO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]


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


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def detect_fruits(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append(boxes[i] + [classes[class_ids[i]], confidences[i]])

    return detections


def calculate_fruit_position(detection, depth_frame):
    x, y, w, h, _, _ = detection
    center_x = x + w // 2
    center_y = y + h // 2
    depth = depth_frame[center_y, center_x]

    # Convert to polar coordinates
    angle = math.atan2(center_x - depth_frame.shape[1] / 2, depth_frame.shape[0] / 2 - center_y)
    return (depth / 1000.0, angle)  # Convert depth to meters


def calculate_repulsive_force(lidar_data):
    repulsive_force = np.zeros(2)
    for i, distance in enumerate(lidar_data):
        if distance < OBSTACLE_INFLUENCE_DISTANCE:
            angle = i * (2 * np.pi / len(lidar_data))
            force_magnitude = REPULSIVE_FORCE_GAIN * (1 / distance - 1 / OBSTACLE_INFLUENCE_DISTANCE) / (distance ** 2)
            force_vector = force_magnitude * np.array([np.cos(angle), np.sin(angle)])
            repulsive_force += force_vector
    return repulsive_force


def calculate_attractive_force(fruit_position):
    if fruit_position is None:
        return np.zeros(2)
    distance, angle = fruit_position
    force_magnitude = ATTRACTIVE_FORCE_GAIN * distance
    force_vector = force_magnitude * np.array([np.cos(angle), np.sin(angle)])
    return force_vector


def determine_movement(resultant_force):
    force_magnitude = np.linalg.norm(resultant_force)
    if force_magnitude == 0:
        return 0, 0  # No movement

    # Normalize the force vector
    normalized_force = resultant_force / force_magnitude

    # Calculate linear and angular velocities
    linear_velocity = MAX_LINEAR_SPEED * normalized_force[0]
    angular_velocity = MAX_ANGULAR_SPEED * math.atan2(normalized_force[1], normalized_force[0])

    return linear_velocity, angular_velocity


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


def get_user_choice():
    global user_choice
    choice_result = {'choice': None}

    def voice_input():
        nonlocal choice_result
        r = sr.Recognizer()

        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            logging.info("Listening for activation phrase...")

        while running and sleep_mode and not choice_event.is_set():
            try:
                with sr.Microphone() as source:
                    audio = r.listen(source, timeout=5, phrase_time_limit=5)
                recognized_phrase = r.recognize_google(audio, language='pt-BR').lower()
                logging.info(f"You said: {recognized_phrase}")

                if any(phrase in recognized_phrase for phrase in ACTIVATION_PHRASES):
                    text_to_speech("Olá, como você está? Gostaria de escolher banana ou maçã?")

                    with sr.Microphone() as source:
                        audio = r.listen(source, timeout=5, phrase_time_limit=5)
                    choice = r.recognize_google(audio, language='pt-BR').lower()
                    logging.info(f"You chose: {choice}")

                    if "banana" in choice:
                        text_to_speech("Ok, procurando por banana")
                        choice_result['choice'] = "banana"
                        choice_event.set()
                        return
                    elif "maçã" in choice or "maça" in choice:
                        text_to_speech("Ok, procurando por maçã")
                        choice_result['choice'] = "apple"
                        choice_event.set()
                        return
                    else:
                        text_to_speech("Desculpe, não entendi. Por favor, diga banana ou maçã.")

            except sr.UnknownValueError:
                logging.info("Could not understand audio")
            except sr.RequestError as e:
                logging.error(f"Could not request results; {e}")
            except sr.WaitTimeoutError:
                pass

    def keyboard_input():
        nonlocal choice_result

        def on_press(key):
            global running
            if choice_event.is_set():
                return False
            try:
                if key == keyboard.KeyCode.from_char('b'):
                    choice_result['choice'] = "banana"
                    logging.info("Keyboard: chosen banana")
                    text_to_speech("Ok, procurando por banana")
                    choice_event.set()
                    return False
                elif key == keyboard.KeyCode.from_char('m'):
                    choice_result['choice'] = "apple"
                    logging.info("Keyboard: chosen apple")
                    text_to_speech("Ok, procurando por maçã")
                    choice_event.set()
                    return False
                elif key == keyboard.KeyCode.from_char('q'):
                    logging.info("Stopping the program...")
                    choice_event.set()
                    running = False
                    return False
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    voice_thread = Thread(target=voice_input)
    keyboard_thread = Thread(target=keyboard_input)
    voice_thread.start()
    keyboard_thread.start()

    choice_event.wait()
    voice_thread.join(timeout=1)
    keyboard_thread.join(timeout=1)

    return choice_result['choice']


def main():
    global running, sleep_mode, user_choice, fruit_position

    rospy.init_node('fruit_detection_navigation_system', anonymous=True)
    rospy.Subscriber("/scan", LaserScan, lidar_callback)
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    speech_thread = Thread(target=speech_worker, daemon=True)
    speech_thread.start()

    # Initialize OAK-D Lite
    pipeline = dai.Pipeline()
    camRgb = pipeline.createColorCamera()
    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    # Setup depth
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.depth.link(xoutDepth.input)

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        speech_queue.put("Sistema pronto.")
        logging.info("System ready.")

        while not rospy.is_shutdown() and running:
            if sleep_mode:
                choice_event.clear()
                user_choice = get_user_choice()
                if user_choice:
                    DETECTION_CLASSES = [user_choice]
                    logging.info(f"Detecting fruit: {user_choice}")
                    sleep_mode = False
                    speech_queue.put(f"Procurando por {user_choice}")
                else:
                    continue

            inRgb = qRgb.tryGet()
            inDepth = qDepth.tryGet()

            if inRgb is not None and inDepth is not None:
                frame = inRgb.getCvFrame()
                depth_frame = inDepth.getFrame()

                detections = detect_fruits(frame)

                if detections:
                    fruit_position = calculate_fruit_position(detections[0], depth_frame)
                    distance, angle = fruit_position
                    speech_queue.put(
                        f"{user_choice} detectado a {distance:.2f} metros, ângulo {math.degrees(angle):.1f} graus")
                else:
                    fruit_position = None

                # Calculate forces
                repulsive_force = calculate_repulsive_force(lidar_data)
                attractive_force = calculate_attractive_force(fruit_position)
                resultant_force = attractive_force + repulsive_force

                # Determine movement
                linear_velocity, angular_velocity = determine_movement(resultant_force)

                # Publish movement command
                twist = Twist()
                twist.linear.x = linear_velocity
                twist.angular.z = angular_velocity
                cmd_vel_pub.publish(twist)

                # Visualization (optional)
                for detection in detections:
                    x, y, w, h, class_name, confidence = detection
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Fruit Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        pygame.init()
        pygame.mixer.init()
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")
    finally:
        shutdown()


def shutdown():
    global running, sleep_mode
    logging.info("Shutting down...")
    running = False
    sleep_mode = True

    # Stop the robot
    try:
        cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        stop_twist = Twist()
        cmd_vel_pub.publish(stop_twist)
        rospy.sleep(1)  # Give time for the stop command to be processed
    except Exception as e:
        logging.error(f"Error stopping the robot: {e}")

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Cleanup pygame
    pygame.quit()

    # Shutdown ROS node
    try:
        rospy.signal_shutdown("User requested shutdown")
    except Exception as e:
        logging.error(f"Error shutting down ROS node: {e}")

    logging.info("Shutdown complete.")


# Additional helper functions

def visualize_forces(frame, attractive_force, repulsive_force, resultant_force):
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    scale = 100  # Scale factor for force visualization

    # Draw attractive force (green)
    cv2.arrowedLine(frame, center,
                    (int(center[0] + attractive_force[0] * scale),
                     int(center[1] - attractive_force[1] * scale)),
                    (0, 255, 0), 2)

    # Draw repulsive force (red)
    cv2.arrowedLine(frame, center,
                    (int(center[0] + repulsive_force[0] * scale),
                     int(center[1] - repulsive_force[1] * scale)),
                    (0, 0, 255), 2)

    # Draw resultant force (blue)
    cv2.arrowedLine(frame, center,
                    (int(center[0] + resultant_force[0] * scale),
                     int(center[1] - resultant_force[1] * scale)),
                    (255, 0, 0), 2)

    return frame


def visualize_lidar(lidar_data, window_name="LIDAR Visualization"):
    if not lidar_data:
        return None

    lidar_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    center = (200, 200)
    max_distance = 5.0  # 5 meters max range

    num_points = len(lidar_data)
    for i, distance in enumerate(lidar_data):
        if distance > 0 and distance < max_distance:
            angle = i * 2 * np.pi / num_points
            x = int(center[0] + np.cos(angle) * distance * 40)
            y = int(center[1] + np.sin(angle) * distance * 40)
            cv2.circle(lidar_frame, (x, y), 2, (0, 255, 0), -1)

    cv2.circle(lidar_frame, center, 3, (0, 0, 255), -1)
    cv2.line(lidar_frame, center, (center[0], center[1] - 100), (255, 0, 0), 1)  # Forward direction

    return lidar_frame


# Modify the main function to include visualization
def main():
    global running, sleep_mode, user_choice, fruit_position

    rospy.init_node('fruit_detection_navigation_system', anonymous=True)
    rospy.Subscriber("/scan", LaserScan, lidar_callback)
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    speech_thread = Thread(target=speech_worker, daemon=True)
    speech_thread.start()

    # Initialize OAK-D Lite
    pipeline = dai.Pipeline()
    camRgb = pipeline.createColorCamera()
    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    # Setup depth
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.depth.link(xoutDepth.input)

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        speech_queue.put("Sistema pronto.")
        logging.info("System ready.")

        while not rospy.is_shutdown() and running:
            if sleep_mode:
                choice_event.clear()
                user_choice = get_user_choice()
                if user_choice:
                    DETECTION_CLASSES = [user_choice]
                    logging.info(f"Detecting fruit: {user_choice}")
                    sleep_mode = False
                    speech_queue.put(f"Procurando por {user_choice}")
                else:
                    continue

            inRgb = qRgb.tryGet()
            inDepth = qDepth.tryGet()

            if inRgb is not None and inDepth is not None:
                frame = inRgb.getCvFrame()
                depth_frame = inDepth.getFrame()

                detections = detect_fruits(frame)

                if detections:
                    fruit_position = calculate_fruit_position(detections[0], depth_frame)
                    distance, angle = fruit_position
                    speech_queue.put(
                        f"{user_choice} detectado a {distance:.2f} metros, ângulo {math.degrees(angle):.1f} graus")
                else:
                    fruit_position = None

                # Calculate forces
                repulsive_force = calculate_repulsive_force(lidar_data)
                attractive_force = calculate_attractive_force(fruit_position)
                resultant_force = attractive_force + repulsive_force

                # Determine movement
                linear_velocity, angular_velocity = determine_movement(resultant_force)

                # Publish movement command
                twist = Twist()
                twist.linear.x = linear_velocity
                twist.angular.z = angular_velocity
                cmd_vel_pub.publish(twist)

                # Visualization
                for detection in detections:
                    x, y, w, h, class_name, confidence = detection
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Visualize forces
                frame = visualize_forces(frame, attractive_force, repulsive_force, resultant_force)

                cv2.imshow("Fruit Detection and Forces", frame)

                # Visualize LIDAR data
                lidar_visualization = visualize_lidar(lidar_data)
                if lidar_visualization is not None:
                    cv2.imshow("LIDAR Visualization", lidar_visualization)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break

    cv2.destroyAllWindows()
