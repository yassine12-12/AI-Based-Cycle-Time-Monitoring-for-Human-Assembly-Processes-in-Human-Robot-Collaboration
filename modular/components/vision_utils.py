"""
Vision utilities for YOLO object detection and MediaPipe hand tracking.
Contains helper functions for distance calculations and object detection.

MODULAR VERSION - Organized in modular/components/
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


def calculate_hand_to_object_distance(hand_landmarks, yolo_results, target_class: str, frame_width: int, frame_height: int) -> Tuple[float, bool]:
    """
    Calculate the minimum distance between any hand landmark and any detected object of the target class.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        yolo_results: YOLO detection results
        target_class: The YOLO class name to find (e.g., 'motor_1', 'pcb_2', 'battery')
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        
    Returns:
        tuple: (min_distance_pixels, object_found)
            min_distance_pixels: Minimum distance in pixels, or float('inf') if no object found
            'object_found': bool,            # Whether the target class was found
    """
    if not hand_landmarks:
        return float('inf'), False
    
    # Find all objects of the target class
    target_objects = []
    for result in yolo_results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                if class_name == target_class:
                    # Convert normalized coordinates to pixel coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    target_objects.append((x1, y1, x2, y2))
    
    if not target_objects:
        return float('inf'), False
    
    # Calculate minimum distance from any hand landmark to any target object
    min_distance = float('inf')
    
    for hand in hand_landmarks:
        for landmark in hand.landmark:
            # Convert normalized hand coordinates to pixel coordinates
            hand_x = landmark.x * frame_width
            hand_y = landmark.y * frame_height
            
            # Check distance to all target objects
            for x1, y1, x2, y2 in target_objects:
                # Calculate distance to the closest point on the bounding box
                closest_x = max(x1, min(hand_x, x2))
                closest_y = max(y1, min(hand_y, y2))
                
                distance = math.sqrt((hand_x - closest_x)**2 + (hand_y - closest_y)**2)
                min_distance = min(min_distance, distance)
    
    return min_distance, True


def is_hand_near_object(hand_landmarks, yolo_results, target_class: str, frame_width: int, frame_height: int, threshold: float = 100.0) -> bool:
    """
    Check if any hand is near any object of the specified class.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        yolo_results: YOLO detection results
        target_class: The YOLO class name to check proximity to
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        threshold: Distance threshold in pixels
        
    Returns:
        bool: True if any hand is within threshold distance of the target class
    """
    distance, object_found = calculate_hand_to_object_distance(
        hand_landmarks, yolo_results, target_class, frame_width, frame_height
    )
    return object_found and distance <= threshold




def detect_yolo_objects(yolo_results) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extract detected YOLO classes and their details from YOLO results.
    
    Args:
        yolo_results: YOLO detection results
        
    Returns:
        tuple: (classes_in_frame, detection_details)
            classes_in_frame: List of detected class names
            detection_details: Dict with detailed detection information
    """
    classes_in_frame = []
    detection_details = {}
    
    for result in yolo_results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                classes_in_frame.append(class_name)
                
                if class_name not in detection_details:
                    detection_details[class_name] = []
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detection_details[class_name].append({
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'class_id': class_id
                })
    
    return classes_in_frame, detection_details


def draw_detection_overlay(frame: np.ndarray, yolo_results, hand_landmarks, mp_drawing, mp_hands) -> np.ndarray:
    """
    Draw YOLO bounding boxes and MediaPipe hand landmarks on the frame.
    
    Args:
        frame: The video frame to draw on
        yolo_results: YOLO detection results
        hand_landmarks: MediaPipe hand landmarks
        mp_drawing: MediaPipe drawing utilities
        mp_hands: MediaPipe hands solution
        
    Returns:
        np.ndarray: Frame with overlays drawn
    """
    # Draw YOLO bounding boxes
    for result in yolo_results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw hand landmarks
    if hand_landmarks:
        for hand in hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    
    return frame


def apply_roi(frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Apply Region of Interest (ROI) to a frame if specified.
    
    Args:
        frame: The input video frame
        roi: Region of interest as (x, y, width, height), or None for full frame
        
    Returns:
        np.ndarray: Cropped frame if ROI is specified, otherwise original frame
    """
    if roi is None:
        return frame
    
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def get_closest_hand_center(hand_landmarks, frame_width: int, frame_height: int) -> Optional[Tuple[int, int]]:
    """
    Get the center position of the closest hand to the top-left corner.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        
    Returns:
        tuple: (x, y) coordinates of hand center, or None if no hands detected
    """
    if not hand_landmarks:
        return None
    
    # Use the first detected hand's center (index finger tip or palm center)
    hand = hand_landmarks[0]
    
    # Use the palm center (landmark 0) or index finger tip (landmark 8)
    if len(hand.landmark) > 8:
        landmark = hand.landmark[8]  # Index finger tip
    else:
        landmark = hand.landmark[0]  # Wrist/palm center
    
    center_x = int(landmark.x * frame_width)
    center_y = int(landmark.y * frame_height)
    
    return (center_x, center_y)


def get_object_center(yolo_results, target_class: str) -> Optional[Tuple[int, int]]:
    """
    Get the center position of the first detected object of the target class.
    
    Args:
        yolo_results: YOLO detection results
        target_class: The YOLO class name to find
        
    Returns:
        tuple: (x, y) coordinates of object center, or None if no object found
    """
    for result in yolo_results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                if class_name == target_class:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Calculate center
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    return (center_x, center_y)
    
    return None

import cv2.aruco as aruco

def detect_aruco_ids(frame, dict_types=None):
    """
    Detect ArUco markers from multiple dictionaries.

    Args:
        frame: BGR image (numpy array)
        dict_types: list of aruco dictionary constants to try

    Returns:
        detected_ids: list of detected marker IDs (across all dicts)
        centers: dict of {id: (center_x, center_y)}
        corner_map: dict of {id: corners}
    """
    if dict_types is None:
        dict_types = [
            aruco.DICT_4X4_50,
            aruco.DICT_4X4_100,
            aruco.DICT_4X4_250,
            aruco.DICT_4X4_1000,
            aruco.DICT_5X5_50,
            aruco.DICT_5X5_100,
            aruco.DICT_5X5_250,
            aruco.DICT_5X5_1000
        ]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_ids = []
    centers = {}
    corner_map = {}

    for dict_type in dict_types:
        aruco_dict = aruco.getPredefinedDictionary(dict_type)
        parameters = aruco.DetectorParameters()
        # Optional tweaks for harder markers
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.minMarkerPerimeterRate = 0.01
        
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for i, corner in enumerate(corners):
                aruco_id = int(ids[i][0])
                if aruco_id not in detected_ids:  # avoid duplicates
                    pts = corner[0]
                    center_x = int(pts[:, 0].mean())
                    center_y = int(pts[:, 1].mean())
                    detected_ids.append(aruco_id)
                    centers[aruco_id] = (center_x, center_y)
                    corner_map[aruco_id] = corner
                    
    return detected_ids, centers, corner_map