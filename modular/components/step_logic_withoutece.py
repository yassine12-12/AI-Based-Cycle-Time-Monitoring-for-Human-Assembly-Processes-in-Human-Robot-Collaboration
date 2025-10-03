
"""
Step logic module for assembly step tracking.
Contains the main step management classes and step completion logic.

MODULAR VERSION - Organized in modular/components/
"""

import time
from typing import List, Optional, Dict, Any

# Import from modular components
from .config import AssemblyStep
from .vision_utils import is_hand_near_object, detect_yolo_objects
from .timer_utils import StepTimer
from .vibration_utils import DetectionSmoother
import cv2.aruco as aruco
from .vision_utils import detect_aruco_ids
# from .kiste_utils import kiste_fall


def get_step_configuration(step_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the configuration for a specific step based on its name.
    This function centralizes ALL step logic configuration in one place.
    
    Args:
        step_name: The name of the assembly step
        
    Returns:
        dict: Configuration for the step, or None if no specific config found
    """
    # step_lower = step_name.lower()
    
    # === STEP DISTANCE CONFIGURATION ===
    # All step completion logic configurations in one place
    step_configs = {
        # Human fixes Motor  
        'Human mounts Motor 1': {
            'completion_type': 'object_timeout_handappear_sameroi',
            'wait_duration': 3.0,
            'roi_1': {"x1": 671, "y1": 518, "x2": 945, "y2": 685},
            'required_objects': ['Motor']
            # 'completion_type': 'distance_timeout',
            # 'wait_duration': 2.0,
            # 'requires_hands_away': True,
            # 'distance_threshold': 90,
            # 'required_objects': ['Motor']
        },
        'Human mounts Motor 2': {
            'completion_type': 'object_timeout_handappear_sameroi',
            'wait_duration': 3.0,
            'roi_1': {"x1": 638, "y1": 280, "x2": 931, "y2": 458},
            'required_objects': ['Motor']
            # 'completion_type': 'distance_timeout',
            # 'wait_duration': 2.0,
            # 'requires_hands_away': True,
            # 'distance_threshold': 90,
            # 'required_objects': ['Motor']
        },
        'Human mounts Motor 3': {
            'completion_type': 'object_timeout_handappear_sameroi',
            'wait_duration': 3.0,
            'roi_1': {"x1": 291, "y1": 515, "x2": 577, "y2": 712},
            'required_objects': ['Motor']
            # 'completion_type': 'distance_timeout',
            # 'wait_duration': 2.0,
            # 'requires_hands_away': True,
            # 'distance_threshold': 90,
            # 'required_objects': ['Motor']
        },
        'Human mounts Motor 4': {
            'completion_type': 'object_timeout_handappear_sameroi',
            'wait_duration': 3.0,
            'roi_1': {"x1": 331, "y1": 318, "x2": 606, "y2": 468},
            'required_objects': ['Motor']
            # 'completion_type': 'distance_timeout',
            # 'wait_duration': 2.0,
            # 'requires_hands_away': True,
            # 'distance_threshold': 90,
            # 'required_objects': ['Motor']
        },
        'Human mounts LModul': { # PCB1
            'completion_type': 'distance_timeout',
            'distance_threshold': 90,
            'wait_duration': 3.0,
            'requires_hands_away': True,
            'required_objects': ['LModul']
        },
        'Human mounts Uno': { # PCB2
            'completion_type': 'distance_timeout',
            'distance_threshold': 90,
            'wait_duration': 3.0,
            'requires_hands_away': True,
            'required_objects': ['Uno']
        },
        'Human mounts Erweiterungsboard': { # PCB3
            'completion_type': 'distance_timeout_withoutkiste',
            'distance_threshold': 90,
            'wait_duration': 3.0,
            'requires_hands_away': True,
            'required_objects': ['Erweiterungsboard']
        },
        'Human mounts Akku': {
            'completion_type': 'distance_timeout',
            'distance_threshold': 90,
            'wait_duration': 1.5,
            'requires_hands_away': True,
            'required_objects': ['Akku']
        },
        'Human mounts Kupfersaeule': {
            'completion_type': 'object_timeout_handappear_diffroi',
            'wait_duration': 3.0,
            'roi_2': [
            {"x1": 788, "y1": 605, "x2": 1016, "y2": 742},  # ROI3
            {"x1": 764, "y1": 237, "x2": 969, "y2": 352},  # ROI4
            ],
            'required_objects': ['Kupfersaeule']
        },
        'Human mounts Sensorbaugruppe': {
            'completion_type': 'distance_timeout',
            'distance_threshold': 90,
            'wait_duration': 3.0,
            'requires_hands_away': True,
            'required_objects': ['Sensorbaugruppe']
        },
        'Human mounts teil b_Motor': {
            'completion_type': 'aruco_mit_kiste',
            'distance_threshold': 80,
            'wait_duration': 3.0,
            'left_plate_id': 584,
            'required_objects': None
        },
        'Human mounts rad': {
            'completion_type': 'object_timeout_handappear_withoutkiste',
            'wait_duration': 3.0,
            'roi_2': [
            {"x1": 591, "y1": 585, "x2": 913, "y2": 810},  # ROI3
            {"x1": 598, "y1": 151, "x2": 923, "y2": 378},  # ROI4
            ],
            'required_objects': ['rad']
        }
    }
    
    # Directly match step name to config key
    config = step_configs.get(step_name)
    if config:
        return config
    return None


class StepManager:
    """Manages the current assembly step and progression through the sequence."""
    
    def __init__(self, steps: List[AssemblyStep]):
        self.steps = steps
        self.current_step_index = 0
        self.completed_steps = []
        self.begin = False
        self.end = False
    
    def get_current_step(self) -> Optional[AssemblyStep]:
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def advance_to_next_step(self):
        if self.current_step_index < len(self.steps):
            self.completed_steps.append(self.steps[self.current_step_index])
            self.current_step_index += 1
    
    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps)
    
    def get_progress(self) -> tuple:
        return self.current_step_index, len(self.steps)


class StepClock:
    """Tracks timing information for assembly steps."""
    
    def __init__(self):
        self.step_start_times = {}
        self.step_durations = {}
        self.current_step_start = None
    
    def start_step(self, step_name: str):
        self.current_step_start = time.time()
        self.step_start_times[step_name] = self.current_step_start
    
    def end_step(self, step_name: str):
        if step_name in self.step_start_times and self.current_step_start:
            duration = time.time() - self.current_step_start
            self.step_durations[step_name] = duration
            self.current_step_start = None
    
    def get_current_step_duration(self) -> float:
        if self.current_step_start:
            return time.time() - self.current_step_start
        return 0.0
    
    def get_total_duration(self) -> float:
        return sum(self.step_durations.values())

def is_hand_in_roi(hand_landmarks, roi, frame_width, frame_height):
    if not hand_landmarks:
        return False
    
    x1r, y1r, x2r, y2r = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    if x1r > x2r: x1r, x2r = x2r, x1r
    if y1r > y2r: y1r, y2r = y2r, y1r

    # hand_landmarks: List[NormalizedLandmarkList]
    for hand in hand_landmarks:
        for lm in hand.landmark:
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            if x1r <= x <= x2r and y1r <= y <= y2r:
                return True
    return False

def is_object_in_roi(yolo_results, roi, allowed_classes):
    classes_in_frame, boxes_by_class = detect_yolo_objects(yolo_results)
    allowed = {str(c).lower() for c in allowed_classes}

    x1r, y1r, x2r, y2r = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    if x1r > x2r: x1r, x2r = x2r, x1r
    if y1r > y2r: y1r, y2r = y2r, y1r

    for cls_name in classes_in_frame:
        if cls_name.lower() not in allowed:
            continue
        for det in boxes_by_class.get(cls_name, []):
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = map(float, bbox[:4])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if x1r <= cx <= x2r and y1r <= cy <= y2r:
                return True
    return False

def generic_step_logic(step_name: str, hand_landmarks, yolo_results, 
                      frame_width: int, frame_height: int, 
                      step_timer: StepTimer, step_manager: StepManager,
                      is_kiste_empty: bool,
                      current_frame=None) -> bool:
    """
    Generic step completion logic based on step configuration.
    
    Args:
        step_name: Name of the current assembly step
        hand_landmarks: MediaPipe hand landmarks
        yolo_results: YOLO detection results
        frame_width: Width of video frame
        frame_height: Height of video frame
        step_timer: Timer instance for this step
        current_frame: Current video frame for kiste_fall analysis ## Kistefunktion:
    
    Returns:
        bool: True if step is completed, False otherwise
    """
    config = get_step_configuration(step_name)
    if not config:
        # Default behavior: just check for object detection
        classes_in_frame, _ = detect_yolo_objects(yolo_results)
        # Extract expected class from step name (simplified)
        step_lower = step_name.lower()
        for class_name in classes_in_frame:
            if any(word in class_name.lower() for word in step_lower.split()):
                return True
        return False
    
    completion_type = config.get('completion_type', 'object_detection')
    distance_threshold = config.get('distance_threshold', 100)
    wait_duration = config.get('wait_duration', 2.0)
    requires_hands_away = config.get('requires_hands_away', False)
    required_objects = config.get('required_objects', [])
    
    # --- Neue Kisten-Status-Logik ---
    # kiste_voll_required = config.get('kiste_voll', None)  ## Kistefunktion:
    # if kiste_voll_required is not None and current_frame is not None:  ## Kistefunktion:
    #     kiste_status, _ = kiste_fall(current_frame)  # True = voll, False = leer ## Kistefunktion:
    #     if kiste_status != kiste_voll_required:  # Zustand passt nicht => Schritt nicht fertig ## Kistefunktion:
    #         return False  # Step not complete if Kistenstatus nicht passt ## Kistefunktion:
    # --------------------------------

    # Get detected objects
    classes_in_frame, _ = detect_yolo_objects(yolo_results)
    
    # Check if all required objects are present
    if required_objects:
        for required_obj in required_objects:
            if required_obj not in classes_in_frame:
                return False  # Required object missing
    
    # Find target object class (simplified matching)
    target_class = None
    step_lower = step_name.lower()
    for class_name in classes_in_frame:
        class_lower = class_name.lower()
        if any(word in class_lower for word in step_lower.split() if len(word) > 2):
            target_class = class_name
            break
    
    if not target_class and not required_objects:
        return False

    # Use first required object as target if no specific target found
    if not target_class and required_objects:
        target_class = required_objects[0]
    
    if completion_type == 'distance_timeout':
        # Check if hands are near the object
        hands_near = is_hand_near_object(
            hand_landmarks, yolo_results, target_class,
            frame_width, frame_height, distance_threshold
        )
        if hands_near and not step_manager.begin and not is_kiste_empty: ###NUTZT DAS UM DAS ZU BEGINNEN
            step_manager.begin = True ###NUTZT DAS UM DAS ZU BEGINNEN
        
        if requires_hands_away and is_kiste_empty:
            if step_manager.begin and hands_near and not step_manager.end: ###NUTZT DAS UM DAS ZU ENDEN
                step_manager.end = True  ###NUTZT DAS UM DAS ZU ENDEN

            # For steps that require hands to move away after interaction
            if not hands_near and step_manager.end:    ####
                if not step_timer.is_active:
                    print("111111111111")
                    step_timer.start()
                elif step_timer.has_elapsed(wait_duration):
                    print("222222222222")
                    return True
                
            else:
                print("xxxxxxxxxxx")
                step_timer.stop()  # Reset timer if hands come back
        print("END:   ", step_manager.end, "\nKiste: ", is_kiste_empty)
        return False
    
    elif completion_type == 'object_timeout_handappear_sameroi':
        roi_1 = config.get("roi_1")
        if not roi_1:
            return False
        # ROI1 = {"x1": 812, "y1": 290, "x2": 812 + 49, "y2": 290 + 32}
    
        # check if hand in roi_1
        hand_in_roi1 = is_hand_in_roi(hand_landmarks, roi_1, frame_width, frame_height)
        object_in_roi1 = is_object_in_roi(
        yolo_results, roi_1,
        allowed_classes={"Motor"}
         )
        print("object_in_roi1: ", object_in_roi1)
        
        # Begin to measure duration
        if not step_manager.begin and not is_kiste_empty and object_in_roi1 and hand_in_roi1:
            step_manager.begin = True

        if is_kiste_empty:
            if step_manager.begin and hand_in_roi1 and not step_manager.end:
                step_manager.end = True 

            # Need hand and Kupfersaeule in ROIs
            if step_manager.end and not hand_in_roi1 and object_in_roi1:
                if not step_timer.is_active:
                    step_timer.start()
                elif step_timer.has_elapsed(wait_duration):
                    return True
            else:
                step_timer.stop()
        return False

    elif completion_type == 'distance_timeout_withoutkiste':
        # Check if hands are near the object
        hands_near = is_hand_near_object(
            hand_landmarks, yolo_results, target_class,
            frame_width, frame_height, distance_threshold
        )
        if requires_hands_away:
            if hands_near and not step_manager.begin and not step_manager.end:
                step_manager.begin = True
                step_manager.end = True

            # For steps that require hands to move away after interaction
            if not hands_near and step_manager.end:
                if not step_timer.is_active:
                    step_timer.start()
                elif step_timer.has_elapsed(wait_duration):
                    return True
            else:
                step_timer.stop()  # Reset timer if hands come back
        
        return False
    
    elif completion_type == 'object_timeout_handappear_diffroi':
        roi_2 = config.get("roi_2", [])
        if len(roi_2) < 2:
            return False
        #    ROI1 = {"x1": 812, "y1": 290, "x2": 812 + 49, "y2": 290 + 32}
        #    ROI2 = {"x1": 807, "y1": 426, "x2": 807 + 56, "y2": 426 + 32}

        start_roi, end_roi = roi_2[0], roi_2[1]

        # check if hand in start_roi and end_roi
        hand_in_start_roi = is_hand_in_roi(hand_landmarks, start_roi, frame_width, frame_height)
        hand_in_end_roi = is_hand_in_roi(hand_landmarks, end_roi, frame_width, frame_height)
        object_in_end_roi = is_object_in_roi(
        yolo_results, end_roi,
        allowed_classes={"Kupfersaeule"}
         )
        
        # Begin to measure duration
        if not step_manager.begin and hand_in_start_roi and not is_kiste_empty:
            step_manager.begin = True
        
        if is_kiste_empty:
            if step_manager.begin and hand_in_end_roi and not step_manager.end:
                step_manager.end = True

            # Need hand and object in end_roi
            if step_manager.end and not hand_in_end_roi and object_in_end_roi:
                if not step_timer.is_active:
                    step_timer.start()
                elif step_timer.has_elapsed(wait_duration):
                    return True
            else:
                step_timer.stop()
        return False

    elif completion_type == 'aruco_mit_kiste':
        detected_ids, _, _ = detect_aruco_ids(current_frame,aruco_dict_type=aruco.DICT_4X4_1000)
        left_id = config.get('left_plate_id', 584)

        if left_id not in detected_ids:
            if missing_start_time is None:
                missing_start_time = time.time()  # Start missing timer
            elif (time.time() - missing_start_time) > 3:  # Missing > 3 seconds
                if not step_manager.begin and not is_kiste_empty:
                    step_manager.begin = True
        else:
            missing_start_time = None  # Reset if ID is detected again

        if requires_hands_away and is_kiste_empty:
            if step_manager.begin and hands_near and not step_manager.end:
                step_manager.end = True 

            # For steps that require hands to move away after interaction
            if not hands_near and step_manager.end:    ####
                if not step_timer.is_active:
                    step_timer.start()
                elif step_timer.has_elapsed(wait_duration):
                    return True
            else:
                step_timer.stop()  # Reset timer if hands come back
        return False
    
    elif completion_type == 'object_timeout_handappear_withoutkiste':
        roi_2 = config.get("roi_2", [])
        if len(roi_2) < 2:
            return False
        #    ROI3 = {"x1": 812, "y1": 290, "x2": 812 + 49, "y2": 290 + 32}
        #    ROI4 = {"x1": 807, "y1": 426, "x2": 807 + 56, "y2": 426 + 32}

        start_roi, end_roi = roi_2[0], roi_2[1]
    
    # check if hand in start_roi and end_roi
        hand_in_start_roi = is_hand_in_roi(hand_landmarks, start_roi, frame_width, frame_height)
        hand_in_end_roi = is_hand_in_roi(hand_landmarks, end_roi, frame_width, frame_height)
        object_in_start_roi = is_object_in_roi(
        yolo_results, start_roi,
        allowed_classes={"rad"}
         )
        object_in_end_roi = is_object_in_roi(
        yolo_results, end_roi,
        allowed_classes={"rad"}
         )
        
        # Begin to measure duration
        if not step_manager.begin and hand_in_start_roi and object_in_start_roi:
            step_manager.begin = True

        if step_manager.begin and hand_in_end_roi and object_in_end_roi and not step_manager.end:
            step_manager.end = True 

        # Need hand and object in end_roi
        if step_manager.end and not hand_in_end_roi and object_in_end_roi:
            if not step_timer.is_active:
                step_timer.start()
            elif step_timer.has_elapsed(wait_duration):
                return True
        else:
            step_timer.stop()
        return False
    
    else: # completion type not found
        return False

def human_step_logic(current_step: AssemblyStep, hand_landmarks, yolo_results, 
                    frame_width: int, frame_height: int, step_timer: StepTimer, step_manager: StepManager, is_kiste_empty: bool,
                    detection_smoother: DetectionSmoother=None, current_frame=None) -> bool:
    """
    Main step logic function that determines when a Human assembly step is completed.
    
    This function prioritizes the configuration-driven generic logic over hardcoded logic.
    
    Args:
        current_step: The current assembly step being tracked
        hand_landmarks: MediaPipe hand landmarks
        yolo_results: YOLO detection results  
        frame_width: Width of video frame
        frame_height: Height of video frame
        step_timer: Timer instance for time-based conditions
        detection_smoother: Optional DetectionSmoother instance to stabilize detections ## Vibrationsfunktion
        current_frame: Current video frame for kiste_fall analysis ## Kistefunktion

    Returns:
        bool: True if the step is completed, False otherwise
    """
    
    # step_name = current_step.name.lower()
    
    # PRIORITY 1: Use configuration-driven generic logic
    config = get_step_configuration(current_step.name)
    if config:
        print(f"Using config-driven logic for: {current_step.name}")

        # Nutze detection_smoother um class presence abzufragen ## Vibrationsfunktion
        # classes_in_frame, _ = detect_yolo_objects(yolo_results)
        # if detection_smoother:
        #     for cls in classes_in_frame:
        #         detection_smoother.update(cls)  #classes_in_frame
        #     # Prüfe, ob die Zielklasse stabil erkannt wird
        #     target_class = None
        #     step_lower = current_step.name.lower()
        #     for class_name in classes_in_frame:
        #         class_lower = class_name.lower()
        #         if any(word in class_lower for word in step_lower.split() if len(word) > 2):
        #             target_class = class_name
        #             break
        #     if target_class and not detection_smoother.is_present(target_class):
        #         # Zielklasse nicht stabil erkannt -> Schritt noch nicht abgeschlossen
        #         print("---------------------")
        #         return False
            
        return generic_step_logic(current_step.name, hand_landmarks, yolo_results, 
                                frame_width, frame_height, step_timer, step_manager, is_kiste_empty,
                                current_frame=current_frame)
    
    # PRIORITY 2: Fallback to specific hardcoded logic for steps without config
    print(f"Using fallback hardcoded logic for: {current_step.name}")
    
    classes_in_frame, _ = detect_yolo_objects(yolo_results)

    if detection_smoother:  ## Vibrationsfunktion
        detection_smoother.update(classes_in_frame)  ## Vibrationsfunktion
        if current_step.yolo_class and not detection_smoother.is_present(current_step.yolo_class):  ## Vibrationsfunktion
            input("---------------------")
            return False  ## Vibrationsfunktion

    # object_seen = current_step.yolo_class in classes_in_frame
    
    # # === ENUMERATED STEP LOGIC (Fallback only) ===
    # # These are only used when no configuration is found
    
    # # Default case for unconfigured steps
    # if object_seen:
    #     return True
    
    return False
