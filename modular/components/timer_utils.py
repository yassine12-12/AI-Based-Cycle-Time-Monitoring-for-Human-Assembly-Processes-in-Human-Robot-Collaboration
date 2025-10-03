"""
Timer utilities for assembly step tracking.
Contains reusable timer classes and functions for time-based conditions.

MODULAR VERSION - Organized in modular/components/
"""

import time
from typing import Callable, Optional, Any


class StepTimer:
    """
    A reusable timer class for managing time-based conditions in assembly steps.
    
    Useful for scenarios like:
    - Waiting for hands to be away from an object for a certain duration
    - Requiring a gesture to be held for a specific time
    - Checking if an object has been stable/detected for a minimum duration
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.is_active: bool = False
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
        self.is_active = True
    
    def stop(self) -> None:
        """Stop the timer."""
        self.is_active = False
        self.start_time = None
    
    def restart(self) -> None:
        """Restart the timer."""
        self.start()
    
    def elapsed_time(self) -> float:
        """Get the elapsed time since the timer started."""
        if not self.is_active or self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def has_elapsed(self, duration: float) -> bool:
        """Check if the specified duration has elapsed."""
        return self.is_active and self.elapsed_time() >= duration
    
    def reset(self) -> None:
        """Reset the timer to initial state."""
        self.stop()


def wait_for_duration(condition_func: Callable[[], bool], duration: float, 
                     check_interval: float = 0.1) -> bool:
    """
    Wait for a condition to remain true for a specified duration.
    
    Args:
        condition_func: Function that returns True/False for the condition to check
        duration: How long the condition must remain true (in seconds)
        check_interval: How often to check the condition (in seconds)
        
    Returns:
        bool: True if condition remained true for the full duration, False otherwise
        
    Example:
        # Wait for hands to be away from motor for 2 seconds
        hands_away = lambda: not is_hand_near_object(hands, yolo_results, "motor_1", w, h, 100)
        if wait_for_duration(hands_away, 2.0):
            print("Hands have been away from motor for 2 seconds!")
    """
    timer = StepTimer()
    
    while True:
        if condition_func():
            if not timer.is_active:
                timer.start()
            elif timer.has_elapsed(duration):
                return True
        else:
            timer.stop()
        
        time.sleep(check_interval)


def wait_for_condition_change(condition_func: Callable[[], bool], 
                            target_state: bool, 
                            timeout: Optional[float] = None) -> bool:
    """
    Wait for a condition to change to a target state.
    
    Args:
        condition_func: Function that returns True/False for the condition to check
        target_state: The desired state (True/False) to wait for
        timeout: Maximum time to wait (in seconds), or None for no timeout
        
    Returns:
        bool: True if condition reached target state, False if timeout occurred
        
    Example:
        # Wait for motor to be detected
        motor_detected = lambda: "motor_1" in detected_classes
        if wait_for_condition_change(motor_detected, True, timeout=10.0):
            print("Motor detected!")
    """
    start_time = time.time()
    
    while True:
        if condition_func() == target_state:
            return True
        
        if timeout is not None and (time.time() - start_time) >= timeout:
            return False
        
        time.sleep(0.1)


# Example usage functions for common assembly scenarios

def wait_for_hands_away(hand_landmarks, yolo_results, target_class: str, 
                       frame_width: int, frame_height: int, 
                       duration: float = 2.0, threshold: float = 100.0) -> bool:
    """
    Wait for hands to be away from a specific object for a given duration.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        yolo_results: YOLO detection results
        target_class: YOLO class to check distance from
        frame_width: Width of video frame
        frame_height: Height of video frame
        duration: How long hands must be away (seconds)
        threshold: Distance threshold in pixels
        
    Returns:
        bool: True if hands were away for the full duration
    """
    from .vision_utils import is_hand_near_object
    
    def hands_away():
        return not is_hand_near_object(
            hand_landmarks, yolo_results, target_class, 
            frame_width, frame_height, threshold
        )
    
    return wait_for_duration(hands_away, duration)


def wait_for_gesture_duration(hand_landmarks, gesture_check_func: Callable[[Any], bool], 
                             duration: float = 1.0) -> bool:
    """
    Wait for a specific hand gesture to be held for a given duration.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        gesture_check_func: Function that takes hand_landmarks and returns True if gesture is detected
        duration: How long gesture must be held (seconds)
        
    Returns:
        bool: True if gesture was held for the full duration
    """
    def gesture_active():
        return gesture_check_func(hand_landmarks)
    
    return wait_for_duration(gesture_active, duration)


def wait_for_object_stability(yolo_results, target_class: str, duration: float = 1.0) -> bool:
    """
    Wait for an object to be consistently detected for a given duration.
    
    Args:
        yolo_results: YOLO detection results
        target_class: YOLO class to check for
        duration: How long object must be detected (seconds)
        
    Returns:
        bool: True if object was detected consistently for the full duration
    """
    from .vision_utils import detect_yolo_objects
    
    def object_detected():
        classes_in_frame, _ = detect_yolo_objects(yolo_results)
        return target_class in classes_in_frame
    
    return wait_for_duration(object_detected, duration)


def wait_for_proximity_change(hand_landmarks, yolo_results, target_class: str,
                             frame_width: int, frame_height: int,
                             target_near: bool, threshold: float = 100.0,
                             timeout: Optional[float] = None) -> bool:
    """
    Wait for hands to move near to or away from an object.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        yolo_results: YOLO detection results
        target_class: YOLO class to check proximity to
        frame_width: Width of video frame
        frame_height: Height of video frame
        target_near: True to wait for hands to get near, False to wait for hands to move away
        threshold: Distance threshold in pixels
        timeout: Maximum time to wait (seconds)
        
    Returns:
        bool: True if proximity changed as expected, False if timeout
    """
    from .vision_utils import is_hand_near_object
    
    def proximity_condition():
        return is_hand_near_object(
            hand_landmarks, yolo_results, target_class,
            frame_width, frame_height, threshold
        )
    
    return wait_for_condition_change(proximity_condition, target_near, timeout)
