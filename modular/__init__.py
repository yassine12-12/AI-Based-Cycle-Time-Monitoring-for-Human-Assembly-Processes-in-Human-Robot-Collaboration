"""
Modular Assembly Step Tracking Components

This package contains the refactored modular components for assembly step tracking.
"""

from .components.config import (
    ASSEMBLY_STEPS, 
    YOLO_MODEL_PATH, 
    VIDEO_SOURCE, 
    #STEP_DISTANCE_CONFIG,
    get_step_config,
    AssemblyStep
)

from .components.vision_utils import (
    detect_yolo_objects,
    draw_detection_overlay,
    apply_roi,
    calculate_hand_to_object_distance,
    is_hand_near_object
)

from .components.timer_utils import (
    StepTimer,
    wait_for_duration,
    wait_for_condition_change
)

from .components.step_logic import (
    StepManager,
    StepClock,
    human_step_logic,
    get_step_configuration  # New centralized config function
)

from .components.kiste_utils import kiste_fall
from .components.vibration_utils import DetectionSmoother

from .components.main_app import main

__version__ = "2.0.0"
__author__ = "Assembly Tracking Team"

__all__ = [
    # Configuration
    'ASSEMBLY_STEPS', 'YOLO_MODEL_PATH', 'VIDEO_SOURCE', 
    'STEP_DISTANCE_CONFIG', 'get_step_config', 'AssemblyStep',
    
    # Vision utilities
    'detect_yolo_objects', 'draw_detection_overlay', 'apply_roi',
    'calculate_hand_to_object_distance', 'is_hand_near_object',
    
    # Timer utilities
    'StepTimer', 'wait_for_duration', 'wait_for_condition_change',
    
    # Step logic
    'StepManager', 'StepClock', 'human_step_logic',

     # kiste_utils
    'kiste_fall'
    
    # Main application
    'main'
]
