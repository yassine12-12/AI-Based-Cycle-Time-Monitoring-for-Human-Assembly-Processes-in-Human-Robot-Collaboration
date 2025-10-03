"""
Configuration module for assembly step tracking application.
Contains all configurable parameters, step definitions, and conditions.

MODULAR VERSION - Organized in modular/components/
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# === Main Application Config ===
YOLO_MODEL_PATH = r"D:\Studium\SoSe2025\IAT PRJ\iat-final\best_jasin.pt"
VIDEO_SOURCE    = r"D:\Studium\SoSe2025\IAT PRJ\videos_dual\basler_1_ganz.mp4"
CONFIDENCE      = 0.40
VIDEO_SPEED     = 1  # Process every Nth frame (set to 1 for real-time, >1 for speed-up)

KISTE_VIDEO = r"D:\Studium\SoSe2025\IAT PRJ\videos_dual\kiste_1_ganz.mp4"
KISTE_ROI = (168, 120, 200, 240) # (635, 185, 406, 644)
KISTE_THRESHOLD = 6.5

# === Step Configuration Moved ===
# Step configurations have been moved to step_logic.py for better organization
# See get_step_configuration() function in step_logic.py

def get_step_config(step_name: str) -> Optional[Dict[str, Any]]:
    """
    DEPRECATED: This function is kept for backwards compatibility.
    Use get_step_configuration() in step_logic.py instead.
    
    Args:
        step_name: The name of the assembly step
        
    Returns:
        dict: Always returns None - configurations moved to step_logic.py
    """
    return None  # All configurations moved to step_logic.py

# === Assembly Step Definition ===
@dataclass
class AssemblyStep:
    name: str
    yolo_class: str            # class-name as stored in best.pt
    wait_for_human: bool = False
    roi: Optional[Tuple[int, int, int, int]] = None          # (x, y, w, h) or None for full frame

# To enable ROI, set roi=(x, y, w, h) for each step below.
# To disable ROI, set roi=None for each step below.
ASSEMBLY_STEPS: List[AssemblyStep] = [
    # Motor 1 sequence - using actual YOLO classes (Sensorbaugruppe, Akku, Motor, LModul, Uno, Erweiterungsboard, platte, Kupfersaeule, rad, teilA, teilB)
    AssemblyStep("Robot positions Motor 1", "Motor", wait_for_human=True), # False
    AssemblyStep("Human mounts Motor 1", "Motor", wait_for_human=False),

    # Motor 2 sequence
    AssemblyStep("Robot positions Motor 2", "Motor", wait_for_human=True), # False
    AssemblyStep("Human mounts Motor 2", "Motor", wait_for_human=False),

    # Motor 3 sequence
    AssemblyStep("Robot positions Motor 3", "Motor", wait_for_human=True), # False
    AssemblyStep("Human mounts Motor 3", "Motor", wait_for_human=False),

    # Motor 4 sequence
    AssemblyStep("Robot positions Motor 4", "Motor", wait_for_human=True), # False
    AssemblyStep("Human mounts Motor 4", "Motor", wait_for_human=False),
    
    # LModul sequence
    AssemblyStep("Robot gives LModul", "LModul", wait_for_human=True), # False
    AssemblyStep("Human mounts LModul", "LModul", wait_for_human=False),    
    
    # Uno sequence 
    AssemblyStep("Robot gives Uno", "Uno", wait_for_human=True), # False
    AssemblyStep("Human mounts Uno", "Uno", wait_for_human=False),

    # Erweiterungsboard sequence
    AssemblyStep("Human mounts Erweiterungsboard", "Erweiterungsboard", wait_for_human=False),
    
    # Akku sequence
    AssemblyStep("Robot gives Akku", "Akku", wait_for_human=True), # False
    AssemblyStep("Human mounts Akku", "Akku", wait_for_human=False),

    # Kupfersaeule sequence
    AssemblyStep("Human mounts Kupfersaeule", "Kupfersaeule", wait_for_human=False),
    
    # Sensorbaugruppe sequence
    AssemblyStep("Human mounts Sensorbaugruppe", "Sensorbaugruppe", wait_for_human=False),

    # teilB sequence
    AssemblyStep("Human mounts teil B on A", "Motor", wait_for_human=False),

    # rad sequence
    AssemblyStep("Human mounts rad", "rad", wait_for_human=False),   

    # Montage ends
    AssemblyStep("Montage ends", "None", wait_for_human=True)
]
