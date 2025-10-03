# vibration_utils.py

"""
Enthält die DetectionSmoother-Klasse zur Verbesserung der YOLO-Erkennung
durch Stabilisierung (Vibrations-Logik).
"""

from collections import defaultdict
from typing import List

class DetectionSmoother:
    """
    Diese Klasse speichert erkannte Objektklassen über mehrere Frames hinweg.
    Ziel: YOLO-Schwankungen (z.B. kurzes Nicht-Erkennen) ausgleichen.
    """
    def __init__(self, hold_frames: int = 10):
        self.hold_frames = hold_frames
        self.counters = defaultdict(int)

    def update(self, current_detections: List[str]):
        # Alle aktiven Klassen herunterzählen
        for cls in list(self.counters.keys()):
            self.counters[cls] = max(0, self.counters[cls] - 1)

        # Neue erkannte Klassen zurücksetzen
        for cls in current_detections:
            self.counters[cls] = self.hold_frames

    def is_present(self, target_class: str) -> bool:
        return self.counters.get(target_class, 0) > 0

