###Man musst den Videopath in config.py definieren und ROI BEreich von der Kiste hier

import cv2
import numpy as np
from .config import (KISTE_VIDEO, KISTE_ROI, KISTE_THRESHOLD)

class KisteDetector:
    def __init__(self,
                 roi=KISTE_ROI, # (575, 318, 300, 452), ## Hier ROI Bereich von Farbeabweichung definieren
                 threshold_stddev=KISTE_THRESHOLD, # 13.6, 
                 history_length=60,
                 min_agree=44):
        self.roi = roi
        self.threshold_stddev = threshold_stddev
        self.history_length = history_length
        self.min_agree = min_agree

        self._status_history = []
        self._last_stable_state = None  # True = leer, False = voll

    def _apply_roi(self, frame):
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w].copy()

    def update(self, frame):
        roi_img = self._apply_roi(frame)
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        stddev = np.std(gray_roi)

        is_empty_current = stddev < self.threshold_stddev

        self._status_history.append(is_empty_current)
        if len(self._status_history) > self.history_length:
            self._status_history.pop(0)

        stable_state = None
        if len(self._status_history) == self.history_length:
            count_empty = self._status_history.count(True)
            count_full = self.history_length - count_empty

            if self._last_stable_state is None:
                # Erstentscheidung
                if count_empty >= self.min_agree:
                    self._last_stable_state = True
                elif count_full >= self.min_agree:
                    self._last_stable_state = False
            else:
                # Zustand nur wechseln, wenn ausreichend Frames neuen Zustand bestätigen
                if self._last_stable_state is True and count_full >= self.min_agree:
                    self._last_stable_state = False
                elif self._last_stable_state is False and count_empty >= self.min_agree:
                    self._last_stable_state = True

            stable_state = self._last_stable_state

        # Wenn noch kein stabiler Zustand oder nicht genug Daten, Default: Voll (False)
        if stable_state is None:
            stable_state = self._last_stable_state if self._last_stable_state is not None else False

        # Overlay vorbereiten (kann optional ausgegeben werden)
        overlay = frame.copy()
        x, y, w, h = self.roi
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)
        label = "Kiste Voll" if stable_state is False else "Kiste Leer"
        cv2.putText(
            overlay,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )
        return stable_state, overlay


# === Globale Instanz für step_logic.py ===
kiste_detector = None  # Singleton-Instanz


# === Öffentliche Funktion, die von step_logic.py importiert wird ===
def kiste_fall(frame):
    """
    Führt eine Kistenstatusprüfung durch (leer/voll) basierend auf dem übergebenen Frame.

    Args:
        frame: Aktueller Videoframe (BGR)

    Returns:
        (bool, overlay): 
            True = leer, False = voll,
            overlay = annotierter Frame
    """
    global kiste_detector
    if kiste_detector is None:
        kiste_detector = KisteDetector()  # Initialisiere beim ersten Aufruf

    return kiste_detector.update(frame)

