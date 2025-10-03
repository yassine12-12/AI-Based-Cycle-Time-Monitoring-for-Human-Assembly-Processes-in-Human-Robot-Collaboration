
"""
Main application module for assembly step tracking.
Refactored from the original app.py to use modular components.

MODULAR VERSION - Organized in modular/components/
"""

import cv2
from datetime import datetime
import os
import sys
import csv
import mediapipe as mp
from ultralytics import YOLO

# Import our modular components
from .config import (
    YOLO_MODEL_PATH, VIDEO_SOURCE, CONFIDENCE, VIDEO_SPEED, ASSEMBLY_STEPS
)
from .config import (KISTE_VIDEO, KISTE_ROI, KISTE_THRESHOLD)

from .vision_utils import detect_yolo_objects, draw_detection_overlay, apply_roi
from .step_logic import StepManager, StepClock, human_step_logic
from .timer_utils import StepTimer
from .vibration_utils import DetectionSmoother
from .kiste_utils import kiste_fall
from .vision_utils import detect_aruco_ids


def main():
    """Main application loop for assembly step tracking."""
    
    # Initialize YOLO model
    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL_PATH)
    print(f"YOLO model loaded from: {YOLO_MODEL_PATH}")
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Initialize step management
    step_manager = StepManager(ASSEMBLY_STEPS)
    step_clock = StepClock(fps=30)
    step_timer = StepTimer()

    # Initialize DetectionSmoother (Vibrationsfunktion)
    detection_smoother = DetectionSmoother(hold_frames=10)  ## Vibrationsfunktion:
    
    # Initialize video capture
    print(f"Opening video: {VIDEO_SOURCE}")

    cap_main = cv2.VideoCapture(VIDEO_SOURCE)
    # cap_main.set(cv2.CAP_PROP_POS_MSEC, 14.2 * 60 * 1000)  # set video begin timestamp
    cap_kiste = cv2.VideoCapture(KISTE_VIDEO)
    # cap_kiste.set(cv2.CAP_PROP_POS_MSEC, 14.2 * 60 * 1000) # set video begin timestamp
    
    if not cap_main.isOpened() or not cap_kiste.isOpened():
        print("Error: Could not open one or both video sources.")
        return
    
    frame_count = 0
    frame_index = None
    
    print("\n=== Assembly Step Tracking Started ===")
    print("Press 'q' to quit, 'n' to advance to next step manually")
    print("=" * 50)

    hand_detected_count = 0
    hand_detected_continuous = False
    
    while True:
        ret_main, frame_main = cap_main.read()
        ret_kiste, frame_kiste = cap_kiste.read()
        if not ret_main:
            print("End of video or failed to read frame.")
            break
        if ret_kiste:  ## Kistefunktion:
            kiste_status_raw, _ = kiste_fall(frame_kiste)  ## Kistefunktion:
            # Änderung: Nur voll oder leer
            kiste_status = bool(kiste_status_raw)  # False = voll, True = leer
        else:  ## Kistefunktion:
            kiste_status = None  ## Kistefunktion:
        # Skip frames for speed control
        frame_count += 1
        if frame_count % VIDEO_SPEED != 0:
            continue

        # Get current step
        current_step = step_manager.get_current_step()
        if current_step is None:
            # All steps completed
            print("\nAll assembly steps completed!")
            break
        
        # Start timing for new step
        frame_index = cap_main.get(cv2.CAP_PROP_POS_FRAMES)
        if current_step.name not in step_clock.step_start_frames and step_manager.begin: # step_start_times
            step_clock.start_step(current_step.name, frame_index)
            step_timer.reset()
            # Step start info
            print(f"\nStarting: {current_step.name}")
            print(f"   Target object: {current_step.yolo_class}")
        
        # Apply ROI if specified
        processed_frame = apply_roi(frame_main, current_step.roi)
        frame_height, frame_width = processed_frame.shape[:2]

        detected_ids, centers, corner_map = detect_aruco_ids(processed_frame)

        if detected_ids:
            # Draw boxes around detected markers
            for marker_id, corners in corner_map.items():
                cv2.polylines(processed_frame, [corners.astype(int)], True, (0, 255, 0), 2)
                # Draw ID text near the first corner
                corner = corners[0][0]
                cv2.putText(processed_frame, str(marker_id), (int(corner[0]), int(corner[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Draw ArUco markers for debugging
        for aruco_id, (cx, cy) in centers.items():
            cv2.putText(processed_frame, f"ID: {aruco_id}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        hand_results = hands.process(rgb_frame)
        hand_landmarks = hand_results.multi_hand_landmarks

        if hand_landmarks:  
            if hand_detected_continuous:  
                hand_detected_count += 1
            else:  
                hand_detected_count = 1
                hand_detected_continuous = True
        else: 
            hand_detected_count = 0
            hand_detected_continuous = False
        
        # Process with YOLO
        yolo_results = model(processed_frame, conf=CONFIDENCE, verbose=False)
        classes_in_frame, _ = detect_yolo_objects(yolo_results)
        
        # Update DetectionSmoother with current classes
        detection_smoother.update(classes_in_frame)  ## Vibrationsfunktion:
        
        # --- Kiste Status prüfen ---
        # Achtung: Hier nochmal kiste_fall auf current_frame angewendet,
        # kann ggf. doppelt sein, ist aber so im Originalcode
        # if ret_kiste:
        #     kiste_status_raw = kiste_fall(frame_main)[0]
        #     kiste_status = bool(kiste_status_raw)  ## Änderung: nur voll oder leer
        # else:
        #     kiste_status = None  ## Kistefunktion:
        # # Kiste-Status Debug (optional)
        # if frame_count % 50 == 0:
        #     print(f"DEBUG: Kiste status: {kiste_status}")  ## Kistefunktion:

        # Debug: Print all detected classes (only occasionally to avoid spam)
        if frame_count % 30 == 0:  # Every 30 frames
            print(f"DEBUG: All detected classes: {classes_in_frame}")
        
        # Check if step requires human confirmation
        step_completed = False
        if current_step.wait_for_human:
            if current_step.name != "Montage ends":
                step_manager.begin = True
            # Steps that require manual confirmation (robot actions)
            print(f"Waiting for: {current_step.name}")
            print("   Press 'y' to confirm completion, 'n' to continue waiting...")
            
            # Display frame with overlays
            display_frame = draw_detection_overlay(
                processed_frame.copy(), yolo_results, hand_landmarks, mp_drawing, mp_hands
            )
            
            # Add step information overlay
            add_step_info_overlay(display_frame, current_step, step_manager, step_clock, frame_index)
            
            # Add detection status and distance (same as human steps)
            add_detection_status_overlay(display_frame, classes_in_frame, current_step.yolo_class, hand_landmarks, yolo_results)
            
            # Optional: Kiste-Status als Overlay anzeigen
            kiste_text = f"KISTE STATUS: {'Voll' if not kiste_status else 'Leer'}"  ## Kistefunktion:
            color = (0, 255, 0) if kiste_status else (0, 0, 255)  ## Kistefunktion:
            h_kiste, w_kiste = frame_kiste.shape[:2]
            cv2.putText(frame_kiste, kiste_text, (10, h_kiste - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            draw_roi(frame_kiste, KISTE_ROI, color=(0, 255, 255), label="KISTE ROI")
            
            cv2.imshow('Assembly Step Tracking', display_frame)
            cv2.imshow("Kiste", frame_kiste)
            
            if hand_detected_count > 1:
                step_completed = True

            if current_step.name == "Montage ends":
                key = cv2.waitKey(1) & 0xFF
                if key == ord('y'):
                    step_completed = True
                elif key == ord('q'):
                    break
                else:
                    continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                step_completed = True
            elif key == ord('q'):
                break
                
        else:
            # Human action steps - use automated detection    
            step_completed = human_step_logic(
                current_step, hand_landmarks, yolo_results,
                frame_width, frame_height, step_timer, 
                step_manager,
                is_kiste_empty=kiste_status,
                detection_smoother=detection_smoother, current_frame=processed_frame,  ## Vibrationsfunktion und Kistefunktion
                detected_ids=detected_ids
            )
            
            # Display frame with overlays
            display_frame = draw_detection_overlay(
                processed_frame.copy(), yolo_results, hand_landmarks, mp_drawing, mp_hands
            )
            
            # Add step information overlay
            add_step_info_overlay(display_frame, current_step, step_manager, step_clock, frame_index)
            
            # Add detection status and distance (same as human steps)
            add_detection_status_overlay(display_frame, classes_in_frame, current_step.yolo_class, hand_landmarks, yolo_results)
            
            # Optional: Kiste-Status als Overlay anzeigen
            kiste_text = f"KISTE STATUS: {'Voll' if not kiste_status else 'Leer'}"  ## Kistefunktion:
            color = (0, 255, 0) if kiste_status else (0, 0, 255)  ## Kistefunktion:
            h_kiste, w_kiste = frame_kiste.shape[:2]
            cv2.putText(frame_kiste, kiste_text, (10, h_kiste - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            draw_roi(frame_kiste, KISTE_ROI, color=(0, 255, 255), label="KISTE ROI")
            
            cv2.imshow('Assembly Step Tracking', display_frame)
            cv2.imshow("Kiste", frame_kiste)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                step_completed = True  # Manual advance
        
        # Advance to next step if completed
        if step_completed:
            step_clock.end_step(current_step.name, frame_index)
            duration = step_clock.step_durations.get(current_step.name, 0)
            # Step completion info
            print(f"Completed: {current_step.name} (Duration: {duration:.1f}s)")
            step_manager.advance_to_next_step()
            step_manager.begin = False
            step_manager.end = False
            step_timer.reset()
            
            hand_detected_count = 0
            hand_detected_continuous = False
    
    # Cleanup
    cap_main.release()
    cap_kiste.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # Print final summary
    print_final_summary(step_manager, step_clock)


def add_step_info_overlay(frame, current_step, step_manager, step_clock, frame_index):
    """Add step information overlay to the frame."""
    h, w = frame.shape[:2]
    
    # Progress information
    current_idx, total_steps = step_manager.get_progress()
    progress_text = f"Step {current_idx + 1}/{total_steps}"
    cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Current step name
    step_text = f"Current: {current_step.name}"
    cv2.putText(frame, step_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Step duration
    duration = step_clock.get_current_step_duration(frame_index)
    duration_text = f"Duration: {duration:.1f}s"
    cv2.putText(frame, duration_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Target object
    target_text = f"Target: {current_step.yolo_class}"
    cv2.putText(frame, target_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def add_detection_status_overlay(frame, classes_in_frame, target_class, hand_landmarks=None, yolo_results=None, target_present=None):
    """Add detection status overlay to the frame."""
    h, w = frame.shape[:2]
    
    # Use target_present from Vibrationsfunktion if provided
    detected = target_present if target_present is not None else (target_class in classes_in_frame)
    
    if detected:
        status_text = f"✓ {target_class} DETECTED"
        color = (0, 255, 0)  # Green
    else:
        status_text = f"✗ {target_class} NOT FOUND"
        color = (0, 0, 255)  # Red
    
    cv2.putText(frame, status_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Distance to target object and visual line
    if hand_landmarks and yolo_results:
        from .vision_utils import calculate_hand_to_object_distance, get_closest_hand_center, get_object_center
        
        distance, found_object = calculate_hand_to_object_distance(
            hand_landmarks, yolo_results, target_class, w, h
        )
        
        if found_object and distance != float('inf'):
            distance_text = f"Distance to {target_class}: {distance:.0f}px"
            distance_color = (0, 255, 255)  # Yellow
            if distance < 50:
                distance_color = (0, 255, 0)  # Green if very close
            elif distance < 100:
                distance_color = (0, 165, 255)  # Orange if close
            
            # Draw distance line between hand and object
            try:
                hand_center = get_closest_hand_center(hand_landmarks, w, h)
                object_center = get_object_center(yolo_results, target_class)
                
                if hand_center and object_center:
                    # Draw line
                    cv2.line(frame, hand_center, object_center, distance_color, 2)
                    # Draw circles at endpoints
                    cv2.circle(frame, hand_center, 5, (255, 0, 255), -1)  # Magenta for hand
                    cv2.circle(frame, object_center, 5, (255, 255, 0), -1)  # Cyan for object
            except Exception as e:
                # Debug: Show why visual line failed
                debug_text = f"Line draw failed: {str(e)[:30]}"
                cv2.putText(frame, debug_text, (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
        else:
            distance_text = f"Distance to {target_class}: N/A"
            distance_color = (128, 128, 128)  # Gray
            
            # Debug: Show why distance is N/A
            if not found_object:
                debug_text = f"Object '{target_class}' not detected"
            elif hand_landmarks is None:
                debug_text = "No hands detected"
            else:
                debug_text = "Distance calculation failed"
            cv2.putText(frame, debug_text, (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, distance_text, (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, distance_color, 2)
    else:
        # Debug: Show why distance calculation was skipped
        if not hand_landmarks:
            debug_text = "No hand landmarks available"
        elif not yolo_results:
            debug_text = "No YOLO results available"
        else:
            debug_text = "Distance calc skipped"
        cv2.putText(frame, debug_text, (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def print_final_summary(step_manager, step_clock, csv_path=None):
    """Print final summary of the assembly process."""
    print("\n" + "=" * 60)
    print("ASSEMBLY PROCESS SUMMARY")
    print("=" * 60)
    
    total_duration = step_clock.get_total_duration()
    completed_count = len(step_manager.completed_steps)
    total_count = len(step_manager.steps)
    
    print(f"Progress: {completed_count}/{total_count-1} steps completed") # "Montage ends" not included!
    print(f"Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Average time per step: {total_duration/max(completed_count, 1):.1f} seconds")
    
    print("\nStep Details:")
    step_data = []
    for step in step_manager.completed_steps:
        duration = step_clock.step_durations.get(step.name, 0)
        print(f"   {step.name}: {duration:.1f}s")
        step_data.append([step.name, f"{duration:.1f}"])
    
    if not step_manager.is_complete():
        current_step = step_manager.get_current_step()
        if current_step:
            print(f"\nStopped at: {current_step.name}")
    
    print("=" * 60)

    if csv_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(os.getcwd(), f"summary_{timestamp}.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Step Name", "Duration (s)"])
        writer.writerows(step_data)
        writer.writerow([])
        writer.writerow(["Progress", f"{completed_count}/{total_count}"])
        writer.writerow(["Total time (s)", f"{total_duration:.1f}"])
        writer.writerow(["Total time (min)", f"{total_duration/60:.1f}"])
        writer.writerow(["Average per step (s)", f"{total_duration/max(completed_count,1):.1f}"])

    print(f"\nSummary also exported to {csv_path}")

def draw_roi(frame, roi, color=(255, 255, 255), label=None, thickness=2):
    """
    Zeichne das ROI auf dem Frame. roi = (x, y, w, h). 
    Gib das modifizierte Frame zurück (in-place verändert).
    """
    if frame is None or roi is None:
        return frame
    x, y, w, h = roi

    # Randbeschnitt, um Koordinatenüberschreitung zu vermeiden.
    h_img, w_img = frame.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))

    p1 = (x, y)
    p2 = (x + w, y + h)
    cv2.rectangle(frame, p1, p2, color, thickness)

    if label:
        # Der Text befindet sich oberhalb des Rahmens. 
        # # Wenn er sich zu nah an der oberen Kante befindet, verschiebe ihn ein wenig nach unten.
        baseline = 0
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y - 8
        if ty - th - baseline < 0:
            ty = y + th + baseline + 8
        cv2.putText(frame, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

if __name__ == "__main__":
    main()
