"""
Main application entry point for Cylinder Tracking System.
"""
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
from collections import deque
import time

from utils import preprocess_image, draw_roi, get_centroid_from_bbox, get_clahe
from tracker import CustomTracker, TrackedObject
from modbus_server import ModbusServer


class CylinderTrackerApp:
    """Main application class."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize application with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Camera settings
        self.camera_source = self.config['camera']['source']
        self.camera_width = self.config['camera']['width']
        self.camera_height = self.config['camera']['height']
        
        # Model settings
        self.model_path = self.config['model']['path']
        self.conf_threshold = self.config['model']['conf_threshold']
        self.iou_threshold = self.config['model']['iou_threshold']
        self.use_masks = bool(self.config['model']['use_masks'])
        self.yolo_task = self.config['model']['task']
        
        # ROI settings
        self.max_rois = self.config['roi']['max_count']
        self.roi_colors = [tuple(color) for color in self.config['roi']['colors']]
        
        # Class colors (BGR format)
        self.class_colors = [tuple(color) for color in self.config['class_colors']]
        
        # Tracker settings
        tracker_config = self.config['tracker']
        self.tracker = CustomTracker(
            class_lock_threshold=tracker_config['class_lock_threshold'],
            class_change_threshold=tracker_config['class_change_threshold'],
            class_change_frames=tracker_config['class_change_frames'],
            max_disappeared=tracker_config['max_disappeared'],
            max_distance=tracker_config['max_distance'],
            iou_prune_threshold=tracker_config['iou_prune_threshold'],
            pixels_per_cm=tracker_config['pixels_per_cm']
        )
        self.pixels_per_cm = tracker_config['pixels_per_cm']
        
        # Preprocessing settings
        self.clahe_clip_limit = self.config['preprocessing']['clahe_clip_limit']
        self.clahe_tile_grid_size = tuple(self.config['preprocessing']['clahe_tile_grid_size'])
        self._clahe = get_clahe(self.clahe_clip_limit, self.clahe_tile_grid_size)
        
        # Modbus settings
        modbus_config = self.config['modbus']
        self.modbus_server = ModbusServer(
            host=modbus_config['host'],
            port=modbus_config['port'],
            max_rois=self.max_rois,
            max_objects_per_roi=modbus_config['max_objects_per_roi']
        )
        
        # Initialize camera
        #self.cap = cv2.VideoCapture(self.camera_source)
        self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.camera_source}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        
        # Initialize YOLO model
        print(f"Loading YOLO model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        print("Model loaded successfully.")
        
        # ROI storage
        self.rois: List[Tuple[int, int, int, int]] = []  # List of (x, y, w, h) tuples
        self.tracking_active = False
        
        # FPS tracking
        self.current_fps = 30.0  # Default to 30, will be updated
        self.fps_history = deque(maxlen=10)  # Store recent FPS values for smoothing
        self.fps_sum = 0.0

    
    def select_rois(self):
        """Allow user to select ROIs interactively."""
        print("\nROI Selection Mode")
        print("Press 's' to start selecting ROIs")
        print("Click and drag to select a region, then press Enter to confirm or Esc to cancel")
        print("Press 'q' when done selecting ROIs")
        
        self.rois = []
        selecting = True
        
        while selecting:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Draw existing ROIs
            for i, roi in enumerate(self.rois):
                color = self.roi_colors[i % len(self.roi_colors)]
                draw_roi(frame, roi, color, f"ROI {i+1}")
            
            # Show instructions
            cv2.putText(frame, f"ROIs selected: {len(self.rois)}/{self.max_rois}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to select ROI, 'q' to finish", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("ROI Selection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if len(self.rois) >= self.max_rois:
                    print(f"Maximum number of ROIs ({self.max_rois}) reached.")
                    continue
                
                # Select ROI
                roi = cv2.selectROI("ROI Selection", frame, False)
                if roi[2] > 0 and roi[3] > 0:  # Valid ROI
                    self.rois.append(roi)
                    print(f"ROI {len(self.rois)} selected: {roi}")
                cv2.destroyWindow("ROI Selection")
            
            elif key == ord('q'):
                if len(self.rois) > 0:
                    selecting = False
                else:
                    print("Please select at least one ROI before starting tracking.")
        
        cv2.destroyAllWindows()
        print(f"\nROI selection complete. {len(self.rois)} ROIs selected.")
        return len(self.rois) > 0
    
    def process_detections(self, results) -> List[Dict]:
        """
        Process YOLO detection results and convert to tracker format.
        
        Args:
            results: YOLO results object
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if results and len(results) > 0:
            result = results[0]

            if result.boxes is None:
                return detections

            boxes = result.boxes
            masks = result.masks if self.use_masks else None
            orig_h, orig_w = result.orig_shape

            for i in range(len(boxes)):
                box = boxes[i]
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = bbox
                bbox_tuple = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

                # Centroid from bbox center (cheap + stable)
                centroid = get_centroid_from_bbox(bbox, fmt="xyxy")

                mask_binary = None
                if self.use_masks and masks is not None and getattr(masks, 'data', None) is not None and i < len(masks.data):
                    mask = masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask, (orig_w, orig_h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

                det = {
                    'centroid': centroid,
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': bbox_tuple
                }

                if mask_binary is not None:
                    det['mask'] = mask_binary

                detections.append(det)
        
        return detections
    
    def draw_detections(self, frame, objects: List[TrackedObject], fps: float):
        """
        Draw detected objects on frame with appropriate colors.
        
        Args:
            frame: Frame to draw on
            roi_id: ROI ID (0-indexed)
            objects: List of tracked objects in ROI
            fps: Current frames per second
        """
        for obj in objects:
            # Get color based on class
            class_id = obj.current_class_id
            if 0 <= class_id < len(self.class_colors):
                color = self.class_colors[class_id]
            else:
                color = (255, 255, 255)  # Default white
            
            # Draw bounding box
            if obj.bbox:
                x, y, w, h = obj.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw centroid
            cx, cy = obj.centroid
            cv2.circle(frame, (cx, cy), 5, color, -1)
            
            # Draw label
            confidence_percent = int(obj.current_confidence * 100)
            speed = getattr(obj, "cached_speed", None)
            if speed is None:
                speed = obj.calculate_speed(self.pixels_per_cm, fps)
                obj.cached_speed = speed

            label_line1 = f"ID:{obj.object_id} {speed:.1f}cm/s"
            label_line2 = f"C:{class_id} {confidence_percent}%"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            line_spacing = 2

            (w1, h1), _ = cv2.getTextSize(label_line1, font, font_scale, thickness)
            (w2, h2), _ = cv2.getTextSize(label_line2, font, font_scale, thickness)
            text_width = max(w1, w2)
            total_text_height = h1 + h2 + line_spacing

            frame_h, frame_w = frame.shape[:2]
            tx = cx
            if tx + text_width + 5 >= frame_w:
                tx = max(0, frame_w - text_width - 6)

            top_y = cy - 5 - total_text_height
            if top_y <= 0:
                top_y = 6

            line1_y = top_y + h1
            line2_y = line1_y + line_spacing + h2

            bg_x1 = max(0, tx - 5)
            bg_y1 = max(0, top_y - 5)
            bg_x2 = min(frame_w - 1, tx + text_width + 5)
            bg_y2 = min(frame_h - 1, line2_y + 5)
            if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

            text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
            cv2.putText(frame, label_line1, (tx, line1_y), font, font_scale, text_color, thickness)
            cv2.putText(frame, label_line2, (tx, line2_y), font, font_scale, text_color, thickness)
    
    def run(self):
        """Main application loop."""
        # Select ROIs
        if not self.select_rois():
            print("No ROIs selected. Exiting.")
            return
        
        # Start Modbus server
        self.modbus_server.start()
        
        # Start tracking
        self.tracking_active = True
        print("\nTracking started. Press 'q' to quit.")
        
        # Initialize frame timing for FPS calculation
        self.last_frame_time = time.time()
        
        try:
            while self.tracking_active:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera.")
                    break
                
                # Preprocess image
                processed_frame = preprocess_image(
                    frame,
                    self.clahe_clip_limit,
                    self.clahe_tile_grid_size,
                    clahe=self._clahe,
                )
                
                # Run YOLO detection
                results = self.model.predict(
                    processed_frame,
                    task=self.yolo_task,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # Process detections
                all_detections = self.process_detections(results)
                
                # Update global tracker with all detections
                self.tracker.update(all_detections)
                
                # Get objects in each ROI
                roi_objects: Dict[int, List[TrackedObject]] = {}
                for i, roi in enumerate(self.rois):
                    objects = self.tracker.get_objects_in_roi(roi)
                    roi_objects[i + 1] = objects  # ROI IDs are 1-indexed
                
                # Calculate FPS for this frame
                current_time = time.time()
                frame_time = current_time - self.last_frame_time
                if frame_time > 0:
                    frame_fps = 1.0 / frame_time
                    if len(self.fps_history) == self.fps_history.maxlen:
                        self.fps_sum -= self.fps_history[0]
                    self.fps_history.append(frame_fps)
                    self.fps_sum += frame_fps
                    # Use rolling average of recent FPS values for stability
                    if len(self.fps_history) > 0:
                        self.current_fps = self.fps_sum / len(self.fps_history)
                self.last_frame_time = current_time

                # Cache per-object speed once per frame (after FPS update)
                for obj in self.tracker.get_tracked_objects():
                    obj.cached_speed = obj.calculate_speed(self.pixels_per_cm, self.current_fps)
                
                # Update Modbus registers (pass current FPS)
                self.modbus_server.update_registers(roi_objects, self.pixels_per_cm, self.current_fps)
                
                # Draw ROIs
                for i, roi in enumerate(self.rois):
                    color = self.roi_colors[i % len(self.roi_colors)]
                    draw_roi(frame, roi, color, f"ROI {i+1}")
                
                # Draw detections (pass current FPS)
                for i, roi in enumerate(self.rois):
                    objects = roi_objects.get(i + 1, [])
                    self.draw_detections(frame, objects, self.current_fps)
                
                cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                           (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Cylinder Tracking", frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.tracking_active = False
                # Check for tracker reset command (either 'c' key or Modbus command register bit 0)
                elif (key == ord('c')) or (self.modbus_server.get_and_clear_command_register() & 0x1):
                    self.tracker.reset()
                    print("Tracker reset")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            self.modbus_server.stop()
            print("Application closed.")


def main():
    """Main entry point."""
    try:
        app = CylinderTrackerApp()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

