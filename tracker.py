"""
Custom object tracker with class locking and speed calculation.
"""
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import cv2
from utils import calculate_detection_overlap, calculate_bbox_overlap


class TrackedObject:
    """Represents a tracked object with its properties."""
    
    def __init__(self, object_id, centroid, class_id, confidence, mask=None, bbox=None,
                 class_lock_threshold=0.7, class_change_threshold=0.8, class_change_frames=5):
        self.object_id = object_id
        self.centroid = centroid  # (x, y)
        self.locked_class_id = None
        self.current_class_id = class_id
        self.current_confidence = confidence
        self.mask = mask
        self.bbox = bbox
        self.class_history = deque(maxlen=10)  # Store recent class predictions
        self.confidence_history = deque(maxlen=10)
        self.centroid_history = deque(maxlen=10)  # For speed calculation
        self.speed_history = deque(maxlen=5)  # Store recent speed calculations for filtering
        self.filtered_speed = 0.0  # Filtered/averaged speed value
        self.disappeared = 0
        self.class_change_counter = 0
        self.last_class_change_candidate = None
        
        # Store thresholds
        self.class_lock_threshold = class_lock_threshold
        self.class_change_threshold = class_change_threshold
        self.class_change_frames = class_change_frames
        
        # Initialize history
        self.centroid_history.append(centroid)
        self.class_history.append((class_id, confidence))
        self.confidence_history.append(confidence)
    
    def update(self, centroid, class_id, confidence, mask=None, bbox=None):
        """Update object with new detection."""
        self.centroid = centroid
        self.centroid_history.append(centroid)
        self.mask = mask
        self.bbox = bbox
        
        # Update class history
        self.class_history.append((class_id, confidence))
        self.confidence_history.append(confidence)
        
        # Class locking logic
        if self.locked_class_id is None:
            # Lock class if confidence exceeds threshold
            if confidence >= self.class_lock_threshold:
                self.locked_class_id = class_id
                self.current_class_id = class_id
                self.current_confidence = confidence
                self.class_change_counter = 0
                self.last_class_change_candidate = None
        else:
            # Class is locked, check if we should allow change
            if class_id == self.locked_class_id:
                # Same class, update normally
                self.current_class_id = class_id
                self.current_confidence = confidence
                self.class_change_counter = 0
                self.last_class_change_candidate = None
            else:
                # Different class detected
                if confidence >= self.class_change_threshold:
                    if self.last_class_change_candidate == class_id:
                        self.class_change_counter += 1
                    else:
                        self.class_change_counter = 1
                        self.last_class_change_candidate = class_id
                    
                    # Allow class change if threshold exceeded for required frames
                    if self.class_change_counter >= self.class_change_frames:
                        self.locked_class_id = class_id
                        self.current_class_id = class_id
                        self.current_confidence = confidence
                        self.class_change_counter = 0
                        self.last_class_change_candidate = None
                    else:
                        # Keep locked class, set confidence to 0
                        self.current_class_id = self.locked_class_id
                        self.current_confidence = 0.0
                else:
                    # Confidence too low, keep locked class with 0 confidence
                    self.current_class_id = self.locked_class_id
                    self.current_confidence = 0.0
                    self.class_change_counter = 0
                    self.last_class_change_candidate = None
        
        self.disappeared = 0
    
    def mark_missing(self):
        """Mark object as missing for one frame."""
        self.disappeared += 1
    
    def calculate_speed(self, pixels_per_cm, fps=30.0):
        """
        Calculate speed in cm/s based on centroid movement with filtering and averaging.
        
        Args:
            pixels_per_cm: Conversion ratio from pixels to centimeters
            fps: Frames per second (default 30.0)
            
        Returns:
            Filtered speed in cm/s
        """
        if len(self.centroid_history) < 2:
            return 0.0
        
        # Method 1: Calculate instantaneous speed from last two points
        prev_centroid = self.centroid_history[-2]
        curr_centroid = self.centroid_history[-1]
        
        dx = curr_centroid[0] - prev_centroid[0]
        dy = curr_centroid[1] - prev_centroid[1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        distance_cm = distance_pixels / pixels_per_cm
        instantaneous_speed = distance_cm * fps
        
        # Method 2: Calculate average speed over multiple points for smoother result
        if len(self.centroid_history) >= 3:
            # Use linear regression on recent points for better velocity estimation
            # Take last N points (up to 5 for stability)
            num_points = min(5, len(self.centroid_history))
            points = list(self.centroid_history)[-num_points:]
            
            # Calculate total distance over the time period
            total_distance_pixels = 0.0
            for i in range(1, len(points)):
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
                total_distance_pixels += np.sqrt(dx**2 + dy**2)
            
            # Convert to cm
            total_distance_cm = total_distance_pixels / pixels_per_cm
            
            # Time period = (num_points - 1) frames
            time_period_frames = num_points - 1
            time_period_seconds = time_period_frames / fps
            
            if time_period_seconds > 0:
                average_speed = total_distance_cm / time_period_seconds
            else:
                average_speed = instantaneous_speed
        else:
            average_speed = instantaneous_speed
        
        # Combine instantaneous and average speeds (weighted average)
        # Give more weight to average speed for stability
        combined_speed = 0.3 * instantaneous_speed + 0.7 * average_speed
        
        # Add to speed history
        self.speed_history.append(combined_speed)
        
        # Apply exponential moving average filter for smoothness
        alpha = 0.3  # Smoothing factor (0-1, lower = more smoothing)
        if len(self.speed_history) == 1:
            self.filtered_speed = combined_speed
        else:
            # Exponential moving average: new = alpha * current + (1-alpha) * previous
            self.filtered_speed = alpha * combined_speed + (1 - alpha) * self.filtered_speed
        
        # Additional outlier filtering: if new speed is very different from filtered, reduce impact
        if len(self.speed_history) > 1:
            speed_diff = abs(combined_speed - self.filtered_speed)
            max_expected_change = 50.0  # cm/s - reasonable max change per frame
            
            if speed_diff > max_expected_change:
                # Outlier detected - use more aggressive smoothing
                alpha = 0.1
                self.filtered_speed = alpha * combined_speed + (1 - alpha) * self.filtered_speed
        
        return max(0.0, self.filtered_speed)  # Ensure non-negative


class CustomTracker:
    """Custom object tracker with class locking and reidentification."""
    
    def __init__(self, class_lock_threshold=0.7, class_change_threshold=0.8,
                 class_change_frames=5, max_disappeared=10, max_distance=100,
                 pixels_per_cm=10.0):
        self.class_lock_threshold = class_lock_threshold
        self.class_change_threshold = class_change_threshold
        self.class_change_frames = class_change_frames
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.pixels_per_cm = pixels_per_cm
        
        self.next_id = 0
        self.objects: Dict[int, TrackedObject] = {}
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update(self, detections: List[Dict]):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries with keys:
                - 'centroid': (x, y) tuple
                - 'class_id': int
                - 'confidence': float
                - 'mask': optional binary mask
                - 'bbox': optional bounding box
        """
        # Mark all objects as missing initially
        for obj_id, obj in self.objects.items():
            obj.mark_missing()
        
        # If no detections, just mark objects as missing
        if len(detections) == 0:
            # Remove objects that have been missing too long
            to_remove = [obj_id for obj_id, obj in self.objects.items() 
                        if obj.disappeared >= self.max_disappeared]
            for obj_id in to_remove:
                del self.objects[obj_id]
            return
        
        # STEP 1: Filter out overlapping detections from raw detection list
        # This prevents duplicate objects from being created
        filtered_detections = []
        detection_indices_to_keep = []
        
        for i, det in enumerate(detections):
            overlaps_with_kept = False
            
            # Check if this detection overlaps with any already kept detection
            for j, kept_det in enumerate(filtered_detections):
                if calculate_detection_overlap(det, kept_det, overlap_threshold=0.5, bbox_format="xywh"):
                    overlaps_with_kept = True
                    # If current detection has higher confidence, replace the kept one
                    if det.get('confidence', 0) > kept_det.get('confidence', 0):
                        filtered_detections[j] = det
                        detection_indices_to_keep[j] = i
                    break
            
            if not overlaps_with_kept:
                filtered_detections.append(det)
                detection_indices_to_keep.append(i)
        
        # Use filtered detections for matching
        detections = filtered_detections
        
        # STEP 2: Match detections to existing objects by distance
        matched_detections = set()
        matched_objects = set()
        
        # First pass: match objects that haven't disappeared
        for obj_id, obj in self.objects.items():
            if obj.disappeared == 0:
                continue  # Already matched in this frame
            
            best_match = None
            best_distance = float('inf')
            
            for i, det in enumerate(detections):
                if i in matched_detections:
                    continue
                
                distance = self._calculate_distance(obj.centroid, det['centroid'])
                if distance < self.max_distance and distance < best_distance:
                    best_match = i
                    best_distance = distance
            
            if best_match is not None:
                det = detections[best_match]
                obj.update(
                    det['centroid'],
                    det['class_id'],
                    det['confidence'],
                    det.get('mask'),
                    det.get('bbox')
                )
                matched_detections.add(best_match)
                matched_objects.add(obj_id)
        
        # Second pass: match disappeared objects (reidentification)
        for obj_id, obj in self.objects.items():
            if obj_id in matched_objects:
                continue
            
            if obj.disappeared >= self.max_disappeared:
                continue  # Too long disappeared, don't try to match
            
            best_match = None
            best_distance = float('inf')
            
            for i, det in enumerate(detections):
                if i in matched_detections:
                    continue
                
                distance = self._calculate_distance(obj.centroid, det['centroid'])
                if distance < self.max_distance and distance < best_distance:
                    best_match = i
                    best_distance = distance
            
            if best_match is not None:
                det = detections[best_match]
                obj.update(
                    det['centroid'],
                    det['class_id'],
                    det['confidence'],
                    det.get('mask'),
                    det.get('bbox')
                )
                matched_detections.add(best_match)
                matched_objects.add(obj_id)
        
        # STEP 3: Check unmatched detections for overlap with existing objects
        # (even if they weren't matched by distance - this catches cases where
        # an object moved but still overlaps with a detection)
        for obj_id, obj in self.objects.items():
            if obj_id in matched_objects:
                continue  # Already matched
            
            best_match = None
            best_overlap_ratio = 0.0
            
            for i, det in enumerate(detections):
                if i in matched_detections:
                    continue
                
                # Create a detection dict from the tracked object for comparison
                obj_det = {
                    'centroid': obj.centroid,
                    'mask': obj.mask,
                    'bbox': obj.bbox
                }
                
                # Calculate overlap ratio
                if False and obj_det.get('mask') is not None and det.get('mask') is not None:
                    overlap_ratio = calculate_mask_overlap(obj_det['mask'], det['mask'])
                elif obj_det.get('bbox') is not None and det.get('bbox') is not None:
                    overlap_ratio = calculate_bbox_overlap(obj_det['bbox'], det['bbox'])
                else:
                    continue
                
                if overlap_ratio >= 0.5 and overlap_ratio > best_overlap_ratio:
                    best_match = i
                    best_overlap_ratio = overlap_ratio
            
            if best_match is not None:
                det = detections[best_match]
                obj.update(
                    det['centroid'],
                    det['class_id'],
                    det['confidence'],
                    det.get('mask'),
                    det.get('bbox')
                )
                matched_detections.add(best_match)
                matched_objects.add(obj_id)
        
        # STEP 4: Create new objects for remaining unmatched detections
        # (These have already been filtered for overlaps in STEP 1)
        for i, det in enumerate(detections):
            if i not in matched_detections:
                new_obj = TrackedObject(
                    self.next_id,
                    det['centroid'],
                    det['class_id'],
                    det['confidence'],
                    det.get('mask'),
                    det.get('bbox'),
                    self.class_lock_threshold,
                    self.class_change_threshold,
                    self.class_change_frames
                )
                self.objects[self.next_id] = new_obj
                self.next_id += 1
        
        # STEP 5: Check for overlapping tracked objects and merge them
        # (Keep the oldest ID when merging)
        objects_to_remove = set()
        object_list = list(self.objects.items())
        
        for i, (obj_id1, obj1) in enumerate(object_list):
            if obj_id1 in objects_to_remove:
                continue
            
            for j, (obj_id2, obj2) in enumerate(object_list[i+1:], start=i+1):
                if obj_id2 in objects_to_remove:
                    continue
                
                # Create detection dicts for comparison
                obj1_det = {
                    'centroid': obj1.centroid,
                    'mask': obj1.mask,
                    'bbox': obj1.bbox
                }
                obj2_det = {
                    'centroid': obj2.centroid,
                    'mask': obj2.mask,
                    'bbox': obj2.bbox
                }
                
                if calculate_detection_overlap(obj1_det, obj2_det, overlap_threshold=0.5):
                    # Objects overlap - keep the one with older ID (smaller number)
                    if obj_id1 < obj_id2:
                        # Keep obj1, remove obj2
                        objects_to_remove.add(obj_id2)
                    else:
                        # Keep obj2, remove obj1
                        objects_to_remove.add(obj_id1)
                        break  # obj1 is being removed, no need to check further
        
        # Remove overlapping objects
        for obj_id in objects_to_remove:
            if obj_id in self.objects:
                del self.objects[obj_id]
        
        # Remove objects that have been missing too long
        to_remove = [obj_id for obj_id, obj in self.objects.items() 
                    if obj.disappeared >= self.max_disappeared and obj_id not in matched_objects]
        for obj_id in to_remove:
            del self.objects[obj_id]
    
    def get_tracked_objects(self) -> List[TrackedObject]:
        """Get list of all currently tracked objects."""
        return list(self.objects.values())
    
    def get_objects_in_roi(self, roi) -> List[TrackedObject]:
        """
        Get tracked objects within a specific ROI.
        
        Args:
            roi: ROI coordinates (x, y, w, h)
            
        Returns:
            List of TrackedObject instances in the ROI
        """
        rx, ry, rw, rh = roi
        objects_in_roi = []
        
        for obj in self.objects.values():
            cx, cy = obj.centroid
            if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
                objects_in_roi.append(obj)
        
        return objects_in_roi

