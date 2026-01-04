#!/usr/bin/env python3

"""Sign detector node using template matching.

This module implements a lightweight template-matching based traffic-sign
detector for the Webots + ROS2 exercise. It loads pre-generated color
templates from the package `resources/` folder and scans incoming camera
frames using OpenCV's `matchTemplate` (TM_CCOEFF_NORMED).

The node publishes simple string tokens on `/traffic_sign` such as
`stop`, `yield` or `speed_limit_50` which the controller consumes.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import re
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import time


class SignDetector(Node):
    """ROS2 node that detects traffic signs by template matching.

    Behavior summary:
    - Loads all image files from `resources/` (including pre-scaled `_scaleNN`
      variants) into memory as BGR templates.
    - For each incoming `/car_camera/image` frame, scans the full image with
      each template using multi-channel BGR matching.
    - Accepts a detection when the best match >= per-type threshold.
    - Publishes a short string on `/traffic_sign` describing the sign.
    """

    def __init__(self):
        super().__init__('sign_detector')
        self.bridge = CvBridge()

        # Detection state and cooldowns
        self.last_sign_detected = None
        self.last_detection_time = 0
        # Cooldowns (seconds) per sign family to avoid repeated publishes
        self.cooldowns = {'stop': 2.0, 'yield': 2.0, 'speed_limit': 1.5}

        # ROS communication
        self.image_sub = self.create_subscription(Image, '/car_camera/image', self.image_callback, 10)
        self.sign_pub = self.create_publisher(String, '/traffic_sign', 10)

        # Load templates from package resources (color BGR images)
        self.base_templates = self.load_base_templates()

        # Per-type acceptance thresholds (TM_CCOEFF_NORMED)
        self.template_thresholds = {
            'stop': 0.60,
            'yield': 0.70,
            'speed_limit': 0.60
        }

        # Previously we required a gap between best and second-best; the
        # gap check is disabled by setting to 0.0 so ties are accepted.
        self.match_gap = 0.0

        # Used to rate-limit occasional debug/info logs
        self._last_match_log_time = 0.0
        
    def load_base_templates(self):
        """Load sign templates from the package `resources/` folder.

        Returns a dict mapping template keys to dicts with keys: 'bgr', 'h', 'w'.
        """
        templates = {}
        try:
            package_share_dir = get_package_share_directory('car_pkg')
            
            # Load from the package `resources` directory
            target_path = os.path.join(package_share_dir, 'resources')
            # Fallback to the workspace copy if package install is not present
            if not os.path.exists(target_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                target_path = os.path.join(current_dir, '..', 'resources')
                target_path = os.path.normpath(target_path)
            files = os.listdir(target_path) if os.path.exists(target_path) else []

            # Keep common image formats
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for filename in image_files:
                base_filename = filename
                if '_aug_' in filename:
                    base_parts = filename.split('_aug_')
                    base_filename = base_parts[0] + os.path.splitext(filename)[1]

                sign_type, speed_value = self.determine_sign_type(base_filename)

                if not sign_type:
                    continue

                filepath = os.path.join(target_path, filename)
                # Load color template (BGR). Use IMREAD_COLOR to ensure a
                # 3-channel image suitable for multi-channel matching.
                template_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if template_bgr is None:
                    continue

                h, w = template_bgr.shape[:2]

                # Do not perform runtime upscaling here; scaled variants are
                # generated offline by `create_augmented.py`.

                if sign_type == 'speed_limit' and speed_value:
                    template_key = f"speed_limit_{speed_value}_{filename}"
                else:
                    template_key = f"{sign_type}_{filename}"

                templates[template_key] = {
                    'bgr': template_bgr,
                    'h': h,
                    'w': w
                }
                    # Intentionally silent: do not emit info logs here.
                    
        except Exception as e:
            self.get_logger().error(f"Error loading templates: {str(e)}")
        
        return templates
    
    def determine_sign_type(self, filename):
        """Determine the sign type from a template filename.

        Returns (sign_type, speed_value) where speed_value is a string
        when a numeric speed is present in the filename (e.g. '50').
        """
        filename_lower = filename.lower()
        speed_value = None
        
        if 'yield' in filename_lower: 
            return 'yield', None
        elif 'stop' in filename_lower: 
            return 'stop', None
        elif 'speed' in filename_lower:
            numbers = re.findall(r'\d+', filename)
            if numbers:
                speed_value = numbers[0]
            return 'speed_limit', speed_value
        return None, None

    def image_callback(self, msg):
        """Process each incoming camera frame and publish detections."""
        # Note: camera frames from Webots are 512x256 (w x h).
        # Typical sign ROIs extracted by downstream logic are very small
        # (about 22x27). The detector upscales tiny input frames to a
        # MIN_WORKING_WIDTH so template matching has enough
        # resolution to work reliably.
        try:
            current_time = time.time()
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Adaptive working width: only downscale very large frames to limit
            # computation. Do not upscale full frames — instead we upscale tiny
            # ROIs before template matching so small crops (e.g. 22x27) gain
            # sufficient resolution without enlarging the whole image.
            MAX_WORKING_WIDTH = 512
            h_orig, w_orig = cv_image.shape[:2]
            if w_orig > MAX_WORKING_WIDTH:
                new_w = MAX_WORKING_WIDTH
                scale = new_w / float(w_orig)
                new_h = max(1, int(h_orig * scale))
                small_img = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1.0
                small_img = cv_image

            detected_sign, confidence, debug = self.detect_sign(small_img, original_image=cv_image, scale=scale)
            
            if detected_sign:
                time_since_last = current_time - self.last_detection_time
                base_sign_type = detected_sign.split('_')[0]
                cooldown = self.cooldowns.get(base_sign_type, 1.0)
                
                if time_since_last > cooldown or detected_sign != self.last_sign_detected:
                    
                    self.last_sign_detected = detected_sign
                    self.last_detection_time = current_time



                    sign_msg = String()
                    sign_msg.data = detected_sign
                    self.sign_pub.publish(sign_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def detect_sign(self, image, original_image=None, scale=1.0):
        """Template-only detection over the whole image.

        Scans the full image with pre-generated templates (no color/edge/shape heuristics).
        Returns (best_detection, best_confidence, debug_dict).
        """
        best_detection = None
        best_confidence = 0.0
        best_debug = None

        # Ensure we have a 3-channel BGR image for multi-channel matching
        try:
            img_bgr = image if (len(image.shape) == 3 and image.shape[2] == 3) else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        except Exception:
            # Fallback: attempt to use the original image directly
            img_bgr = image

        img_h, img_w = img_bgr.shape[:2]

        best_key = None
        second_best = 0.0

        # Iterate over all loaded templates
        for template_key, template_dict in self.base_templates.items():
                template_bgr = template_dict.get('bgr')
                if template_bgr is None:
                    continue

                t_h, t_w = template_bgr.shape[:2]
                # matchTemplate requires template to be <= image
                if t_h > img_h or t_w > img_w:
                    continue

                # Perform template matching on full color image
                try:
                    result = cv2.matchTemplate(img_bgr, template_bgr, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                except Exception:
                    continue

                # Determine sign_type from key
                if template_key.startswith('speed_limit'):
                    sign_type = 'speed_limit'
                else:
                    sign_type = template_key.split('_')[0]

                thresh = self.template_thresholds.get(sign_type, 0.4)
                # Update best/second-best trackers
                if max_val > best_confidence:
                    second_best = best_confidence
                    best_confidence = float(max_val)
                    best_key = template_key
                    # bbox in image coords
                    x, y = max_loc
                    bbox = (int(x), int(y), int(t_w), int(t_h))
                    best_debug = {'template_key': template_key, 'bbox': bbox, 'type': sign_type, 'template_val': float(max_val)}
                elif max_val > second_best:
                    second_best = float(max_val)

        # Apply gap check to avoid ambiguous/weak matches
        if best_key is None:
            # Silent: do not emit info logs when no templates match
            return None, 0.0, None

        # Determine sign_type from best key and threshold
        if best_key.startswith('speed_limit'):
            best_type = 'speed_limit'
        else:
            best_type = best_key.split('_')[0]

        thresh = self.template_thresholds.get(best_type, 0.4)
        gap = best_confidence - second_best

        if best_confidence >= thresh and gap >= self.match_gap:
            # Create detection string
            if best_type == 'speed_limit' and 'speed_limit_' in best_key:
                m = re.match(r'speed_limit_(\d+)', best_key)
                speed_num = m.group(1) if m else None
                detection = f"speed_limit_{speed_num}" if speed_num else 'speed_limit_55'
            else:
                detection = best_type

            # attach second_best for debugging
            if best_debug is not None:
                best_debug['second_best'] = float(second_best)
            return detection, best_confidence, best_debug

        # Ambiguous or too-weak match: remain silent (no info logging)
        return None, 0.0, best_debug

def main(args=None):
    rclpy.init(args=args)
    node = SignDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
