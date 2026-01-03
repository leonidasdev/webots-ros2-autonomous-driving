#!/usr/bin/env python3

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
try:
    import pytesseract
    _HAS_PYTESSERACT = True
except Exception:
    _HAS_PYTESSERACT = False

class SignDetector(Node):
    def __init__(self):
        super().__init__('sign_detector')
        self.bridge = CvBridge()
        
        # Control de estado y cooldowns
        self.last_sign_detected = None
        self.last_detection_time = 0
        self.cooldowns = {'stop': 2.0, 'yield': 2.0, 'speed_limit': 1.5}
        
        # Comunicación ROS
        self.image_sub = self.create_subscription(
            Image, '/car_camera/image', self.image_callback, 10)
        self.sign_pub = self.create_publisher(String, '/traffic_sign', 10)
        
        # Carga de plantillas
        self.base_templates = self.load_base_templates()

        # Reutilizables para reducir allocs por frame
        self.morph_kernel = np.ones((5, 5), np.uint8)
        self.small_kernel = np.ones((3, 3), np.uint8)
        
        # Ponderaciones
        self.weights = {
            'stop': {'shape': 0.50, 'color': 0.30, 'template': 0.20},
            'yield': {'shape': 0.45, 'color': 0.40, 'template': 0.15},
            'speed_limit': {'shape': 0.50, 'color': 0.30, 'template': 0.20}
        }
        
        # Umbrales
        self.thresholds = {
            'stop': {'shape': 0.20, 'color': 0.20, 'template': 0.20, 'total': 0.35},
            'yield': {'shape': 0.15, 'color': 0.25, 'template': 0.15, 'total': 0.35},
            # Make template threshold stricter for speed signs to reduce false positives
            'speed_limit': {'shape': 0.15, 'color': 0.25, 'template': 0.35, 'total': 0.40}
        }
        
        # Parámetros para detección de formas
        self.shape_params = {
            'stop': {'min_vertices': 6, 'max_vertices': 12, 'min_aspect': 0.5, 'max_aspect': 1.5},
            'yield': {'min_vertices': 3, 'max_vertices': 8, 'min_aspect': 0.4, 'max_aspect': 1.6},
            'speed_limit': {'min_vertices': 4, 'max_vertices': 10, 'min_aspect': 0.4, 'max_aspect': 2.2}
        }
        
    def load_base_templates(self):
        """Carga plantillas de señales desde el directorio de recursos"""
        templates = {}
        try:
            package_share_dir = get_package_share_directory('car_pkg')
            
            # Cargar desde el directorio `resources` en el paquete
            target_path = os.path.join(package_share_dir, 'resources')
            
            # Fallback al directorio actual
            if not os.path.exists(target_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                target_path = os.path.join(current_dir, '..', 'resources')
                target_path = os.path.normpath(target_path)
            
            files = os.listdir(target_path) if os.path.exists(target_path) else []
            
            # Filtrar archivos de imagen
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for filename in image_files:
                base_filename = filename
                if '_aug_' in filename:
                    base_parts = filename.split('_aug_')
                    base_filename = base_parts[0] + os.path.splitext(filename)[1]
                
                sign_type, speed_value = self.determine_sign_type(base_filename)
                
                if sign_type:
                    filepath = os.path.join(target_path, filename)
                    template_bgr = cv2.imread(filepath)
                    if template_bgr is not None:
                        try:
                            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
                        except Exception:
                            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

                        h, w = template_gray.shape[:2]

                        # Upscale very small templates to a minimum reference size
                        max_dim = max(h, w)
                        MIN_TEMPLATE_DIM = 48
                        if max_dim < MIN_TEMPLATE_DIM:
                            scale = int(np.ceil(MIN_TEMPLATE_DIM / float(max_dim)))
                            new_w = max(8, int(w * scale))
                            new_h = max(8, int(h * scale))
                            try:
                                template_gray = cv2.resize(template_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                                template_bgr = cv2.resize(template_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                                h, w = template_gray.shape[:2]
                            except Exception:
                                pass

                        if sign_type == 'speed_limit' and speed_value:
                            template_key = f"speed_limit_{speed_value}_{filename}"
                        else:
                            template_key = f"{sign_type}_{filename}"

                        templates[template_key] = {
                            'bgr': template_bgr,
                            'gray': template_gray,
                            'h': h,
                            'w': w
                        }
                    
        except Exception as e:
            self.get_logger().error(f"Error cargando plantillas: {str(e)}")
        
        return templates

    def extract_speed_from_roi(self, image, bbox):
        """Attempt to OCR digits from the speed-limit ROI. Returns int or None."""
        if not _HAS_PYTESSERACT:
            return None

        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        if roi.size == 0 or w < 12 or h < 12:
            return None

        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Preprocess for OCR: resize and adaptive threshold
        scale = max(1, int(100.0 / max(h, w)))
        if scale > 1:
            gray = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Optional inversion if digits are dark on light
        white_ratio = np.sum(th > 0) / (th.shape[0]*th.shape[1])
        if white_ratio > 0.7:
            th = cv2.bitwise_not(th)

        config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        try:
            text = pytesseract.image_to_string(th, config=config)
            digits = re.findall(r'\d+', text)
            if digits:
                return int(digits[0])
        except Exception:
            return None

        return None
    
    def determine_sign_type(self, filename):
        """Determina el tipo de señal basado en el nombre del archivo"""
        filename_lower = filename.lower()
        speed_value = None
        
        if 'yield' in filename_lower or 'ceda' in filename_lower: 
            return 'yield', None
        elif 'stop' in filename_lower or 'pare' in filename_lower: 
            return 'stop', None
        elif 'speed' in filename_lower or 'limit' in filename_lower:
            numbers = re.findall(r'\d+', filename)
            if numbers:
                speed_value = numbers[0]
            return 'speed_limit', speed_value
        return None, None

    def detect_shapes(self, image):
        """Detección de formas geométricas en la imagen"""
        shapes_detected = []
        height, width = image.shape[:2]
        
        # Preprocesamiento
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 20, 80)

        # Reuse a preallocated kernel to avoid allocations each frame
        edges = cv2.dilate(edges, self.morph_kernel, iterations=2)
        edges = cv2.erode(edges, self.morph_kernel, iterations=1)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 150 or area > 50000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 15 or h < 15:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            # Use a smaller epsilon to preserve vertices in low-res shapes
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            aspect_ratio = w / h if h > 0 else 0
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Evaluar cada tipo de señal
            shape_scores = {}
            
            # YIELD - Triángulo
            yield_params = self.shape_params['yield']
            if (yield_params['min_vertices'] <= vertices <= yield_params['max_vertices']):
                yield_score = 0.0
                if vertices == 3:
                    # Strong bonus for triangles, but check orientation (yield is inverted triangle)
                    yield_score += 0.6
                    try:
                        ys = approx[:, 0, 1]
                        centroid_y = np.mean(ys)
                        # If the lowest vertex is not clearly below centroid, penalize
                        if (np.max(ys) - centroid_y) < (h * 0.10):
                            yield_score -= 0.35
                    except Exception:
                        pass

                    # Penalize small, cone-like triangles (likely traffic cones)
                    try:
                        roi = image[y:y+h, x:x+w]
                        if roi.size > 0:
                            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                            # orange range for cones
                            orange_lower = np.array([5, 80, 80])
                            orange_upper = np.array([25, 255, 255])
                            orange_mask = cv2.inRange(hsv_roi, orange_lower, orange_upper)
                            orange_ratio = np.sum(orange_mask > 0) / float(max(1, w*h))

                            # red mask for comparison
                            red_lower1 = np.array([0, 40, 40])
                            red_upper1 = np.array([20, 255, 255])
                            red_lower2 = np.array([160, 40, 40])
                            red_upper2 = np.array([180, 255, 255])
                            red_mask1 = cv2.inRange(hsv_roi, red_lower1, red_upper1)
                            red_mask2 = cv2.inRange(hsv_roi, red_lower2, red_upper2)
                            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                            red_ratio = np.sum(red_mask > 0) / float(max(1, w*h))

                            # If the patch is dominated by orange (cone) or very small, heavily penalize
                            if orange_ratio > 0.05 and orange_ratio > red_ratio * 0.6:
                                yield_score -= 0.6
                            if max(w, h) < 24:
                                yield_score -= 0.45
                    except Exception:
                        pass
                elif vertices in [4, 5, 6]:
                    yield_score += 0.4
                else:
                    yield_score += 0.2
                
                if 0.4 <= aspect_ratio <= 1.6:
                    yield_score += 0.3
                elif 0.3 <= aspect_ratio <= 2.0:
                    yield_score += 0.1
                
                if solidity > 0.65:
                    yield_score += 0.3
                elif solidity > 0.5:
                    yield_score += 0.1
                
                area_bonus = min(1.0, area / 800)
                yield_score += area_bonus * 0.2
                shape_scores['yield'] = min(yield_score, 1.0)
            
            # SPEED_LIMIT - Rectángulo/cuadrado
            speed_params = self.shape_params['speed_limit']
            if (speed_params['min_vertices'] <= vertices <= speed_params['max_vertices']):
                speed_score = 0.0
                
                if vertices == 4:
                    speed_score += 0.5
                elif vertices in [5, 6, 7, 8]:
                    speed_score += 0.3
                
                if 0.7 <= aspect_ratio <= 1.3:
                    speed_score += 0.4
                elif 0.4 <= aspect_ratio <= 2.2:
                    speed_score += 0.2
                
                if solidity > 0.75:
                    speed_score += 0.3
                elif solidity > 0.6:
                    speed_score += 0.1
                
                area_bonus = min(1.0, area / 1000)
                speed_score += area_bonus * 0.2
                shape_scores['speed_limit'] = min(speed_score, 1.0)
            
            # STOP - Octágono
            stop_params = self.shape_params['stop']
            if (stop_params['min_vertices'] <= vertices <= stop_params['max_vertices']):
                stop_score = 0.0
                
                if vertices == 8:
                    stop_score += 0.6
                elif vertices in [7, 9, 10, 11, 12]:
                    stop_score += 0.4
                
                if 0.6 <= aspect_ratio <= 1.4:
                    stop_score += 0.4
                elif 0.4 <= aspect_ratio <= 1.8:
                    stop_score += 0.2
                
                if solidity > 0.75:
                    stop_score += 0.3
                elif solidity > 0.6:
                    stop_score += 0.1
                
                area_bonus = min(1.0, area / 1500)
                stop_score += area_bonus * 0.2
                shape_scores['stop'] = min(stop_score, 1.0)
            
            # Seleccionar el mejor tipo
            if shape_scores:
                best_type = max(shape_scores, key=shape_scores.get)
                best_score = shape_scores[best_type]
                
                if best_score > 0.15:
                    area_confidence = min(1.0, area / 800)
                    final_confidence = (best_score * 0.8 + area_confidence * 0.2)
                    
                    expand = 8
                    x_exp = max(0, x - expand)
                    y_exp = max(0, y - expand)
                    w_exp = min(width - x_exp, w + 2*expand)
                    h_exp = min(height - y_exp, h + 2*expand)
                    
                    shapes_detected.append({
                        'type': best_type, 
                        'bbox': (x_exp, y_exp, w_exp, h_exp), 
                        'confidence': final_confidence
                    })
        
        return shapes_detected

    def verify_with_color(self, image, bbox, expected_type):
        """Verificación de colores específicos para cada tipo de señal"""
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0 or w < 8 or h < 8:
            return 0.0
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        if expected_type == 'yield':
            # RANGOS para rojo
            red_lower1 = np.array([0, 40, 40])
            red_upper1 = np.array([20, 255, 255])
            red_lower2 = np.array([160, 40, 40])
            red_upper2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            white_lower = np.array([0, 0, 120])
            white_upper = np.array([180, 80, 255])
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            
            red_ratio = np.sum(red_mask > 0) / (w * h)
            white_ratio = np.sum(white_mask > 0) / (w * h)
            
            if red_ratio > 0.10:
                color_confidence = red_ratio
                if white_ratio > 0.03:
                    color_confidence *= 1.2
                    color_confidence = min(color_confidence, 1.0)
            elif red_ratio > 0.03:
                color_confidence = red_ratio * 0.8
            else:
                color_confidence = 0.0
            
            return min(color_confidence, 1.0)
            
        elif expected_type == 'stop':
            # STOP: Rojo
            red_lower1 = np.array([0, 40, 40])
            red_upper1 = np.array([20, 255, 255])
            red_lower2 = np.array([160, 40, 40])
            red_upper2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            red_ratio = np.sum(red_mask > 0) / (w * h)
            return red_ratio
            
        elif expected_type == 'speed_limit':
            # SPEED_LIMIT: Blanco
            white_mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 100, 255]))
            white_ratio = np.sum(white_mask > 0) / (w * h)
            return white_ratio
        
        return 0.0

    def verify_with_template(self, image, bbox, sign_type):
        """Template matching con plantillas cargadas"""
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return 0.0, None

        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi

        # Upscale very small ROIs to improve template matching / OCR
        MIN_ROI_DIM = 64
        roi_h, roi_w = roi_gray.shape[:2]
        if roi_w < MIN_ROI_DIM or roi_h < MIN_ROI_DIM:
            scale_up = max(MIN_ROI_DIM / float(roi_w), MIN_ROI_DIM / float(roi_h))
            new_w = max(8, int(roi_w * scale_up))
            new_h = max(8, int(roi_h * scale_up))
            roi_gray = cv2.resize(roi_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            roi_h, roi_w = new_h, new_w
        
        best_confidence = 0.0
        best_template_key = None

        if sign_type == 'speed_limit':
            templates_to_try = [(k, t) for k, t in self.base_templates.items() if k.startswith('speed_limit')]
        else:
            templates_to_try = [(k, t) for k, t in self.base_templates.items() if sign_type == k.split('_')[0]]

        for template_key, template_dict in templates_to_try:
            # template_dict: {'bgr','gray','h','w'}
            template_gray = template_dict.get('gray')
            if template_gray is None:
                continue

            t_h, t_w = template_gray.shape[:2]

            # Scale template to fit ROI if necessary (use ROI's current size)
            if t_h > roi_h or t_w > roi_w:
                scale_factor = min(roi_h / float(t_h), roi_w / float(t_w))
                new_h = max(8, int(t_h * scale_factor))
                new_w = max(8, int(t_w * scale_factor))
                resized_template = cv2.resize(template_gray, (new_w, new_h))
            else:
                resized_template = template_gray

            try:
                result = cv2.matchTemplate(roi_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > best_confidence:
                    best_confidence = max_val
                    best_template_key = template_key
            except Exception:
                continue

        return best_confidence, best_template_key

    def image_callback(self, msg):
        """Procesa cada frame de la cámara"""
        # Note: camera frames from Webots are 512x256 (w x h).
        # Typical sign ROIs extracted by downstream logic are very small
        # (about 22x27). The detector upscales tiny input frames to a
        # MIN_WORKING_WIDTH so template matching and OCR have enough
        # resolution to work reliably.
        try:
            current_time = time.time()
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Adaptive working width: only downscale very large frames to limit
            # computation. Do not upscale full frames — instead we upscale tiny
            # ROIs before template matching so small crops (e.g. 22x27) gain
            # sufficient resolution without enlarging the whole image.
            MAX_WORKING_WIDTH = 800
            h_orig, w_orig = cv_image.shape[:2]
            if w_orig > MAX_WORKING_WIDTH:
                new_w = MAX_WORKING_WIDTH
                scale = new_w / float(w_orig)
                new_h = max(1, int(h_orig * scale))
                small_img = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1.0
                small_img = cv_image

            detected_sign, confidence, debug = self.detect_sign_comprehensive(small_img, original_image=cv_image, scale=scale)
            
            if detected_sign:
                time_since_last = current_time - self.last_detection_time
                base_sign_type = detected_sign.split('_')[0]
                cooldown = self.cooldowns.get(base_sign_type, 1.0)
                
                if time_since_last > cooldown or detected_sign != self.last_sign_detected:
                    # Log detection with per-cue confidences when available
                    # Detection logging disabled to reduce runtime noise
                    # if debug:
                    #     s_conf = debug.get('shape', 0.0)
                    #     c_conf = debug.get('color', 0.0)
                    #     t_conf = debug.get('template', 0.0)
                    #     bbox = debug.get('bbox')
                    #     self.get_logger().info(
                    #         f"Señal detectada: {detected_sign} ({confidence:.3f}) "
                    #         f"shape={s_conf:.3f} color={c_conf:.3f} template={t_conf:.3f} bbox={bbox}"
                    #     )
                    # else:
                    #     self.get_logger().info(f"Señal detectada: {detected_sign} ({confidence:.3f})")
                    
                    self.last_sign_detected = detected_sign
                    self.last_detection_time = current_time
                    
                    sign_msg = String()
                    sign_msg.data = detected_sign
                    self.sign_pub.publish(sign_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error procesando imagen: {str(e)}")

    def detect_sign_comprehensive(self, image, original_image=None, scale=1.0):
        """Pipeline principal de detección de señales"""
        best_detection = None
        best_confidence = 0
        best_debug = None
        
        shapes = self.detect_shapes(image)
        
        if not shapes:
            return None, 0, None
        
        for shape in shapes:
            sign_type, bbox = shape['type'], shape['bbox']
            shape_conf = shape['confidence']
            
            weights = self.weights.get(sign_type, {'shape': 0.5, 'color': 0.3, 'template': 0.2})
            thresholds = self.thresholds.get(sign_type, {'shape': 0.15, 'color': 0.2, 'template': 0.15, 'total': 0.35})
            # Primero obtener la confianza por color (rápida) y decidir si
            # merece la pena ejecutar la coincidencia de plantilla (costosa).
            color_conf = self.verify_with_color(image, bbox, sign_type)

            template_conf = 0.0
            template_key = None

            # Ejecutar template-matching sólo si la forma y el color son razonablemente buenos
            # (esto evita lanzar coincidencias de plantilla en objetos blancos/rojos imprecisos)
            if (shape_conf > thresholds['shape'] and color_conf > thresholds['color']) \
               or (shape_conf >= thresholds['shape'] * 2.0):
                template_conf, template_key = self.verify_with_template(image, bbox, sign_type)

            total_conf = (shape_conf * weights['shape'] +
                          color_conf * weights['color'] +
                          template_conf * weights['template'])
            # Log per-candidate confidences to diagnose why detections are suppressed
            # per-candidate logging disabled to reduce console spam
            # try:
            #     self.get_logger().info(
            #         f"Candidate {sign_type} bbox={bbox} "
            #         f"shape={shape_conf:.3f} color={color_conf:.3f} "
            #         f"template={template_conf:.3f} total={total_conf:.3f}"
            #     )
            # except Exception:
            #     pass
            
            shape_ok = shape_conf > thresholds['shape']
            color_ok = color_conf > thresholds['color']
            template_ok = template_conf > thresholds['template']
            total_ok = total_conf > thresholds['total'] and total_conf > best_confidence
            
            # Require both a good shape and color before accepting candidates
            if shape_ok and color_ok and total_ok:
                # For speed limits, accept only when template OR OCR confirms digits
                if sign_type == 'speed_limit':
                    # If original_image provided, map bbox back to original for OCR
                    speed_from_ocr = None
                    if original_image is not None and scale and scale != 1.0:
                        try:
                            x, y, w, h = bbox
                            orig_bbox = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                        except Exception:
                            orig_bbox = bbox
                        speed_from_ocr = self.extract_speed_from_roi(original_image, orig_bbox)
                    else:
                        speed_from_ocr = self.extract_speed_from_roi(image, bbox)

                    if template_ok or speed_from_ocr:
                        best_confidence = total_conf
                        best_debug = {
                            'shape': float(shape_conf),
                            'color': float(color_conf),
                            'template': float(template_conf),
                            # report bbox in original-image coordinates if original_image provided
                            'bbox': (orig_bbox if original_image is not None and scale and scale != 1.0 else bbox),
                            'type': sign_type
                        }

                        if speed_from_ocr:
                            best_detection = f"speed_limit_{speed_from_ocr}"
                        elif template_key and 'speed_limit' in template_key:
                            parts = template_key.split('_')
                            if len(parts) >= 3:
                                speed_num = parts[2]
                                best_detection = f"speed_limit_{speed_num}"
                            else:
                                best_detection = "speed_limit"
                        else:
                            best_detection = "speed_limit"
                else:
                    # For other signs require a decent template match as confirmation
                    if template_ok:
                        best_confidence = total_conf
                        best_debug = {
                            'shape': float(shape_conf),
                            'color': float(color_conf),
                            'template': float(template_conf),
                            'bbox': (bbox if original_image is None or scale == 1.0 else (int(bbox[0]/scale), int(bbox[1]/scale), int(bbox[2]/scale), int(bbox[3]/scale))),
                            'type': sign_type
                        }
                        best_detection = sign_type
        
        return best_detection, best_confidence, best_debug

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
