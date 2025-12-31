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
            'speed_limit': {'shape': 0.15, 'color': 0.25, 'template': 0.20, 'total': 0.35}
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
            
            # Intentar cargar de resources_augmented o resources normal
            augmented_path = os.path.join(package_share_dir, 'resources_augmented')
            normal_path = os.path.join(package_share_dir, 'resources')
            
            if os.path.exists(augmented_path):
                target_path = augmented_path
            else:
                target_path = normal_path
            
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
                    template = cv2.imread(filepath)
                    
                    if template is not None:
                        if sign_type == 'speed_limit' and speed_value:
                            template_key = f"speed_limit_{speed_value}_{filename}"
                            templates[template_key] = template
                        else:
                            template_key = f"{sign_type}_{filename}"
                            templates[template_key] = template
                    
        except Exception as e:
            self.get_logger().error(f"Error cargando plantillas: {str(e)}")
        
        return templates
    
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
        
        kernel = np.ones((5,5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
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
            epsilon = 0.05 * perimeter
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
                    yield_score += 0.6
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
        if roi.size == 0 or w < 8 or h < 8:
            return 0.0, None
        
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        best_confidence = 0.0
        best_template_key = None
        
        if sign_type == 'speed_limit':
            templates_to_try = [(k, t) for k, t in self.base_templates.items() 
                               if 'speed_limit' in k]
        else:
            templates_to_try = [(k, t) for k, t in self.base_templates.items() 
                               if sign_type == k.split('_')[0]]
        
        for template_key, template in templates_to_try:
            if len(template.shape) == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template
            
            if template_gray.shape[0] > h or template_gray.shape[1] > w:
                scale_factor = min(h / template_gray.shape[0], w / template_gray.shape[1])
                new_h = int(template_gray.shape[0] * scale_factor)
                new_w = int(template_gray.shape[1] * scale_factor)
                if new_h < 8 or new_w < 8:
                    continue
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
        try:
            current_time = time.time()
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            detected_sign, confidence = self.detect_sign_comprehensive(cv_image)
            
            if detected_sign:
                time_since_last = current_time - self.last_detection_time
                base_sign_type = detected_sign.split('_')[0]
                cooldown = self.cooldowns.get(base_sign_type, 1.0)
                
                if time_since_last > cooldown or detected_sign != self.last_sign_detected:
                    self.get_logger().info(f"Señal detectada: {detected_sign} ({confidence:.3f})")
                    
                    self.last_sign_detected = detected_sign
                    self.last_detection_time = current_time
                    
                    sign_msg = String()
                    sign_msg.data = detected_sign
                    self.sign_pub.publish(sign_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error procesando imagen: {str(e)}")

    def detect_sign_comprehensive(self, image):
        """Pipeline principal de detección de señales"""
        best_detection = None
        best_confidence = 0
        
        shapes = self.detect_shapes(image)
        
        if not shapes:
            return None, 0
        
        for shape in shapes:
            sign_type, bbox = shape['type'], shape['bbox']
            shape_conf = shape['confidence']
            
            weights = self.weights.get(sign_type, {'shape': 0.5, 'color': 0.3, 'template': 0.2})
            thresholds = self.thresholds.get(sign_type, {'shape': 0.15, 'color': 0.2, 'template': 0.15, 'total': 0.35})
            
            color_conf = self.verify_with_color(image, bbox, sign_type)
            template_conf, template_key = self.verify_with_template(image, bbox, sign_type)
            
            total_conf = (shape_conf * weights['shape'] + 
                         color_conf * weights['color'] + 
                         template_conf * weights['template'])
            
            shape_ok = shape_conf > thresholds['shape']
            color_ok = color_conf > thresholds['color']
            template_ok = template_conf > thresholds['template']
            total_ok = total_conf > thresholds['total'] and total_conf > best_confidence
            
            if shape_ok and total_ok:
                if shape_conf > 0.4 and color_conf > thresholds['color'] * 0.7:
                    color_ok = True
                
                if shape_ok and color_ok and template_ok and total_ok:
                    best_confidence = total_conf
                    
                    if sign_type == 'speed_limit' and template_key and 'speed_limit' in template_key:
                        parts = template_key.split('_')
                        if len(parts) >= 3:
                            speed_num = parts[2]
                            best_detection = f"speed_limit_{speed_num}"
                        else:
                            best_detection = "speed_limit"
                    else:
                        best_detection = sign_type
        
        return best_detection, best_confidence

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
