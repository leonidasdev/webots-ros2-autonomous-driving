#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32
import cv2
import time

class RoadFollower(Node):
    def __init__(self):
        super().__init__('road_follower')
        self.bridge = CvBridge()
        
        # Subscriber para road_camera
        self.image_sub = self.create_subscription(
            Image,
            '/road_camera/image',
            self.image_callback,
            10
        )
        
        # Publisher para dirección
        self.steering_pub = self.create_publisher(Float32, '/control/steering', 10)
        
        # Sistema de memoria
        self.last_known_road_center = 256  # Centro de la imagen (512/2)
        
        # Historial para suavizado (reducido para respuesta más rápida ~2x)
        self.road_center_history = []
        self.history_size = 4
        
        # Modo cruce de peatones
        self.crosswalk_mode = False
        self.crosswalk_start_time = 0
        # Tiempo de bloqueo cuando se detectaba un crosswalk.
        # Reducido 5x para reaccionar más rápido al tráfico.
        # Antes: 3.0s -> Ahora: 0.6s
        self.crosswalk_lock_time = 0.6  # segundos
        
        # Umbrales de detección
        self.yellow_threshold = 0.30  # >30% amarillo = crosswalk
        self.thin_line_threshold = 0.15  # <15% amarillo = línea fina
        
        # Contadores para estabilidad (reducción para transiciones más rápidas)
        self.consecutive_yellow_frames = 0
        self.min_yellow_frames = 1
        self.consecutive_thin_line_frames = 0
        self.min_thin_line_frames = 1
        
        # Historial para filtrado (más reactivo)
        self.yellow_ratio_history = []
        self.yellow_history_size = 3
        
        # Controlador PID
        self.Kp = 0.005
        self.Ki = 0.0003
        self.Kd = 0.0002
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.integral_max = 100.0
        self.max_steering = 0.25
        
        # Filtros de suavizado reducidos para mayor reactividad (~2x)
        self.error_history = []
        self.error_history_size = 3
        self.steering_history = []
        self.steering_history_size = 3
        self.center_history = []
        self.center_history_size = 5
        self.error_trend = []
        self.error_trend_size = 5
        
        # Estado de curva
        self.in_curve = False
        self.curve_direction = 0
        
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            current_time = time.time()
            
            if cv_image.shape[0] == 16 and cv_image.shape[1] == 512:
                # Detectar porcentaje de amarillo
                yellow_ratio = self.detect_yellow_ratio(cv_image)
                
                # Filtrado con historial
                self.yellow_ratio_history.append(yellow_ratio)
                if len(self.yellow_ratio_history) > self.yellow_history_size:
                    self.yellow_ratio_history.pop(0)
                avg_yellow_ratio = np.mean(self.yellow_ratio_history) if self.yellow_ratio_history else yellow_ratio
                
                # Determinar tipo de detección
                is_crosswalk = avg_yellow_ratio > self.yellow_threshold  # >30%
                is_thin_line = self.is_thin_yellow_line(cv_image, avg_yellow_ratio)
                
                # Lógica principal
                if not self.crosswalk_mode:
                    # Modo normal
                    if is_crosswalk:
                        self.consecutive_yellow_frames += 1
                        if self.consecutive_yellow_frames >= self.min_yellow_frames:
                            # Activar modo crosswalk
                            self.crosswalk_mode = True
                            self.crosswalk_start_time = current_time
                            self.consecutive_yellow_frames = 0
                            self.consecutive_thin_line_frames = 0
                            self.reset_pid()
                    else:
                        self.consecutive_yellow_frames = 0
                    
                    # Seguir carretera normal
                    road_center, confidence = self.detect_road_center(cv_image)
                    
                    if confidence:
                        self.add_to_history(road_center)
                        self.update_curve_state(road_center)
                    else:
                        if self.road_center_history:
                            road_center = int(np.median(self.road_center_history))
                        else:
                            road_center = self.last_known_road_center
                    
                    smoothed_center = self.smooth_road_center(road_center)
                    self.last_known_road_center = smoothed_center
                    steering = self.calculate_pid(smoothed_center, confidence)
                    
                else:
                    # Modo crosswalk activado
                    time_in_crosswalk = current_time - self.crosswalk_start_time
                    
                    # Fase 1: 3 segundos de lock
                    if time_in_crosswalk < self.crosswalk_lock_time:
                        steering = 0.0  # Ir recto
                    
                    # Fase 2: Buscar línea fina
                    else:
                        steering = 0.0  # Seguir recto
                        
                        if is_thin_line:
                            self.consecutive_thin_line_frames += 1
                            if self.consecutive_thin_line_frames >= self.min_thin_line_frames:
                                # Volver a modo normal
                                self.crosswalk_mode = False
                                self.consecutive_thin_line_frames = 0
                                self.reset_pid()
                        else:
                            self.consecutive_thin_line_frames = 0
                
                # Suavizar y publicar steering
                if self.crosswalk_mode:
                    smoothed_steering = steering
                else:
                    smoothed_steering = self.smooth_steering(steering)
                
                steering_msg = Float32()
                steering_msg.data = smoothed_steering
                self.steering_pub.publish(steering_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error procesando imagen: {e}")
    
    def detect_yellow_ratio(self, image):
        """Detecta el ratio de píxeles amarillos usando múltiples rangos HSV"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Rangos HSV para amarillo
            yellow_lower1 = np.array([20, 70, 70])
            yellow_upper1 = np.array([30, 255, 255])
            mask1 = cv2.inRange(hsv, yellow_lower1, yellow_upper1)
            
            yellow_lower2 = np.array([15, 70, 70])
            yellow_upper2 = np.array([35, 255, 255])
            mask2 = cv2.inRange(hsv, yellow_lower2, yellow_upper2)
            
            # Combinar máscaras
            yellow_mask = cv2.bitwise_or(mask1, mask2)
            
            # Limpiar ruido
            kernel = np.ones((3, 3), np.uint8)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
            
            # Calcular ratio
            total_pixels = image.shape[0] * image.shape[1]
            yellow_pixels = np.sum(yellow_mask > 0)
            
            return yellow_pixels / total_pixels
            
        except Exception:
            return 0.0
    
    def is_thin_yellow_line(self, image, yellow_ratio):
        """Detecta si hay una línea fina amarilla para salir de modo crosswalk"""
        if yellow_ratio > self.thin_line_threshold or yellow_ratio < 0.05:
            return False
        
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            yellow_lower = np.array([20, 70, 70])
            yellow_upper = np.array([35, 255, 255])
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            
            height, width = image.shape[:2]
            column_sums = np.sum(yellow_mask, axis=0) / 255.0
            
            # Encontrar columnas con amarillo
            significant_columns = np.where(column_sums > height * 0.2)[0]
            
            if len(significant_columns) == 0:
                return False
            
            # Agrupar columnas contiguas
            groups = []
            current_group = [significant_columns[0]]
            
            for i in range(1, len(significant_columns)):
                if significant_columns[i] - significant_columns[i-1] <= 3:
                    current_group.append(significant_columns[i])
                else:
                    if current_group:
                        groups.append(current_group)
                    current_group = [significant_columns[i]]
            
            if current_group:
                groups.append(current_group)
            
            # Verificar si hay líneas delgadas
            for group in groups:
                group_width = len(group)
                
                # Línea fina: 2-8 píxeles de ancho
                if 2 <= group_width <= 8:
                    group_start = max(0, min(group) - 2)
                    group_end = min(width-1, max(group) + 2)
                    
                    roi = yellow_mask[:, group_start:group_end]
                    roi_height_ratio = np.sum(roi > 0) / (roi.shape[0] * roi.shape[1])
                    
                    # Línea fina no llena toda la altura
                    if roi_height_ratio < 0.6:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def detect_road_center(self, image):
        """Detección del centro de la carretera basado en intensidad"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            column_means = np.mean(blurred, axis=0)
            column_means_smooth = np.convolve(column_means, np.ones(5)/5, mode='same')
            
            # Buscar pico más brillante
            max_idx = np.argmax(column_means_smooth)
            max_val = column_means_smooth[max_idx]
            
            # Verificar si es un pico significativo
            if max_val > 30:
                window = 40
                start = max(0, max_idx - window)
                end = min(511, max_idx + window)
                
                if end > start:
                    local_avg = np.mean(column_means_smooth[start:end])
                    if max_val > local_avg * 1.15:
                        return max_idx, True
            
            # Búsqueda en área prioritaria
            search_center = self.last_known_road_center
            search_window = 180
            start = max(0, search_center - search_window//2)
            end = min(511, search_center + search_window//2)
            
            if end > start:
                local_max = np.argmax(column_means_smooth[start:end]) + start
                if column_means_smooth[local_max] > 20:
                    return local_max, False
            
            return self.last_known_road_center, False
                
        except Exception:
            return self.last_known_road_center, False
    
    def update_curve_state(self, road_center):
        """Detección de estado de curva basado en error de posición"""
        image_center = 256
        error = road_center - image_center
        
        self.error_trend.append(error)
        if len(self.error_trend) > self.error_trend_size:
            self.error_trend.pop(0)
        
        error_magnitude = np.mean(np.abs(self.error_trend)) if self.error_trend else 0
        curve_threshold = 40 + error_magnitude * 0.5
        
        if abs(error) > curve_threshold:
            if not self.in_curve:
                self.in_curve = True
                self.curve_direction = 1 if error > 0 else -1
        else:
            if self.in_curve:
                self.in_curve = False
    
    def calculate_pid(self, road_center, confidence):
        """Controlador PID para seguimiento de carretera"""
        if road_center == 256 and not confidence:
            return 0.0
        
        image_center = 256
        current_time = time.time()
        dt = max(0.001, min(current_time - self.prev_time, 0.1))
        
        # Calcular error y filtrar
        error = road_center - image_center
        self.error_history.append(error)
        if len(self.error_history) > self.error_history_size:
            self.error_history.pop(0)
        filtered_error = np.mean(self.error_history) if self.error_history else error
        
        # Término Proporcional
        P = self.Kp * filtered_error
        
        # Término Integral
        integral_gain = 0.3 if not confidence else 1.0
        self.integral += filtered_error * dt * integral_gain
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        I = self.Ki * self.integral
        
        # Término Derivativo
        if dt > 0.01 and len(self.error_history) >= 2:
            derivative = (filtered_error - self.prev_error) / dt
            derivative = np.clip(derivative, -15, 15)
            D = self.Kd * derivative
        else:
            D = 0.0
        
        # Calcular steering total
        steering = P + I + D
        
        # Reducir agresividad sin confianza
        if not confidence:
            steering *= 0.5
        
        # Aplicar límites
        steering = max(-self.max_steering, min(self.max_steering, steering))
        
        # Actualizar estado
        self.prev_error = filtered_error
        self.prev_time = current_time
        
        return steering
    
    def reset_pid(self):
        """Resetea el estado del controlador PID"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.error_history = []
        self.error_trend = []
        self.in_curve = False
    
    def smooth_road_center(self, center):
        """Suavizado del centro detectado"""
        self.center_history.append(center)
        if len(self.center_history) > self.center_history_size:
            self.center_history.pop(0)
        
        if len(self.center_history) >= 3:
            return int(np.mean(self.center_history[-3:]))
        else:
            return center
    
    def smooth_steering(self, steering):
        """Suavizado del comando de dirección"""
        self.steering_history.append(steering)
        if len(self.steering_history) > self.steering_history_size:
            self.steering_history.pop(0)
        
        if len(self.steering_history) >= 3:
            return np.mean(self.steering_history[-3:])
        else:
            return steering
    
    def add_to_history(self, road_center):
        """Mantener historial de centros detectados"""
        self.road_center_history.append(road_center)
        if len(self.road_center_history) > self.history_size:
            self.road_center_history.pop(0)

def main(args=None):
    rclpy.init(args=args)
    node = RoadFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
