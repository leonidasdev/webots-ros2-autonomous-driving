#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
import time

class CarController(Node):
    def __init__(self):
        super().__init__('car_controller')
        
        # CONFIGURACIÓN
        self.base_speed = 80.0
        self.current_speed = self.base_speed
        self.max_speed = self.base_speed
        
        self.speed_conversion_factor = 1.0
        
        # Estado de control
        self.steering = 0.0
        self.last_sign = None
        self.last_sign_time = 0
        self.sign_cooldown = 1.5
        
        # Estado de parada (STOP)
        self.is_stopped = False
        self.stop_start_time = 0
        self.stop_duration = 1.0
        
        # Estado de yield
        self.yield_speed_active = False
        
        # Subscriptores
        self.steering_sub = self.create_subscription(
            Float32, '/control/steering', self.steering_callback, 10)
        self.sign_sub = self.create_subscription(
            String, '/traffic_sign', self.sign_callback, 10)
            
        # Publicadores
        self.speed_pub = self.create_publisher(Float32, '/control/speed', 10)
        self.steering_pub = self.create_publisher(Float32, '/control/steering', 10)
        
        # Timer para loop de control
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Car Controller iniciado")
        
    def steering_callback(self, msg):
        """Recibe dirección del road follower"""
        self.steering = msg.data
        
    def sign_callback(self, msg):
        """Procesa señales de tráfico del sign_detector"""
        current_time = time.time()
        sign_data = msg.data
        
        # Verificar cooldown
        if current_time - self.last_sign_time < self.sign_cooldown:
            return
        
        # Solo procesar si es una señal nueva
        if sign_data != self.last_sign:
            self.last_sign = sign_data
            self.last_sign_time = current_time
            
            # Procesar la señal según tipo
            self.handle_traffic_sign()
            
    def handle_traffic_sign(self):
        """Maneja diferentes tipos de señales"""
        
        if self.last_sign == 'yield':
            self.handle_yield()
            
        elif self.last_sign == 'stop':
            self.handle_stop()
            
        elif self.last_sign.startswith('speed_limit'):
            self.handle_speed_limit()
            
    def handle_yield(self):
        """Maneja señal YIELD - mitad de velocidad máxima"""
        # Activar modo yield
        self.yield_speed_active = True
        # Establecer velocidad máxima a la mitad
        self.max_speed = self.base_speed / 2.0
        
    def handle_stop(self):
        """Maneja señal STOP - parada completa por 1 segundo"""
        # Parar completamente
        self.current_speed = 0.0
        self.is_stopped = True
        self.stop_start_time = time.time()
        
        # Desactivar yield temporalmente
        self.yield_speed_active = False
        
    def handle_speed_limit(self):
        """Maneja señal de velocidad máxima"""
        try:
            # Extraer número de velocidad
            parts = self.last_sign.split('_')
            speed_number = int(parts[-1])
            
            self.max_speed = speed_number * self.speed_conversion_factor
            
            # Limitar valores razonables
            if self.max_speed < 10:
                self.max_speed = 10
            elif self.max_speed > 100:
                self.max_speed = 100
                
            # Si yield está activo, la velocidad máxima debe ser la mitad del límite
            if self.yield_speed_active:
                self.max_speed = self.max_speed / 2.0
                
            # Si no estamos parados, ajustar velocidad actual
            if not self.is_stopped:
                self.current_speed = min(self.current_speed, self.max_speed)
                
        except (ValueError, IndexError):
            self.max_speed = self.base_speed
            
    def check_stop_timer(self):
        """Verifica si ha pasado el tiempo de STOP"""
        if self.is_stopped and (time.time() - self.stop_start_time) >= self.stop_duration:
            self.resume_after_stop()
        
    def resume_after_stop(self):
        """Reanuda la marcha después de STOP"""
        self.is_stopped = False
        
        # Determinar velocidad máxima después del stop
        if self.yield_speed_active:
            self.max_speed = self.base_speed / 2.0
        elif self.last_sign and self.last_sign.startswith('speed_limit'):
            self.handle_speed_limit()
        else:
            self.max_speed = self.base_speed
            
        self.current_speed = self.max_speed
        
    def control_loop(self):
        """Loop principal de control"""
        
        # Verificar timer de STOP
        if self.is_stopped:
            self.check_stop_timer()
        
        if not self.is_stopped:
            # Ajuste gradual de velocidad hacia max_speed
            if self.current_speed > self.max_speed:
                self.current_speed = max(self.max_speed, self.current_speed - 1.0)
            elif self.current_speed < self.max_speed:
                self.current_speed = min(self.max_speed, self.current_speed + 1.0)
            
            # Publicar velocidad
            speed_msg = Float32()
            speed_msg.data = self.current_speed
            self.speed_pub.publish(speed_msg)
            
            # Publicar dirección
            steering_msg = Float32()
            steering_msg.data = self.steering
            self.steering_pub.publish(steering_msg)
        else:
            # En STOP: velocidad cero pero mantener dirección
            speed_msg = Float32()
            speed_msg.data = 0.0
            self.speed_pub.publish(speed_msg)
            
            # Mantener dirección para cuando reanude
            steering_msg = Float32()
            steering_msg.data = self.steering
            self.steering_pub.publish(steering_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CarController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
