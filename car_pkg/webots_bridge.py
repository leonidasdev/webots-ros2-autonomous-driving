#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from controller import Robot
import sys
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Float32

class WebotsBridge(Node):
    def __init__(self):
        super().__init__('webots_bridge')
        
        self.get_logger().info("Iniciando conexión con Webots...")
        
        # Conexión con Webots
        try:
            self.robot = Robot()
            self.timestep = int(self.robot.getBasicTimeStep())
            self.get_logger().info("Conectado a Webots correctamente")
        except Exception as e:
            self.get_logger().error(f"Error conectando a Webots: {e}")
            sys.exit(1)
            
        # Obtener dispositivos de camara
        try:
            self.car_camera = self.robot.getDevice('car_camera')
            self.road_camera = self.robot.getDevice('road_camera')
            self.get_logger().info("Camaras encontradas")
        except:
            self.get_logger().error("No se pudieron encontrar las camaras")
            sys.exit(1)
        
        # Solo los motores que SABEMOS que existen
        self.motors = {}
        motor_names = [
            'right_steer', 
            'left_steer', 
            'right_rear_wheel', 
            'left_rear_wheel'
        ]
        
        for name in motor_names:
            try:
                motor = self.robot.getDevice(name)
                self.motors[name] = motor
                if 'wheel' in name:
                    motor.setPosition(float('inf'))
                    motor.setVelocity(0.0)
                self.get_logger().info(f"Motor configurado: {name}")
            except Exception as e:
                self.get_logger().error(f"Error configurando motor {name}: {e}")
                sys.exit(1)
        
        # Habilitar dispositivos
        self.car_camera.enable(self.timestep)
        self.road_camera.enable(self.timestep)
        
        self.bridge = CvBridge()
        self.current_speed = 0.0
        self.current_steering = 0.0
        
        # Publishers para imágenes
        self.car_camera_pub = self.create_publisher(Image, '/car_camera/image', 10)
        self.road_camera_pub = self.create_publisher(Image, '/road_camera/image', 10)
        
        # Subscribers para control
        self.speed_sub = self.create_subscription(
            Float32, '/control/speed', self.speed_callback, 10)
        self.steering_sub = self.create_subscription(
            Float32, '/control/steering', self.steering_callback, 10)
        
        # Timer para publicar datos — use el basicTimeStep de Webots (ms -> s)
        self.timer = self.create_timer(self.timestep / 1000.0, self.publish_data)
        
        self.get_logger().info("Webots Bridge listo!")
        
    def speed_callback(self, msg):
        self.current_speed = max(0.0, min(msg.data, 10.0))
        # Aplicar velocidad solo a las ruedas traseras
        self.motors['right_rear_wheel'].setVelocity(self.current_speed)
        self.motors['left_rear_wheel'].setVelocity(self.current_speed)
        
    def steering_callback(self, msg):
        self.current_steering = max(-0.5, min(msg.data, 0.5))
        # Aplicar dirección a ambos motores de dirección
        self.motors['right_steer'].setPosition(self.current_steering)
        self.motors['left_steer'].setPosition(self.current_steering)
        
    def publish_data(self):
        if self.robot.step(self.timestep) != -1:
            try:
                # Publicar imagen de car_camera
                car_image_data = self.car_camera.getImage()
                if car_image_data:
                    car_image = np.frombuffer(car_image_data, np.uint8).reshape(
                        (self.car_camera.getHeight(), self.car_camera.getWidth(), 4))
                    car_image_rgb = cv2.cvtColor(car_image, cv2.COLOR_BGRA2BGR)
                    car_msg = self.bridge.cv2_to_imgmsg(car_image_rgb, "bgr8")
                    self.car_camera_pub.publish(car_msg)
                
                # Publicar imagen de road_camera
                road_image_data = self.road_camera.getImage()
                if road_image_data:
                    road_image = np.frombuffer(road_image_data, np.uint8).reshape(
                        (self.road_camera.getHeight(), self.road_camera.getWidth(), 4))
                    road_image_rgb = cv2.cvtColor(road_image, cv2.COLOR_BGRA2BGR)
                    road_msg = self.bridge.cv2_to_imgmsg(road_image_rgb, "bgr8")
                    self.road_camera_pub.publish(road_msg)
            except Exception as e:
                self.get_logger().error(f"Error publicando imagenes: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = WebotsBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
