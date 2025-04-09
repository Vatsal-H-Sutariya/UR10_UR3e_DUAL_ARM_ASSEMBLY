#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker
import math
import time
import numpy as np
import tf2_ros
from tf_transformations import quaternion_from_euler, quaternion_from_matrix, quaternion_about_axis

class Cylinder:
    def __init__(self, position, direction, length, diameter, robot_type, link_id):
        self.position = np.array(position)  # Center point
        self.direction = np.array(direction)  # Unit vector
        self.length = length
        self.radius = diameter / 2
        self.robot_type = robot_type
        self.link_id = link_id

class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_mover')

        # Publishers
        self.ur10e_pub = self.create_publisher(JointState, '/ur10e/joint_states', 10)
        self.ur3_pub = self.create_publisher(JointState, '/ur3/joint_states', 10)
        self.marker_pub = self.create_publisher(Marker, '/robot_link_markers', 10)

        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Joint names
        self.ur10e_joint_names = [
            'ur10e_shoulder_pan_joint',
            'ur10e_shoulder_lift_joint',
            'ur10e_elbow_joint',
            'ur10e_wrist_1_joint',
            'ur10e_wrist_2_joint',
            'ur10e_wrist_3_joint'
        ]
        self.ur3_joint_names = [
            'ur3_shoulder_pan_joint',
            'ur3_shoulder_lift_joint',
            'ur3_elbow_joint',
            'ur3_wrist_1_joint',
            'ur3_wrist_2_joint',
            'ur3_wrist_3_joint'
        ]

        # Store current positions
        self.ur10e_current_positions = [0.0, -math.pi/4, math.pi/4, 0.0, 0.0, 0.0]
        self.ur3_current_positions = [0.0, -math.pi/4, 0.0, 0.0, 0.0, 0.0]
        
        
        
        # DH parameters [a, alpha, d]
        self.ur10e_dh = [
            [0.0, math.pi/2, 0.1807],
            [-0.6127, 0, 0.0],
            [-0.57155, 0.0, 0.0],
            [0.0, math.pi/2, 0.17415],
            [0.0, -math.pi/2, 0.11985],
            [0.0, 0.0, 0.11655]
        ]
        self.ur3_dh = [
            [0.0, math.pi/2, 0.1519],
            [   -0.24365, 0.0, 0.0],
            [-0.21325, 0.0, 0.0],
            [0.0, math.pi/2,    0.11235],
            [0.0, -math.pi/2,0.08535],
            [0.0, 0.0, 0.0819]
        ]

        # Approximate link diameters (in meters)
        self.base_to_origin_ur10e = self.get_transform( math.pi, 0,0,0)
        self.base_to_origin_ur3 = self.get_transform( math.pi, 1,0,0)
        self.ur10e_base_offset = np.array([0.0, 0.0, 0.0])  # UR10e at origin
        self.ur3_base_offset = np.array([1.0, 0.0, 0.0])   # UR3 1m to the right
        self.ur10e_diameters = [0.15, 0.12, 0.10, 0.08, 0.08, 0.06]
        self.ur3_diameters = [0.12, 0.10, 0.08, 0.06, 0.06, 0.05]
        self.second_link_offset_ur10e = 0.2
        self.second_link_offset_ur3 = 0.1
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.cylinders = []  # List to store all cylinders
        self.start_time = time.time()

        self.get_logger().info('Robot Mover node started')

    def get_transform(self, theta, a, alpha, d):
        return np.array([
            [math.cos(theta), -math.sin(theta)*math.cos(alpha), math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
            [math.sin(theta), math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
            [0.0, math.sin(alpha), math.cos(alpha), d],
            [0.0, 0.0, 0.0, 1.0]
        ])

    def get_link_poses(self, robot_type: str):
        """Calculate position and orientation of each link"""
        robot_type = robot_type.lower()
        if robot_type == 'ur10e':
            joints = self.ur10e_current_positions
            dh_params = self.ur10e_dh
            diameters = self.ur10e_diameters

        elif robot_type == 'ur3':
            joints = self.ur3_current_positions
            dh_params = self.ur3_dh
            diameters = self.ur3_diameters
        else:
            return []

        poses = []
        T = np.eye(4)
        if robot_type == 'ur10e':
            T = self.base_to_origin_ur10e
        elif robot_type == 'ur3':
            T = self.base_to_origin_ur3
        prev_pos = np.array(T[:3, 3])
                
        T_third = np.eye(4)
        for j in range(3):  # Up to third joint (elbow_joint)
            T_third = T_third @ self.get_transform(joints[j], dh_params[j][0], dh_params[j][1], dh_params[j][2])
        z_axis = T_third[:3, 2]  # Third column is z-axis
        z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 1e-6 else np.array([0, 0, 1])
        
        for i in range(6):
            
            T = T @ self.get_transform(joints[i], dh_params[i][0], dh_params[i][1], dh_params[i][2])            
            curr_pos = T[:3, 3]
            # Store position (midpoint) and orientation
            midpoint = (prev_pos + curr_pos) / 2
            length = np.linalg.norm(curr_pos - prev_pos)

            direction = (curr_pos - prev_pos) / length if length > 0 else np.array([0, 0, 1])
            if i == 1:
                if robot_type == 'ur10e':
                    offset_vector = z_axis * self.second_link_offset_ur10e
                elif robot_type == 'ur3':
                    offset_vector = z_axis * self.second_link_offset_ur3
                else:
                    return []
                midpoint = midpoint - offset_vector
            poses.append((midpoint.tolist(), direction.tolist() , length, diameters[i]))
            prev_pos = curr_pos.copy()
        
        return poses

    def update_cylinder_list(self):
        """Update the list of cylinders for both robots"""
        self.cylinders = []
        
        # Get poses for both robots
        ur10e_poses = self.get_link_poses('ur10e')
        ur3_poses = self.get_link_poses('ur3')
        
        # Add UR10e cylinders
        for i, (pos, direction, length, diameter) in enumerate(ur10e_poses):
            self.cylinders.append(Cylinder(pos, direction, length, diameter, 'ur10e', i))
            
        # Add UR3 cylinders
        for i, (pos, direction, length, diameter) in enumerate(ur3_poses):
            self.cylinders.append(Cylinder(pos, direction, length, diameter, 'ur3', i))    

    def check_collisions(self):
        """Check for collisions between all cylinders"""
        self.update_cylinder_list()
        collisions = []
        
        for i, cyl1 in enumerate(self.cylinders):
            for j, cyl2 in enumerate(self.cylinders[i+1:]):
                if self.cylinder_collision(cyl1, cyl2):
                    collisions.append((
                        f"{cyl1.robot_type}_link_{cyl1.link_id}",
                        f"{cyl2.robot_type}_link_{cyl2.link_id}"
                    ))
                    self.get_logger().warn(
                        f"Collision detected between {cyl1.robot_type}_link_{cyl1.link_id} "
                        f"and {cyl2.robot_type}_link_{cyl2.link_id}"
                    )
        
        return collisions

    def publish_markers(self, robot_type: str):
        """Publish cylinder markers for each link"""
        poses = self.get_link_poses(robot_type)
        if not poses:
            return

        collisions = self.check_collisions()
        collision_links = set()
        for link1, link2 in collisions:
            collision_links.add(link1)
            collision_links.add(link2)

        for i, (pos, direction, length, diameter) in enumerate(poses):
            marker = Marker()
            marker.header.frame_id = f"world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"{robot_type}_links"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position at midpoint
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            
            # Convert rotation matrix to quaternion
            default_axis = np.array([0, 0, 1])  # Default cylinder orientation (along z-axis)
            axis = np.cross(default_axis, direction)
            angle = math.acos(np.dot(default_axis, direction))
            if np.linalg.norm(axis) > 1e-6:  # Avoid division by zero
                axis = axis / np.linalg.norm(axis)
                q = quaternion_about_axis(angle, axis)
            else:
                q = [0, 0, 0, 1] if angle < 1e-6 else [0, 0, 1, 0]  # Identity or 180-degree rotation
            
            marker.pose.orientation.x = float(q[0])
            marker.pose.orientation.y = float(q[1])
            marker.pose.orientation.z = float(q[2])
            marker.pose.orientation.w = float(q[3])


            # Scale: x,y for diameter, z for length
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = length
            
            # Color
            link_name = f"{robot_type}_link_{i}"
            if link_name in collision_links:
                # Purple for collision
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            else:
                # Normal colors
                marker.color.r = 1.0 if robot_type == 'ur10e' else 0.0
                marker.color.g = 0.0 if robot_type == 'ur10e' else 1.0
                marker.color.b = 0.0
            marker.color.a = 0.8
            
            self.marker_pub.publish(marker)
            
            self.marker_pub.publish(marker)



    def cylinder_collision(self, cyl1: Cylinder, cyl2: Cylinder) -> bool:
        """Check collision between two cylinders"""
        # Don't check collision between adjacent links of same robot
        if cyl1.robot_type == cyl2.robot_type and abs(cyl1.link_id - cyl2.link_id) <= 1:
            return False

        # Vector between cylinder centers
        v = cyl2.position - cyl1.position
        
        # Direction vectors
        d1, d2 = cyl1.direction, cyl2.direction
        
        # Calculate closest points parameters using line segment distance
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, v)
        e = np.dot(d2, v)
        
        denom = a * c - b * b
        
        # If parallel, use simpler calculation
        if abs(denom) < 1e-6:
            t = 0.0
            s = d / a if a > 1e-6 else 0.0
        else:
            s = (b * e - c * d) / denom
            t = (a * e - b * d) / denom
        
        # Clamp to segment lengths
        s = max(-cyl1.length/2, min(cyl1.length/2, s))
        t = max(-cyl2.length/2, min(cyl2.length/2, t))
        
        # Closest points on each cylinder axis
        p1 = cyl1.position + s * d1
        p2 = cyl2.position + t * d2
        
        # Distance between closest points
        distance = np.linalg.norm(p2 - p1)
        
        # Collision if distance less than sum of radii
        return distance < (cyl1.radius + cyl2.radius)        

    def publish_tf(self, robot_type: str):
        position, orientation = self.get_end_effector_pose(robot_type)
        if position is None or orientation is None:
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = f"world"
        t.child_frame_id = f"{robot_type}_ee_link"
        
        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        
        q = quaternion_from_euler(orientation[0], orientation[1], orientation[2])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def get_end_effector_pose(self, robot_type: str):
        robot_type = robot_type.lower()
        if robot_type == 'ur10e':
            joints = self.ur10e_current_positions
            dh_params = self.ur10e_dh
        elif robot_type == 'ur3':
            joints = self.ur3_current_positions
            dh_params = self.ur3_dh
        else:
            self.get_logger().error('Invalid robot_type')
            return None, None

        T = np.eye(4)
        if robot_type == 'ur10e':
            T = self.base_to_origin_ur10e
        elif robot_type == 'ur3':
            T = self.base_to_origin_ur3
        
        for i in range(6):
            T = T @ self.get_transform(joints[i], dh_params[i][0], dh_params[i][1], dh_params[i][2])
        position = T[:3, 3]
        roll = math.atan2(T[2,1], T[2,2])
        pitch = math.atan2(-T[2,0], math.sqrt(T[2,1]**2 + T[2,2]**2))
        yaw = math.atan2(T[1,0], T[0,0])

        return position.tolist(), [roll, pitch, yaw]

    def move_robot(self, robot_type: str, joint_positions: list, oscillate: bool = False):
        if len(joint_positions) != 6:
            self.get_logger().error('Joint positions must contain exactly 6 values')
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        robot_type = robot_type.lower()

        if robot_type == 'ur10e':
            msg.name = self.ur10e_joint_names
            msg.position = joint_positions
            self.ur10e_pub.publish(msg)
            self.ur10e_current_positions = joint_positions.copy()
            self.ur10e_oscillate = oscillate
        
        elif robot_type == 'ur3':
            msg.name = self.ur3_joint_names
            msg.position = joint_positions
            self.ur3_pub.publish(msg)
            self.ur3_current_positions = joint_positions.copy()
            self.ur3_oscillate = oscillate
        
        else:
            self.get_logger().error('Invalid robot_type')
            return

        self.publish_tf(robot_type)
        self.publish_markers(robot_type)

    def timer_callback(self):
        current_time = time.time() - self.start_time
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        # UR10e
        ur10e_positions = self.ur10e_current_positions.copy()
        
        msg.name = self.ur10e_joint_names
        msg.position = ur10e_positions
        self.ur10e_pub.publish(msg)
        self.publish_tf('ur10e')
        self.publish_markers('ur10e')

        # UR3
        ur3_positions = self.ur3_current_positions.copy()
        
        msg.name = self.ur3_joint_names
        msg.position = ur3_positions
        self.ur3_pub.publish(msg)
        self.publish_tf('ur3')
        self.publish_markers('ur3')
        self.check_collisions()

def main(args=None):
    rclpy.init(args=args)
    node = RobotMover()
    
    # Example usage
    node.move_robot('ur10e', [2, 0, 2, 2, 0.0, 0.0], oscillate=False)
    node.move_robot('ur3', [2, 0, 2, 2, 0.0, 0.0], oscillate=False)
    time.sleep(2)
    node.move_robot('ur10e', [0.0, 0, 0, 0.0, 0.0, 0.0], oscillate=False)
    node.move_robot('ur3', [0.0, 0, 0, 0.0, 0.0, 0.0], oscillate=False)
    time.sleep(2)
    node.move_robot('ur10e', [2, 0, 2, 0.0, 0.0, 0.0], oscillate=False)
    node.move_robot('ur3', [2, 0, 2, 0.0, 0.0, 0.0], oscillate=False)
    time.sleep(2)
    node.move_robot('ur10e', [3, 2, 1, 0.0, 3, 0.0], oscillate=False)
    node.move_robot('ur3', [3, 2, 1, 0.0, 3, 0.0], oscillate=False)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
