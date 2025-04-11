#!/usr/bin/env python3

import rclpy
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker
import math
import time
import numpy as np
import tf2_ros
from tf_transformations import quaternion_from_euler, quaternion_from_matrix, quaternion_about_axis, quaternion_matrix
import threading
import socket
import time
ROBOT_IP_UR3 = "192.168.0.50"  # Change this to match your setup
PORT = 30003  # UR10 secondary interface port
ROBOT_IP_UR10e = "192.168.0.51"

class PP:
    def __init__(self, pos, g=0, h=0):
        self.pos = tuple(pos)  # (x, y, z) in meters
        self.g = g  # Cost from start
        self.h = h  # Heuristic to goal
        self.f = g + h  # Total cost
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

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
        self.base_to_origin_ur10e = self.pose_to_transform(0.7525 , 0.4275, 0, 0, 0, -1.57079632)
        self.base_to_origin_ur3 = self.pose_to_transform( -0.6025 , 0.2225  ,0, 0, 0, 1.57079632)
 
        self.ur10e_diameters = [0.15, 0.12, 0.10, 0.08, 0.08, 0.06]
        self.ur3_diameters = [0.12, 0.10, 0.08, 0.06, 0.06, 0.05]

        self.second_link_offset_ur10e = 0.2
        self.second_link_offset_ur3 = 0.1

        self.UR10e_ee_length = 0.19
        self.UR10e_ee_dia = 0.115
        self.UR10e_ee_TCP = 0.17

        self.UR3_ee_length = 0.19
        self.UR3_ee_dia = 0.10
        self.UR3_ee_TCP = 0.17

        self.joint_limits = []
        self.obstacles = []
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

    def pose_to_transform(self, x, y, z, roll, pitch, yaw):
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combine rotations: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx

        # Construct the homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T


    def get_end_effector_pose(self, robot_type: str):
        robot_type = robot_type.lower()
        if robot_type == 'ur10e':
            joints = self.ur10e_current_positions
            dh_params = self.ur10e_dh
            T_tool = self.pose_to_transform(0,0,self.UR10e_ee_TCP,0,0,0)
        elif robot_type == 'ur3':
            joints = self.ur3_current_positions
            dh_params = self.ur3_dh
            T_tool = self.pose_to_transform(0,0,self.UR3_ee_TCP,0,0,0)
        else:
            self.get_logger().error('Invalid robot_type')
            return None, None

        T = np.eye(4)
        if robot_type == 'ur10e':
            T = self.base_to_origin_ur10e
        elif robot_type == 'ur3':
            T = self.base_to_origin_ur3
        else:
            print("ffffffffffffffffffffffff")
        for i in range(6):
            T = T @ self.get_transform(joints[i], dh_params[i][0], dh_params[i][1], dh_params[i][2])
        T = T @  T_tool
        position = T[:3, 3]
        roll = math.atan2(T[2,1], T[2,2])
        pitch = math.atan2(-T[2,0], math.sqrt(T[2,1]**2 + T[2,2]**2))
        yaw = math.atan2(T[1,0], T[0,0])

        return position.tolist(), [roll, pitch, yaw]

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



    def get_link_poses(self, robot_type: str):
        
        robot_type = robot_type.lower()
        if robot_type == 'ur10e':
            joints = self.ur10e_current_positions
            dh_params = self.ur10e_dh
            diameters = self.ur10e_diameters
            T_tool = self.pose_to_transform(0,0,self.UR10e_ee_TCP,0,0,0)
            tool_dia = self.UR10e_ee_dia
        elif robot_type == 'ur3':
            joints = self.ur3_current_positions
            dh_params = self.ur3_dh
            diameters = self.ur3_diameters
            T_tool = self.pose_to_transform(0,0,self.UR3_ee_TCP,0,0,0)
            tool_dia = self.UR3_ee_dia
        else:
            return []

        poses = []
        T = np.eye(4)
        T_third = np.eye(4)
        if robot_type == 'ur10e':
            T = self.base_to_origin_ur10e
            T_third = self.base_to_origin_ur3
        elif robot_type == 'ur3':
            T = self.base_to_origin_ur3
            T_third = self.base_to_origin_ur10e
        prev_pos = np.array(T[:3, 3])
                
        
        for j in range(2):  # Up to third joint (elbow_joint)
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

        T = T @  T_tool

        curr_pos = T[:3, 3]
        length = np.linalg.norm(curr_pos - prev_pos)
        direction = (curr_pos - prev_pos) / length if length > 0 else np.array([0, 0, 1])
        midpoint = (prev_pos + curr_pos) / 2    
        poses.append((midpoint.tolist(), direction.tolist() , length, tool_dia))


        return poses

    def update_cylinder_list(self):
        
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

    def cylinder_collision(self, cyl1: Cylinder, cyl2: Cylinder) -> bool:
        
        if cyl1.robot_type == cyl2.robot_type and abs(cyl1.link_id - cyl2.link_id) <= 1:
            return False

        # Line segment endpoints
        p1_start = cyl1.position - (cyl1.direction * cyl1.length / 2)
        p1_end = cyl1.position + (cyl1.direction * cyl1.length / 2)
        p2_start = cyl2.position - (cyl2.direction * cyl2.length / 2)
        p2_end = cyl2.position + (cyl2.direction * cyl2.length / 2)

        # Vector between lines
        v = p1_start - p2_start
        d1 = p1_end - p1_start  # Direction vector * length
        d2 = p2_end - p2_start

        # Calculate closest points
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, v)
        e = np.dot(d2, v)

        denom = a * c - b * b

        if abs(denom) < 1e-6:  # Parallel case
            t = 0.0
            s = max(0.0, min(1.0, -d / a if a > 1e-6 else 0.0))
        else:
            s = (b * e - c * d) / denom
            t = (a * e - b * d) / denom
            # Clamp to [0,1] range (line segment bounds)
            s = max(0.0, min(1.0, s))
            t = max(0.0, min(1.0, t))

        # Closest points
        closest1 = p1_start + s * d1
        closest2 = p2_start + t * d2

        # Distance between closest points
        distance = np.linalg.norm(closest1 - closest2)

        # Check if within sum of radii
        collision = distance < (cyl1.radius + cyl2.radius)

        # Additional check: if no collision but segments might still intersect
        if not collision:
            # Check if either endpoint of one cylinder is within the other
            for p in [p1_start, p1_end]:
                t = np.dot(p - p2_start, d2) / (c * c) if c > 1e-6 else 0.0
                t = max(0.0, min(1.0, t))
                closest = p2_start + t * d2
                if np.linalg.norm(p - closest) < (cyl1.radius + cyl2.radius):
                    return True
            for p in [p2_start, p2_end]:
                t = np.dot(p - p1_start, d1) / (a * a) if a > 1e-6 else 0.0
                t = max(0.0, min(1.0, t))
                closest = p1_start + t * d1
                if np.linalg.norm(p - closest) < (cyl1.radius + cyl2.radius):
                    return True

        return collision    

    def check_collisions(self):
       
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

    def publish_obstacle_markers(self,obstacles, namespace="obstacles"):
    
        for i, obs in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = namespace
            marker.id = i
            marker.type = Marker.CUBE  # Use cube for rectangular cuboids
            marker.action = Marker.ADD
            
            # Position at cuboid center
            center = obs['center']
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            
            # Orientation: Identity (aligned with world axes)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Scale: sx, sy, sz for x, y, z dimensions
            size = obs['size']
            marker.scale.x = float(size[0])  # Width (x-axis)
            marker.scale.y = float(size[1])  # Depth (y-axis)
            marker.scale.z = float(size[2])  # Height (z-axis)
            
            # Color: Gray to match scatter-point style, semi-transparent
            marker.color.r = 0.9
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 0.5  # Semi-transparent
            
            # Publish the marker
            self.marker_pub.publish(marker)



    def inverse_kinematics(self, T_desired, robot_type: str):
         
        if robot_type.lower() == 'ur10e':
            d1, a2, a3, d4, d5, d6 = 0.1807, -0.6127, -0.57155, 0.17415, 0.11985, 0.11655
            T_base = self.base_to_origin_ur10e
            T_tool = self.pose_to_transform(0,0,self.UR10e_ee_TCP,0,0,0)
        elif robot_type.lower() == 'ur3':
            d1, a2, a3, d4, d5, d6 = 0.1519, -0.24365, -0.21325, 0.11235, 0.08535, 0.0819
            T_base = self.base_to_origin_ur3
            T_tool = self.pose_to_transform(0,0,self.UR3_ee_TCP,0,0,0)
        else:
            self.get_logger().error('Invalid robot_type')
            return 0

        # Transform desired pose to base frame
        T_base_inv = np.linalg.inv(T_base)
        T_desired_base = T_base_inv @ T_desired

        T_tool_inv = np.linalg.inv(T_tool)
        T_desired_flange = T_desired_base @ T_tool_inv

        # Extract position and orientation
        nx, ny, nz = T_desired_flange[0, 0], T_desired_flange[1, 0], T_desired_flange[2, 0]
        ox, oy, oz = T_desired_flange[0, 1], T_desired_flange[1, 1], T_desired_flange[2, 1]
        ax, ay, az = T_desired_flange[0, 2], T_desired_flange[1, 2], T_desired_flange[2, 2]
        px, py, pz = T_desired_flange[0, 3], T_desired_flange[1, 3], T_desired_flange[2, 3]

        solutions = []

        # θ1 (two solutions: shoulder left/right)
        m = d6 * ay - py
        n = ax * d6 - px
        R = math.sqrt(m**2 + n**2)
        if R < abs(d4):
            print("theta 1 error : R < d4")
            return 0
        theta1_options = [
            math.atan2(m, n) - math.atan2(d4, math.sqrt(R**2 - d4**2)),
            math.atan2(m, n) - math.atan2(d4, -math.sqrt(R**2 - d4**2))
        ]

        for theta1 in theta1_options:
            s1, c1 = math.sin(theta1), math.cos(theta1)

            # θ (two solutions: wrist up/down)
            arg = ax * s1 - ay * c1
            if abs(arg) > 1:
                print("theta 5 error : arg > 1")
                continue
            theta5_options = [math.acos(arg), -math.acos(arg)]

            for theta5 in theta5_options:
                s5, c5 = math.sin(theta5), math.cos(theta5)

                # θ
                mm = nx * s1 - ny * c1
                nn = ox * s1 - oy * c1
                theta6 = math.atan2(mm, nn) - math.atan2(s5, 0) if abs(s5) > 1e-6 else 0.0

                # Compute wrist position for θ, θ
                m = d5 * (math.sin(theta6) * (nx * c1 + ny * s1) + math.cos(theta6) * (ox * c1 + oy * s1)) - \
                    d6 * (ax * c1 + ay * s1) + px * c1 + py * s1
                n = pz - d1 - az * d6 + d5 * (oz * math.cos(theta6) + nz * math.sin(theta6))

                # θ (two solutions: elbow up/down)
                D = (m**2 + n**2 - a2**2 - a3**2) / (2 * abs(a2) * abs(a3))
                if abs(D) > 1.01:  # Relaxed boundary
                    print("theta 3 error : D > 1")
                    return 0
                D = max(min(D, 1.0), -1.0)
                theta3_options = [math.acos(D), -math.acos(D)]

                for theta3 in theta3_options:
                    s3, c3 = math.sin(theta3), math.cos(theta3)

                    # θ
                    s2 = ((a3 * c3 + a2) * n - a3 * s3 * m) / (a2**2 + a3**2 + 2 * a2 * a3 * c3)
                    c2 = (m + a3 * s3 * s2) / (a3 * c3 + a2)
                    theta2 = math.atan2(s2, c2)

                    # θ
                    theta4 = math.atan2(
                        -math.sin(theta6) * (nx * c1 + ny * s1) - math.cos(theta6) * (ox * c1 + oy * s1),
                        oz * math.cos(theta6) + nz * math.sin(theta6)
                    ) - theta2 - theta3

                    solutions.append([theta1, theta2, theta3, theta4, theta5, theta6])

        return solutions

    def is_links_colliding_with_obs(self, joint_angles: list, robot_type: str) -> bool:
        
        
        original_joints = (self.ur3_current_positions.copy() if robot_type == 'ur3' 
                          else self.ur10e_current_positions.copy())
        if robot_type == 'ur3':
            self.ur3_current_positions = joint_angles.copy()
        else:
            self.ur10e_current_positions = joint_angles.copy()

        # Get link poses (position, direction, length, diameter)
        link_poses = self.get_link_poses(robot_type)
        if self.check_collisions():
            return  True

        # Restore original joint positions
        if robot_type == 'ur3':
            self.ur3_current_positions = original_joints
        else:
            self.ur10e_current_positions = original_joints

        # Check each link against each obstacle
        for pos, direction, length, diameter in link_poses:
            cyl_pos = np.array(pos)
            cyl_dir = np.array(direction)
            cyl_radius = diameter / 2
            cyl_start = cyl_pos - (cyl_dir * length / 2)
            cyl_end = cyl_pos + (cyl_dir * length / 2)

            for obs in self.obstacles:
                obs_center = np.array(obs['center'])
                obs_half_size = np.array(obs['size']) / 2

                # Find closest point on cylinder axis to obstacle center
                t = np.dot(obs_center - cyl_start, cyl_dir) / np.dot(cyl_dir, cyl_dir)
                t = max(0, min(1, t))  # Clamp to segment
                closest_point = cyl_start + t * cyl_dir

                # Transform to obstacle's local frame (assuming axis-aligned cuboid)
                diff = obs_center - closest_point
                dist = np.abs(diff)

                # Check if closest point is within obstacle bounds plus cylinder radius
                if (dist[0] <= obs_half_size[0] + cyl_radius and
                    dist[1] <= obs_half_size[1] + cyl_radius and
                    dist[2] <= obs_half_size[2] + cyl_radius):
                    return True

                # Check cylinder endpoints
                for p in [cyl_start, cyl_end]:
                    diff = obs_center - p
                    dist = np.abs(diff)
                    if (dist[0] <= obs_half_size[0] + cyl_radius and
                        dist[1] <= obs_half_size[1] + cyl_radius and
                        dist[2] <= obs_half_size[2] + cyl_radius):
                        return True

        return False    

    def is_in_obstacle(self, pos , robot_type):
        x, y, z = pos
        
        T_desired = np.eye(4)
        T_desired[:3, 3] = [x, y, z]
        q = quaternion_from_euler(0,math.pi,0)
        T_desired[:3, :3] = quaternion_matrix(q)[:3, :3]
        try:            
            solutions = self.inverse_kinematics(T_desired, robot_type)
        except:
            return 0
        if solutions == 0:
            return True
            print(solutions)
        sol = solutions[5]
        for obs in self.obstacles:
            center = obs['center']
            size = obs['size']  # (sx, sy, sz) for x, y, z edge lengths
            half_sx, half_sy, half_sz = size[0] / 2, size[1] / 2, size[2] / 2
            cx, cy, cz = center
            # Check if point is within the cuboid's bounds
            if (cx - half_sx <= x <= cx + half_sx and
                cy - half_sy <= y <= cy + half_sy and
                cz - half_sz <= z <= cz + half_sz):

                return True
        if self.is_links_colliding_with_obs(sol, robot_type):
            return True
           
         
        return False

    
    

    # 3D A* Path Planning
    def a_star_3d(self, start, goal,  robot_type, grid_size,  step_size=0.2):
        # Directions: 6-connectivity (up, down, left, right, forward, backward)
        directions = [(step_size, 0, 0), (-step_size, 0, 0), 
                      (0, step_size, 0), (0, -step_size, 0), 
                      (0, 0, step_size), (0, 0, -step_size)]
        
        # Initialize open and closed lists
        open_list = []
        closed_set = set()
        
        # Create start node
        start = tuple(np.round(np.array(start) / step_size) * step_size)  # Snap to grid
        goal = tuple(np.round(np.array(goal) / step_size) * step_size)    # Snap to grid
        start_node = PP(start, g=0, h=np.linalg.norm(np.array(start) - np.array(goal)))
        heapq.heappush(open_list, start_node)
        
        while open_list:
            
            # Get node with lowest f score
            current = heapq.heappop(open_list)
            current_pos = np.array(current.pos)
            
            # Check if close enough to goal (within step_size)
            if np.linalg.norm(current_pos - np.array(goal)) <= step_size:
                print("W")
                path = []
                while current:
                    path.append(current.pos)
                    current = current.parent
                return path[::-1]  # Reverse path
             
            # Add to closed set
            closed_set.add(current.pos)
            
            # Explore neighbors
            for direction in directions:
                neighbor_pos = tuple(np.round((current_pos + direction) / step_size) * step_size)
                x, y, z = neighbor_pos
                
                # Check bounds and obstacles
                if (-2 <= x <= grid_size[0] and -2 <= y <= grid_size[1] and -2 <= z <= grid_size[2] and
                    
                    not self.is_in_obstacle(neighbor_pos , robot_type) and neighbor_pos not in closed_set):
                     
                    # Calculate costs
                    g = current.g + step_size  # Cost per step
                    h = np.linalg.norm(np.array(neighbor_pos) - np.array(goal))
                    
                    # Check if neighbor is already in open list
                    neighbor = PP(neighbor_pos, g, h)
                    neighbor.parent = current
                    
                    # If neighbor is in open list with higher f, skip
                    if any(n.pos == neighbor_pos and n.f <= neighbor.f for n in open_list):
                        continue
                    
                    heapq.heappush(open_list, neighbor)
        
        return None  # No path found

    # Visualization (scatter points for grid cells in rectangular cuboids)
    def visualize_path(self, start, goal, path , grid_size):
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set plot limits
        ax.set_xlim(0, grid_size[0])
        ax.set_ylim(0, grid_size[1])
        ax.set_zlim(0, grid_size[2])
        
        # Plot obstacles as scatter points for all grid cells within bounds
        for obs in self.obstacles:
            center = np.array(obs['center'])
            size = obs['size']  # (sx, sy, sz) for x, y, z edge lengths
            half_sx, half_sy, half_sz = size[0] / 2, size[1] / 2, size[2] / 2
            # Define bounds in meters
            x_min = center[0] - half_sx
            x_max = center[0] + half_sx
            y_min = center[1] - half_sy
            y_max = center[1] + half_sy
            z_min = center[2] - half_sz
            z_max = center[2] + half_sz
            # Generate points for each grid cell within the cuboid
            obs_points = [
                (x, y, z)
                for x in np.arange(np.floor(x_min), np.ceil(x_max) + 0.1, 0.2)  # Step by 0.2m
                for y in np.arange(np.floor(y_min), np.ceil(y_max) + 0.1, 0.2)
                for z in np.arange(np.floor(z_min), np.ceil(z_max) + 0.1, 0.2)
                if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max
            ]
            if obs_points:
                ox, oy, oz = zip(*obs_points)
                ax.scatter(ox, oy, oz, color='gray', alpha=0.5, s=40,
                           label='Obstacle' if 'Obstacle' not in ax.get_legend_handles_labels()[1] else "")
        
        # Plot path
        if path:
            path_x, path_y, path_z = zip(*path)
            ax.plot(path_x, path_y, path_z, marker='o', color='red', linewidth=2, label='Path')
        
        # Plot start and goal
        ax.scatter([start[0]], [start[1]], [start[2]], color='green', s=100, label='Start')
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color='blue', s=100, label='Goal')
        
        # Labels and legend
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.legend()
        plt.title('3D Path Planning with A*')
        plt.show()    

    def plan_run_path(self, start_pos, goal_pos, orientation, robot_type, obstacles):
        
        path = self.a_star_path(start_pos, goal_pos, obstacles)
        if not path:
            self.get_logger().error("Path planning failed")
            return

        # Convert orientation to 4x4 matrix
        T_orient = np.eye(4)
        T_orient[0:3, 0:3] = orientation

        self.get_logger().info(f"Path found with {len(path)} waypoints")
        for waypoint in path:
            T_desired = T_orient.copy()
            T_desired[0:3, 3] = waypoint
            solutions = self.inverse_kinematics(T_desired, robot_type)
            if solutions:
                # Use the first valid solution (could add selection logic)
                joint_angles = solutions[0]
                self.move_robot(robot_type, joint_angles)
                T_fk = self.forward_kinematics(joint_angles, robot_type)
                self.get_logger().info(f"Moved to {waypoint}, Joints: {joint_angles}")
                self.get_logger().info(f"FK Position: {T_fk[0:3, 3]}")
                time.sleep(1)  # Simulate movement delay
            else:
                self.get_logger().warning(f"No IK solution for waypoint {waypoint}")


    def move_robot(self, robot_type: str, joint_positions: list):
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
            
        
        elif robot_type == 'ur3':
            msg.name = self.ur3_joint_names
            msg.position = joint_positions
            self.ur3_pub.publish(msg)
            self.ur3_current_positions = joint_positions.copy()
            
        
        else:
            self.get_logger().error('Invalid robot_type')
            return

        self.publish_tf(robot_type)
        self.publish_markers(robot_type)

    def move_real_robot(self, robot_type: str, joint_positions: list, s_UR10e, s_UR3):
        self.move_robot(robot_type, joint_positions) 
        collisions = self.check_collisions()
        print(collisions)
        if len(collisions) != 0:
            return 0
        move_command = f"movej([{', '.join(map(str, joint_positions))}], a=0.4, v=0.2)\n"
        if robot_type == "ur10e":
            s_UR10e.sendall(move_command.encode('utf-8'))
        else:
            s_UR3.sendall(move_command.encode('utf-8'))

        print(f"Sent command: {move_command.strip()}")
        return 1

    def connect_to_robots(self,ROBOT_IP_UR10e,ROBOT_IP_UR3,PORT):
        s_UR10e =  socket.create_connection((ROBOT_IP_UR10e, PORT), timeout=5) 
        print("Connected to UR10e!")
        s_UR3 = socket.create_connection((ROBOT_IP_UR3, PORT), timeout=5) 
        print("Connected to UR3!")
        return s_UR10e, s_UR3

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

        self.publish_obstacle_markers(self.obstacles,  namespace="obstacles")
        self.check_collisions()







def main(args=None):
    rclpy.init(args=args)
    node = RobotMover()
    
    

    '''
    suc = node.example(-1)
    if suc ==1 :
        print("path complete")
    else:
        print("collision!!!!!!!!!!!!!!")
    '''




    '''
    node.move_robot('ur3', [0, -math.pi/2, math.pi/2, 0, 0, 0.0])
    node.move_robot('ur10e', [0, -math.pi/2, -math.pi/2, 0, 0, 0.0])
    print("moving to ") 
    position, orientation = node.get_end_effector_pose('ur3')
    position1, orientation1 = node.get_end_effector_pose('ur10e')
    print(position)
    print(orientation)
    print(position1)
    print(orientation1)
    
    s_UR10e, s_UR3 =node.connect_to_robots(ROBOT_IP_UR10e,ROBOT_IP_UR3,PORT,)


    #node.move_real_robot('ur3', [0, -math.pi/2, math.pi/2, 0, 0, 0.0], s_UR10e, s_UR3)
    #node.move_real_robot('ur10e', [0, -math.pi/2, -1.75, 0, 0, 0.0], s_UR10e, s_UR3)
    print("done")
'''
    grid_size = (2, 2, 2)  # 10x10x10 meters
    
    # Define step size (meters)
    step_size = 0.1 # Move 0.2 meters per step
    
    # Define start and goal points
    start = (-0.1, -0.0, 0.1)
    goal = (-0.1, 0.7, 0.1)
    
    # Define obstacles as variable-size cubes: {'center': (x, y, z), 'size': edge_length}
    node.obstacles = [
        {'center': (0.0, 0.4, 0.1), 'size': (0.2, 0.1, 3)},   
         
    ]
    node.move_robot('ur3', [1.40,-0, 0, 0,math.pi/2,0])
    # Ensure start and goal are not in obstacles
    if node.is_in_obstacle(start , 'ur10e') :
        print("Start   is inside an obstacle!")
        return
    if node.is_in_obstacle(goal , 'ur10e') :
        print("  goal is inside an obstacle!")
        return
     
    # Run A* algorithm
    path = node.a_star_3d(start, goal, 'ur10e',grid_size , step_size)
    
    # Visualize
    if path:
        print("Path found with", len(path), "steps")
        node.visualize_path(start, goal, path , grid_size)
    else:
        print("No path found!")
        node.visualize_path(start, goal, None  ,grid_size)
    







         
    for p in path:
        
        T_desired = np.eye(4)
        T_desired[:3, 3] = [p[0],p[1],p[2]]
        # Convert RPY to rotation matrix
        q = quaternion_from_euler(0,math.pi,0)
        T_desired[:3, :3] = quaternion_matrix(q)[:3, :3]
        
         

        solutions = node.inverse_kinematics(T_desired, 'ur10e')
         
        node.move_robot('ur10e', solutions[5])
        time.sleep(0.2)

    '''
    T_desired = np.eye(4)
    T_desired[:3, 3] = [-0.5,0,0.1]
    # Convert RPY to rotation matrix
    q = quaternion_from_euler(0,math.pi,0)
    T_desired[:3, :3] = quaternion_matrix(q)[:3, :3]


    solutions = node.inverse_kinematics(T_desired, 'ur3')
    print(solutions)  
    node.move_robot('ur3', solutions[5])
    time.sleep(5)

    #s_UR10e, s_UR3 =node.connect_to_robots(ROBOT_IP_UR10e,ROBOT_IP_UR3,PORT)
    #node.move_real_robot('ur10e',  solutions[1], s_UR10e, s_UR3)

    print('done')
 
    '''









    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
