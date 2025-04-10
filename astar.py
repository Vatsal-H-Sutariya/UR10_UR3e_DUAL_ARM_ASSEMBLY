#!/usr/bin/env python3

import rclpy
import heapq
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
        self.base_to_origin_ur10e = self.pose_to_transform(0.774 , 0.4113 , 0, 0, 0, -1.57079632)
        self.base_to_origin_ur3 = self.pose_to_transform( -0.5785 , 0.2265 , 0, 0, 0, 1.57079632)
 
        self.ur10e_diameters = [0.15, 0.12, 0.10, 0.08, 0.08, 0.06]
        self.ur3_diameters = [0.12, 0.10, 0.08, 0.06, 0.06, 0.05]

        self.second_link_offset_ur10e = 0.2
        self.second_link_offset_ur3 = 0.1

        self.UR10e_ee_length = 0.21
        self.UR10e_ee_dia = 0.10
        self.UR10e_ee_TCP = 0.19

        self.UR3_ee_length = 0.21
        self.UR3_ee_dia = 0.10
        self.UR3_ee_TCP = 0.19

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

    def rpy_to_rotation_matrix(self, roll, pitch, yaw):
        
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])

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
            return []

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
            return []
        theta1_options = [
            math.atan2(m, n) - math.atan2(d4, math.sqrt(R**2 - d4**2)),
            math.atan2(m, n) - math.atan2(d4, -math.sqrt(R**2 - d4**2))
        ]

        for theta1 in theta1_options:
            s1, c1 = math.sin(theta1), math.cos(theta1)

            # θ (two solutions: wrist up/down)
            arg = ax * s1 - ay * c1
            if abs(arg) > 1:
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
                    continue
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

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

    def is_in_obstacle(self, pos, obstacles):
        x, y, z = pos
        for (x_min, y_min, z_min, x_max, y_max, z_max) in obstacles:
            if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
                return True
        return False

    def a_star_path(self, start_pos, goal_pos, obstacles, resolution=0.05):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Initialize plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()  # Interactive mode on
        
        # Plot start and goal points
        ax.scatter([start_pos[0]], [start_pos[1]], [start_pos[2]], c='g', marker='o', label='Start')
        ax.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]], c='r', marker='o', label='Goal')

        open_list = []
        closed_set = set()
        heapq.heappush(open_list, (0, start_pos, []))

        dist = self.heuristic(start_pos, goal_pos)
        stepsize = resolution
        print("stepsize: ", stepsize)
        directions = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ]

        max_iterations = 1000
        i = 0
        
        while open_list and i < max_iterations:
            i += 1
            _, current, path = heapq.heappop(open_list)
            
            # Plot current node being evaluated
             
            
            # Plot path so far
            if path:
                path_x, path_y, path_z = zip(*path)
                ax.plot(path_x, path_y, path_z, c='c', alpha=0.5)
            
             
            
            if abs(current[0] - goal_pos[0]) < resolution * 2 and \
               abs(current[1] - goal_pos[1]) < resolution * 2 and \
               abs(current[2] - goal_pos[2]) < resolution * 2:
                # Plot final path
                
                return final_path
                
            closed_set.add(current)
            for move in directions:
                neighbor = (
                    current[0] + move[0] * stepsize,
                    current[1] + move[1] * stepsize,
                    current[2] + move[2] * stepsize
                )
                if neighbor not in closed_set and not self.is_in_obstacle(neighbor, obstacles):
                    g_cost = len(path) + 1
                    h_cost = self.heuristic(neighbor, goal_pos)
                    f_cost = g_cost + h_cost
                    heapq.heappush(open_list, (f_cost, neighbor, path + [current]))
        
        self.get_logger().warning("No path found")
        final_path = path + [current]
        path_x, path_y, path_z = zip(*final_path)
        ax.plot(path_x, path_y, path_z, c='m', linewidth=2, label='Final Path')
        plt.legend()
        plt.show(block=False)

        time.sleep(10)
        plt.close()
        return None

    def plan_run_path(self, start_pos, goal_pos, start_rpy,goal_rpy, robot_type, obstacles):
        
        path = self.a_star_path(start_pos, goal_pos, obstacles)
        if not path:
            self.get_logger().error("Path planning failed")
            return
        self.get_logger().info(f"Path found with {len(path)} waypoints")
        num_steps = len(path)
        
        # Interpolate RPY
        start_r, start_p, start_y = start_rpy
        goal_r, goal_p, goal_y = goal_rpy

        self.get_logger().info(f"Path found with {len(path)} waypoints")
        for i, waypoint in enumerate(path):
            # Linear interpolation of RPY
            t = i / (num_steps - 1) if num_steps > 1 else 1.0
            roll = start_r + t * (goal_r - start_r)
            pitch = start_p + t * (goal_p - start_p)
            yaw = start_y + t * (goal_y - start_y)
            
            # Convert to rotation matrix
            R = self.rpy_to_rotation_matrix(roll, pitch, yaw)
            T_desired = np.eye(4)
            T_desired[0:3, 0:3] = R
            T_desired[0:3, 3] = waypoint

            solutions = self.inverse_kinematics(T_desired, robot_type)
            if solutions:
                joint_angles = solutions[0]  # First valid solution
                self.move_robot(robot_type, joint_angles)
                T_fk = self.forward_kinematics(joint_angles, robot_type)
                self.get_logger().info(f"Moved to {waypoint}, RPY: ({roll:.3f}, {pitch:.3f}, {yaw:.3f})")
                self.get_logger().info(f"Joints: {joint_angles}")
                self.get_logger().info(f"FK Position: {T_fk[0:3, 3]}")
                time.sleep(1)
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

    def example(self, shoulder):

        
        time.sleep(2)
        gripper_activate = "rq_activate_and_wait()\n"  # Activates the gripper
        gripper_close = "rq_set_pos(0)\n"

        
        s_UR10e =  socket.create_connection((ROBOT_IP_UR10e, PORT), timeout=5) 
        s_UR3 = socket.create_connection((ROBOT_IP_UR3, PORT), timeout=5) 
        print("Connected to UR10!")

        s_UR10e.sendall(gripper_activate.encode('utf-8'))
        print(f"Sent command: {gripper_activate.strip()}")
        time.sleep(2)

        self.move_robot('ur10e', [0, -1.57, 1.57, 0, 0.0, 0.0])
        self.move_robot('ur3', [math.pi, -1.57, 1.57, 0, 0.0, 0.0])
        collisions = self.check_collisions()
        print(collisions)
        if len(collisions) != 0:
            return 0
        move_command1 = f"movej([{', '.join(map(str, [0, -1.65, -1.57, 0, math.pi, 0.0]))}], a=0.4, v=0.2)\n"
        move_command2 = f"movej([{', '.join(map(str, [0, -1.65, 1.57, 0, 0.0, 0.0]))}], a=0.4, v=0.2)\n"
        
        s_UR10e.sendall(move_command1.encode('utf-8'))
        s_UR3.sendall(move_command2.encode('utf-8'))

        print(f"Sent command: {move_command1.strip()}")
        time.sleep(5)

        s_UR10e.sendall(gripper_close.encode('utf-8'))
        print(f"Sent command: {gripper_close.strip()}")
        time.sleep(5)

        


        
       
        time.sleep(2)
        self.move_robot('ur10e', [-0.3, shoulder, -1.57, 3.14, 0.0, 0.0])
        self.move_robot('ur3', [0, -1.57, 1.57, 0, 0.0, 0.0])
        collisions = self.check_collisions()
        print(collisions)
        if len(collisions) != 0:
            return 0
        time.sleep(2)
        self.move_robot('ur10e', [0, shoulder, -1.57, 3.14, 0.0, 0.0])
        self.move_robot('ur3', [0, -1.57, 1.57, 0, 0.0, 0.0])
        collisions = self.check_collisions()
        print(collisions)
        if len(collisions) != 0:
            return 0
        time.sleep(2)
        self.move_robot('ur10e', [0.2, shoulder, -1.57, 3.14, 0.0, 0.0])
        self.move_robot('ur3', [0, -1.57, 1.57, 0, 0.0, 0.0])
        collisions = self.check_collisions()
        print(collisions)
        if len(collisions) != 0:
            return 0
        time.sleep(2)
        return 1



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
    
    

    '''
    suc = node.example(-1)
    if suc ==1 :
        print("path complete")
    else:
        print("collision!!!!!!!!!!!!!!")
    '''

     
     
    

    start_pos = (-0.3652499973890609, 0.6107499985509883, 0.31020000000000003)  # meters
    goal_pos = (-0.19425000144901167, 0.013249997389060924, 0.31020000000000003)
    start_rpy = (1.5707963267948966, -0.0, 3.1415926467948965)  # radians (identity)
    goal_rpy = (1.5707963267948966, -0.0, 1.57079632)  # 90° roll, 45° yaw
    obstacles = [ ]
 
    # Start from zero pose
    start_config = [0.0] * 6

    print("\nPlanning and Executing Path for UR5:")
    node.plan_run_path(start_pos, goal_pos, start_rpy, goal_rpy, 'ur3', obstacles)

  
    '''
    node.move_robot('ur3', [math.pi/2, -math.pi/2, math.pi/2, 0, 0, 0.0])
    print("moving to ")
    time.sleep(5)
    position, orientation = node.get_end_effector_pose('ur3')
    print(position)
    print(orientation)
    '''


    '''
    node.move_robot('ur3', [0, -math.pi/2, math.pi/2, 0, 0, 0.0])
    print("moving to ")
    time.sleep(5)
    position, orientation = node.get_end_effector_pose('ur3')
    
    T_desired = np.eye(4)
    T_desired[:3, 3] = position
    # Convert RPY to rotation matrix
    q = quaternion_from_euler(orientation[0], orientation[1], orientation[2])
    T_desired[:3, :3] = quaternion_matrix(q)[:3, :3]
    print(position)
    print(orientation)
    print("Desired Pose (T_desired):")
    print(T_desired)

    solutions = node.inverse_kinematics(T_desired, 'ur3')
    print(solutions)
    for i, sol in enumerate(solutions):
        print(f"Solution {i + 1}: {sol}")
        node.move_robot('ur3', sol)
        time.sleep(5)

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
