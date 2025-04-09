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
from tf_transformations import quaternion_from_euler, quaternion_from_matrix, quaternion_about_axis,quaternion_matrix

import socket
import time
ROBOT_IP = "192.168.0.51"  # Change this to match your setup
PORT = 30002  # UR10 secondary interface port



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
        self.base_to_origin_ur10e = self.get_transform( math.pi, -1,0,0)
        self.base_to_origin_ur3 = self.get_transform( math.pi, 0,0,0)
 
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
        
        for i in range(5):
            T = T @ self.get_transform(joints[i], dh_params[i][0], dh_params[i][1], dh_params[i][2])
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

    def cylinder_collision(self, cyl1: Cylinder, cyl2: Cylinder) -> bool:
        """Check collision between two cylinders with improved accuracy"""
        # Skip adjacent links of same robot
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



    def inverse_kinematics(self, T_desired, robot_type: str):
        """
        Calculate inverse kinematics for UR10e or UR3 given desired end-effector pose.
        
        Args:
            T_desired: 4x4 numpy array representing desired end-effector transformation
            robot_type: 'ur10e' or 'ur3'
            
        Returns:
            List of possible joint angle solutions [theta1, theta2, theta3, theta4, theta5, theta6]
        """
        # Extract position and orientation from desired transform
        px = T_desired[0, 3]
        py = T_desired[1, 3]
        pz = T_desired[2, 3]
        
        # Rotation matrix components
        nx, ny, nz = T_desired[0, 0], T_desired[1, 0], T_desired[2, 0]
        ox, oy, oz = T_desired[0, 1], T_desired[1, 1], T_desired[2, 1]
        ax, ay, az = T_desired[0, 2], T_desired[1, 2], T_desired[2, 2]

        # Select DH parameters based on robot type
        if robot_type.lower() == 'ur10e':
            d1 = 0.1807
            a2 = -0.6127
            a3 = -0.57155
            d4 = 0.17415
            d5 = 0.11985
            d6 = 0.11655
        elif robot_type.lower() == 'ur3':
            d1 = 0.1519
            a2 = -0.24365
            a3 = -0.21325
            d4 = 0.11235
            d5 = 0.08535
            d6 = 0.0819
        else:
            self.get_logger().error('Invalid robot_type')
            return []

        solutions = []

        # Step 1: Solve for theta1 (two solutions: shoulder left/right)
        p05 = np.array([px - d6 * ax, py - d6 * ay, pz - d6 * az])
        print(p05)
        R = np.sqrt(p05[0]**2 + p05[1]**2)
        
        if R < abs(d4):
            return []  # No solution possible
        
        alpha1 = math.atan2(p05[1], p05[0])
        alpha2 = math.acos(d4 / R)
        theta1_options = [
            alpha1 + alpha2 + math.pi/2,
            alpha1 - alpha2 + math.pi/2
        ]

        # For each theta1 solution
        for theta1 in theta1_options:
            c1 = math.cos(theta1)
            s1 = math.sin(theta1)

            # Step 2: Solve for theta5 (two solutions: wrist up/down)
            arg = (px * s1 - py * c1 - d4) / d6
            if abs(arg) > 1:
                continue
            theta5_options = [
                math.acos(arg),
                -math.acos(arg)
            ]

            # For each theta5 solution
            for theta5 in theta5_options:
                s5 = math.sin(theta5)
                c5 = math.cos(theta5)

                # Step 3: Solve for theta6
                if abs(s5) < 1e-6:  # Singularity case
                    theta6 = 0.0  # Arbitrary choice when s5 = 0
                else:
                    theta6 = math.atan2(
                        (-ny * s1 + oy * c1) / s5,
                        -(-nx * s1 + ox * c1) / s5
                    )

                solution = [theta1,theta5]
                solutions.append(solution)

            

        return solutions

    def plan_path(self):
        return



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

        try:
            with socket.create_connection((ROBOT_IP, PORT), timeout=2) as s:

                print("Connected to UR10!")

                s.sendall(gripper_activate.encode('utf-8'))
                print(f"Sent command: {gripper_activate.strip()}")
                time.sleep(2)

                self.move_robot('ur10e', [-0.4, shoulder, -1.57, 3.14, 0.0, 0.0])
                self.move_robot('ur3', [0, -1.57, 1.57, 0, 0.0, 0.0])
                collisions = self.check_collisions()
                print(collisions)
                if len(collisions) != 0:
                    return 0

                move_command1 = f"movej([{', '.join(map(str, [-0.4, shoulder, -1.57, 3.14, 0.0, 0.0]))}], a=0.4, v=0.5)\n"
                s.sendall(move_command1.encode('utf-8'))
                print(f"Sent command: {move_command1.strip()}")
                time.sleep(5)

                s.sendall(gripper_close.encode('utf-8'))
                print(f"Sent command: {gripper_close.strip()}")
                time.sleep(5)

        except Exception as e:
            print(f"Error: {e}")


        
       
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
    node.move_robot('ur3', [math.pi, 0, 0, math.pi/2, math.pi, 0.0])
    time.sleep(3)
    position, orientation = node.get_end_effector_pose('ur10e')
    
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
    for i, sol in enumerate(solutions):
        print(f"Solution {i + 1}: {sol}")
        node.move_robot('ur3', [sol[0], 0, 0, 0, sol[1], 0.0])
        time.sleep(2)

    print('done')
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
