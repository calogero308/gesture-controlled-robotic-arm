"""
Avvia l'intero sistema dal vivo:
  camera_node -> gesture_node -> arm_control_node -> /joint_states -> robot_state_publisher -> RViz

Uso (il server webcam deve girare su Windows):
  ros2 launch braccio_bringup live.launch.py source:=http://IP:8080/video
Argomenti opzionali: repo_dir:=...  show:=true|false (finestra video annotata)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

DEFAULT_REPO = "/mnt/c/Users/calog/Desktop/Secret_tasks/robot-gesti-cuda"


def generate_launch_description():
    urdf_path = os.path.join(get_package_share_directory("braccio_description"), "urdf", "robotic_arm.urdf")
    with open(urdf_path) as f:
        robot_description = f.read()
    rviz_config = os.path.join(get_package_share_directory("braccio_bringup"), "rviz", "live.rviz")

    source = LaunchConfiguration("source")
    repo_dir = LaunchConfiguration("repo_dir")
    show = LaunchConfiguration("show")

    return LaunchDescription([
        DeclareLaunchArgument("source", description="Indirizzo dello stream (http://IP:8080/video), un file video o un indice di camera"),
        DeclareLaunchArgument("repo_dir", default_value=DEFAULT_REPO),
        DeclareLaunchArgument("show", default_value="true"),

        Node(
          package="robot_state_publisher", 
          executable="robot_state_publisher",
          parameters=[{"robot_description": robot_description}], 
          output="screen"),

        Node(
          package="braccio_perception", 
          executable="camera_node", 
          output="screen",
          parameters=[{"source": ParameterValue(source, value_type=str)}]),
        
        Node(
          package="braccio_perception", 
          executable="gesture_node", 
          output="screen",
          parameters=[{"repo_dir": ParameterValue(repo_dir, value_type=str),
                         "show": ParameterValue(show, value_type=bool)}]),
        
        Node(
          package="braccio_control", 
          executable="arm_control_node", 
          output="screen",
          parameters=[{"repo_dir": ParameterValue(repo_dir, value_type=str)}]),
        
        Node(
          package="rviz2", 
          executable="rviz2", 
          arguments=["-d", rviz_config], 
          output="screen"),
    ])
