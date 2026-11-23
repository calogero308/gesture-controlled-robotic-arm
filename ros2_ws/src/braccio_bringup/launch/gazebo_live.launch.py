import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

DEFAULT_REPO = "/mnt/c/Users/calog/Desktop/Secret_tasks/robot-gesti-cuda"

def generate_launch_description():

    # Ottengo il percorso di gazebo.launch.py nel pacchetto braccio_bringup
    bringup_dir = get_package_share_directory("braccio_bringup")
    gazebo_launch_path = os.path.join(bringup_dir, "launch", "gazebo.launch.py")

    source = LaunchConfiguration("source")
    repo_dir = LaunchConfiguration("repo_dir")
    show = LaunchConfiguration("show")

    # Include di gazebo.launch.py
    gazebo = IncludeLaunchDescription(PythonLaunchDescriptionSource(gazebo_launch_path))

    return LaunchDescription([
        DeclareLaunchArgument("source", description="Indirizzo dello stream (http://IP:8080/video), un file video o un indice di camera"),
        DeclareLaunchArgument("repo_dir", default_value=DEFAULT_REPO),
        DeclareLaunchArgument("show", default_value="true"),
        gazebo,

        Node(   # Camera
            package="braccio_perception",
            executable="camera_node",
            output="screen",
            parameters=[{"source": ParameterValue(source, value_type=str)}]
        ),

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
            parameters=[{"repo_dir": ParameterValue(repo_dir, value_type=str), "sim": True}]),
    ])


