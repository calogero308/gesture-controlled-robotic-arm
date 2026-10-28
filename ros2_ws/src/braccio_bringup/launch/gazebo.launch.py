import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import Command    # Utile per effettuare la conversione da xacro a solo URDF
from launch_ros.parameter_descriptions import ParameterValue # Utile ad estrarre il testo dell'URDF


def generate_launch_description():
    description_pkg = get_package_share_directory('braccio_description')
    gz_ros_sim_path = get_package_share_directory('ros_gz_sim')
    ros_urdf_xacro_path = os.path.join(description_pkg, 'urdf', 'robotic_arm.urdf.xacro')
    ros_urdf_text = ParameterValue(Command(["xacro ", ros_urdf_xacro_path]), value_type=str)
    gz_launch_path = os.path.join(gz_ros_sim_path, 'launch', 'gz_sim.launch.py')

    return LaunchDescription([

        Node(   # Creazione del nodo publisher
            package="robot_state_publisher",    # Nome package
            executable="robot_state_publisher", # Nome eseguibile
            parameters=[{"robot_description": ros_urdf_text,    # Parametri 
                         "use_sim_time": True}],
            output="screen"     # Output
        ),

        IncludeLaunchDescription(   # Lancio di Gazebo con creazione di un mondo vuoto
            PythonLaunchDescriptionSource(gz_launch_path),
            launch_arguments={
                'gz_args': '-r empty.sdf',   # -r lancia subito la simulazione -> empty: mondo vuoto
                'on_exit_shutdown': 'True'
            }.items()
        ),

        Node(   # Inserimento del braccio dentro il mondo vuoto
            package="ros_gz_sim",
            executable="create",
            arguments=["-topic", "robot_description", "-name", "braccio"]
            # Arguments contiene descrizione del topic e il nome del robot dentro Gazebo 
        ),

        Node(   # Nodo per la gestione del tempo dentro ROS -> Bridge node dell'orologio   
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
            # Composizione di arguments: /clock è il topic, il tipo ROS è rosgraph_msgs/msg/Clock, il tipo Gazebo è
            # /gz.msgs.Clock, e il [ indica che il flusso va da Gazebo verso ROS
            output="screen"
        ),

        # I seguenti nodi non fanno altro che caricare e attivare i controllori
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster"]
            # Arguments contiene: Nome del controllore
        ),

        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["arm_position_controller"]
            # Arguments contiene: Nome del controllore
        )

    ])