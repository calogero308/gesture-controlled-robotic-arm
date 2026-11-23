from setuptools import find_packages, setup

package_name = "braccio_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Calogero",
    maintainer_email="75779763+calogero308@users.noreply.github.com",
    description="Percezione: webcam e gesto stabile",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_node = braccio_perception.camera_node:main",
            "gesture_node = braccio_perception.gesture_node:main",
        ],
    },
)
