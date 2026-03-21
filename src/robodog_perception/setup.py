from setuptools import setup

package_name = 'robodog_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'easyocr', 'opencv-python'],
    zip_safe=True,
    maintainer='robodog',
    maintainer_email='robodog@todo.todo',
    description='Perception module for ROBOCON auto dog',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'puzzle_solver_node = robodog_perception.puzzle_solver_node:main'
        ],
    },
)
