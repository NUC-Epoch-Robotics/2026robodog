import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import easyocr
import re
import numpy as np

class PuzzleSolverNode(Node):
    def __init__(self):
        super().__init__('puzzle_solver_node')
        
        # Declare parameters
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('result_topic', '/perception/puzzle_result')
        self.declare_parameter('enable_debug_window', False)
        
        image_topic = self.get_parameter('image_topic').value
        result_topic = self.get_parameter('result_topic').value
        self.debug_window = self.get_parameter('enable_debug_window').value

        # Initialize CV Bridge and OCR Reader
        # Using EasyOCR with English language since math equations are numbers and symbols
        self.get_logger().info("Initializing EasyOCR... (this may take a moment)")
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.bridge = CvBridge()

        # Publishers and Subscribers
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        self.result_pub = self.create_publisher(Int32, result_topic, 10)
        
        # State
        self.puzzle_solved = False
        self.get_logger().info("Puzzle Solver Node has been started. Waiting for images...")

    def solve_equation(self, text):
        """
        Extract math expression from raw text and safely evaluate it.
        Supported operators: +, -, *, /
        """
        # Remove spaces and find the mathematical expression
        clean_text = text.replace(" ", "").replace("x", "*").replace("X", "*")
        
        # Regular expression to match simple math equations: e.g., 12+34, 15*3-2
        match = re.search(r'(\d+[\+\-\*\/]\d+(?:[\+\-\*\/]\d+)*)', clean_text)
        
        if match:
            expression = match.group(1)
            try:
                # Safely evaluate the math expression
                # Note: eval is generally unsafe, but we restrict it to valid math characters via regex
                result = eval(expression)
                return int(result), expression
            except Exception as e:
                self.get_logger().error(f"Failed to evaluate '{expression}': {e}")
                return None, None
        return None, None

    def image_callback(self, msg):
        # If we already solved it, we might want to shut down or just ignore
        if self.puzzle_solved:
            return

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # Perform OCR on the image
        # detail=0 returns a list of strings instead of bounding boxes
        results = self.reader.readtext(cv_image, detail=0)
        
        for text in results:
            value, expression = self.solve_equation(text)
            if value is not None:
                high_score_zone = value % 4
                self.get_logger().info(f"Detected Expression: {expression} = {value}")
                self.get_logger().info(f"Target High-Score Zone: {high_score_zone}")
                
                # Publish the result
                msg_out = Int32()
                msg_out.data = high_score_zone
                self.result_pub.publish(msg_out)
                
                self.puzzle_solved = True
                self.get_logger().info("Puzzle solved successfully. Stopping further OCR processing to save compute.")
                break

        if self.debug_window:
            cv2.imshow("Puzzle Solver Debug", cv_image)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = PuzzleSolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down.\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
