import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import cv2
import pytesseract
import re
import sys

class PuzzleSolverNode(Node):
    def __init__(self):
        super().__init__('puzzle_solver_node')
        
        # Declare parameters
        self.declare_parameter('result_topic', '/perception/puzzle_result')
        self.declare_parameter('enable_debug_window', True)
        self.declare_parameter('confirm_threshold', 3) # Number of consecutive identical answers required
        self.declare_parameter('camera_id', 0) # /dev/video0 by default
        
        self.result_topic = self.get_parameter('result_topic').value
        self.debug_window = self.get_parameter('enable_debug_window').value
        self.confirm_threshold = self.get_parameter('confirm_threshold').value
        self.camera_id = self.get_parameter('camera_id').value

        self.get_logger().info("Using Tesseract OCR (Offline/Lightweight).")

        # Result Publisher
        self.result_pub = self.create_publisher(Int32, self.result_topic, 10)
        
        # State tracking for multi-frame confirmation
        self.last_result = None
        self.confirm_count = 0
        self.frame_count = 0
        
        # Open Camera directly
        self.get_logger().info(f"Opening physical camera (ID: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera /dev/video{self.camera_id}. Check permissions or USB connection.")
            sys.exit(1)

        # Run timer at ~30Hz (0.033s) for smooth video display
        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("Puzzle Solver Started. Displaying camera feed...")

    def solve_equation(self, text):
        """
        Extract math expression from raw text and safely evaluate it.
        Supported operators: +, -, *, /
        """
        clean_text = text.replace(" ", "").replace("x", "*").replace("X", "*")
        
        # Regex to match simple math equations: e.g., 12+34, 15*3-2
        match = re.search(r'(\d+[\+\-\*\/]\d+(?:[\+\-\*\/]\d+)*)', clean_text)
        
        if match:
            expression = match.group(1)
            try:
                # Safely evaluate
                result = eval(expression)
                return int(result), expression
            except Exception as e:
                return None, None
        return None, None

    def timer_callback(self):
        ret, cv_image = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame from camera.")
            return

        self.frame_count += 1

        # Draw current confirmation status on the original image if debug window is on
        status_text = f"Confirmations: {self.confirm_count}/{self.confirm_threshold}"
        cv2.putText(cv_image, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Always display the camera feed smoothly at 30fps
        if self.debug_window and self.cap.isOpened():
            cv2.imshow("Puzzle Solver Camera Feed", cv_image)
            cv2.waitKey(1)

        # Only run expensive OCR every 15 frames (~2Hz)
        if self.frame_count % 15 != 0:
            return

        # Preprocessing for Tesseract
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Apply binary thresholding
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Perform OCR
        custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789+-*/='
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        results = [line.strip() for line in text.split('\n') if line.strip()]
        
        for matched_text in results:
            value, expression = self.solve_equation(matched_text)
            if value is not None:
                high_score_zone = value % 4
                
                self.get_logger().info(f"Detected: {expression} = {value} -> Zone {high_score_zone}")
                
                # Multi-frame confirmation logic
                if self.last_result == high_score_zone:
                    self.confirm_count += 1
                    self.get_logger().info(f"Target matches previous frame. Confidence: {self.confirm_count}/{self.confirm_threshold}")
                else:
                    self.get_logger().warn("Result changed or is new. Resetting confidence counter.")
                    self.last_result = high_score_zone
                    self.confirm_count = 1
                
                # Check if we have gathered enough consecutive confident results
                if self.confirm_count >= self.confirm_threshold:
                    self.get_logger().info("==================================================")
                    self.get_logger().info(f"[SUCCESS] High-Score Zone Confirmed as: {high_score_zone}")
                    self.get_logger().info(f"[SUCCESS] Mathematical derivation: {expression} = {value}")
                    self.get_logger().info("==================================================")
                    
                    # Publish the absolute confident result
                    msg_out = Int32()
                    msg_out.data = high_score_zone
                    self.result_pub.publish(msg_out)
                    
                    # Stop the main OCR timer
                    self.timer.cancel()

                    # Clean up and shutdown successfully
                    self.get_logger().info("Shutting down puzzle solver node successfully.")
                    self.create_timer(1.0, self._delayed_shutdown)
                    
                break # Only process the first valid equation found per frame

    def _delayed_shutdown(self):
        # Exit cleanly
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = PuzzleSolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down.\n")
    finally:
        # Failsafe cleanup
        if node.cap.isOpened():
            node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
