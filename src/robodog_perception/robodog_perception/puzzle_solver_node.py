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
        
        # --- New parameters for small distance & screen ---
        self.declare_parameter('camera_width', 1920)   # Use 1080p if possible for more clarity
        self.declare_parameter('camera_height', 1080)
        self.declare_parameter('zoom_factor', 2.0)     # Crop the center 1/2 of the image (ignores the dark room)
        self.declare_parameter('ocr_scale', 2.0)       # Enlarge the cropped text before OCR
        
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

        # Attempt to set high resolution
        cam_w = self.get_parameter('camera_width').value
        cam_h = self.get_parameter('camera_height').value
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

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

        h, w, _ = cv_image.shape
        self.frame_count += 1

        # --- Digital Zoom (Crop Center ROI) ---
        zoom = self.get_parameter('zoom_factor').value
        if zoom > 1.0:
            roi_w = int(w / zoom)
            roi_h = int(h / zoom)
            x1 = (w - roi_w) // 2
            y1 = (h - roi_h) // 2
            roi_image = cv_image[y1:y1+roi_h, x1:x1+roi_w]
        else:
            roi_image = cv_image.copy()
            x1, y1 = 0, 0
            roi_w, roi_h = w, h

        # Draw current confirmation status and ROI box on the original image
        status_text = f"Confirmations: {self.confirm_count}/{self.confirm_threshold}"
        cv2.putText(cv_image, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(cv_image, (x1, y1), (x1+roi_w, y1+roi_h), (0, 255, 0), 2)
        
        # Always display the camera feed smoothly at 30fps
        if self.debug_window and self.cap.isOpened():
            cv2.imshow("Puzzle Solver Camera Feed", cv_image)
            cv2.imshow("Cropped Screen ROI", roi_image)
            cv2.waitKey(1)

        # Only run expensive OCR every 15 frames (~2Hz)
        if self.frame_count % 15 != 0:
            return

        # --- Enlarge ROI before OCR ---
        # Tesseract performs poorly on very small text. Scaling up the cropped area helps.
        scale = self.get_parameter('ocr_scale').value
        if scale != 1.0:
            roi_image = cv2.resize(roi_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Preprocessing for Tesseract on the *cropped ROI*
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # By using Otsu on the cropped ROI (mostly the bright screen), we avoid the dark background 
        # of the room skewing the threshold.
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # CRITICAL: Tesseract expects BLACK text on WHITE background!
        # Your screen shows white text on a black background. We must invert it if necessary.
        num_white = cv2.countNonZero(thresh)
        num_black = thresh.size - num_white
        if num_white < num_black:  # If background is mostly black, invert the colors
            thresh = cv2.bitwise_not(thresh)

        # Perform OCR
        # Added 'x' and 'X' to whitelist because your screen uses 'x' for multiplication
        custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789+-*/=xX'
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
