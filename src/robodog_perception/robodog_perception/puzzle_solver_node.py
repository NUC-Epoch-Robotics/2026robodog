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
        self.declare_parameter('camera_width', 960)   # Use 1080p if possible for more clarity
        self.declare_parameter('camera_height', 540)
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
        self.smoothed_bbox = None  # Add this to smooth the green box
        
        # Open Camera directly
        self.get_logger().info(f"Opening physical camera (ID: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # VERY IMPORTANT: Lower resolution to 640x480 to kill the lag
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera /dev/video{self.camera_id}. Check permissions or USB connection.")
            sys.exit(1)

        # We enforce 640x480 for extreme speed (no lag)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

        # ---------------------------------------------------------
        # OPTIMIZATION: ROI Extraction (Find the text block & Crop)
        # ---------------------------------------------------------
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur and binarize to find bright spots (the screen text)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # BUGFIX: Stop using Otsu for the whole room, it causes the threshold to jump randomly and the box to drift!
        # Instead, we clamp it to > 200, which guarantees we ONLY pick up the bright LED light from the monitor.
        _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # 2. Use Morphological Close to smear the letters together into one huge white block
        # A 40x10 rectangle kernel will connect characters horizontally
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 10))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 3. Find the contours of these huge blocks
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text = ""  # Initialize text to empty string to prevent UnboundLocalError

        if contours:
            # 4. Grab the largest block (assuming it's the puzzle screen)
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Reasonably large
                # 5. Get bounding box and add a little padding
                x, y, w, h = cv2.boundingRect(largest_contour)
                pad = 15
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(cv_image.shape[1] - x, w + 2*pad)
                h = min(cv_image.shape[0] - y, h + 2*pad)
                
                # OPTIMIZATION: Temporal Smoothing (EMA) to kill the green box jitter (lock on target)
                if self.smoothed_bbox is None:
                    self.smoothed_bbox = [x, y, w, h]
                else:
                    alpha = 0.1  # 10% new info, 90% old info (extremely stable)
                    self.smoothed_bbox[0] = int(alpha * x + (1 - alpha) * self.smoothed_bbox[0])
                    self.smoothed_bbox[1] = int(alpha * y + (1 - alpha) * self.smoothed_bbox[1])
                    self.smoothed_bbox[2] = int(alpha * w + (1 - alpha) * self.smoothed_bbox[2])
                    self.smoothed_bbox[3] = int(alpha * h + (1 - alpha) * self.smoothed_bbox[3])

                sx, sy, sw, sh = self.smoothed_bbox

                # Draw a green box on the video to show exactly what it's cutting out
                cv2.rectangle(cv_image, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                cv2.putText(cv_image, "ROI", (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 6. Actually CROP the image!
                roi_crop_gray = gray[sy:sy+sh, sx:sx+sw]

                # Draw current confirmation status
                status_text = f"Confirmations: {self.confirm_count}/{self.confirm_threshold}"
                cv2.putText(cv_image, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Always display the camera feed smoothly at 30fps
                if self.debug_window and self.cap.isOpened():
                    cv2.imshow("Puzzle Solver Camera Feed", cv_image)
                    cv2.waitKey(1)

                # Only run expensive OCR every 15 frames (~2Hz)
                if self.frame_count % 15 == 0:
                    # ---------------------------------------------------------
                    # Tesseract OCR on the tiny cropped image
                    # ---------------------------------------------------------
                    # Apply Otsu's thresholding to the tiny crop
                    _, thresh = cv2.threshold(roi_crop_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                    # Invert if the background is mostly black (white text)
                    num_white = cv2.countNonZero(thresh)
                    num_black = thresh.size - num_white
                    if num_white < num_black:
                        thresh = cv2.bitwise_not(thresh)

                    # Perform OCR (Because the image is tiny, this takes ~0.01 seconds instead of 1 second!)
                    # Whitelist ensures it ONLY guesses math symbols
                    custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789+-*/=xX'
                    text = pytesseract.image_to_string(thresh, config=custom_config)
            else:
                if self.debug_window and self.cap.isOpened():
                    cv2.imshow("Puzzle Solver Camera Feed", cv_image)
                    cv2.waitKey(1)
        else:
            if self.debug_window and self.cap.isOpened():
                cv2.imshow("Puzzle Solver Camera Feed", cv_image)
                cv2.waitKey(1)
        
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
