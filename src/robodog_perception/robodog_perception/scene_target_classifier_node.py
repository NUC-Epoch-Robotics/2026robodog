from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

# Camera mode imports are intentionally commented for now (offline image mode).
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge


class SceneTargetClassifierNode(Node):
    """Classify scene target into index 0..3 using traditional CV only.

    Class mapping (left to right in your reference):
    0 -> food (green)
    1 -> tools (gray)
    2 -> instrument (blue)
    3 -> medicine (red)
    """

    def __init__(self) -> None:
        super().__init__('scene_target_classifier_node')

        # Offline image classification parameters.
        self.declare_parameter('image_dir', '')
        self.declare_parameter('scan_period_sec', 1.0)
        self.declare_parameter('auto_shutdown_after_done', False)

        # Camera mode parameter kept for future reuse.
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('result_topic', '/perception/scene_target_result')
        self.declare_parameter('enable_debug_window', True)
        self.declare_parameter('min_confidence', 0.08)
        self.declare_parameter('min_roi_area_ratio', 0.01)
        self.declare_parameter('max_roi_area_ratio', 0.22)
        self.declare_parameter('save_debug_dir', '')

        self.image_dir = str(self.get_parameter('image_dir').value)
        self.scan_period_sec = float(self.get_parameter('scan_period_sec').value)
        self.auto_shutdown_after_done = bool(self.get_parameter('auto_shutdown_after_done').value)
        self.image_topic = self.get_parameter('image_topic').value
        self.result_topic = self.get_parameter('result_topic').value
        self.enable_debug_window = self.get_parameter('enable_debug_window').value
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.min_roi_area_ratio = float(self.get_parameter('min_roi_area_ratio').value)
        self.max_roi_area_ratio = float(self.get_parameter('max_roi_area_ratio').value)
        self.save_debug_dir = str(self.get_parameter('save_debug_dir').value)

        if self.save_debug_dir:
            Path(self.save_debug_dir).mkdir(parents=True, exist_ok=True)

        self.result_pub = self.create_publisher(Int32, self.result_topic, 10)

        # Camera mode subscriber kept as comment for future switch-back.
        # self.bridge = CvBridge()
        # self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)

        self.processed_files = set()
        self.empty_rounds = 0
        self.timer = self.create_timer(self.scan_period_sec, self.scan_and_process_images)

        self.class_names = {
            0: 'food',
            1: 'tools',
            2: 'instrument',
            3: 'medicine',
        }

        if not self.image_dir:
            self.get_logger().warn('image_dir is empty. Please set image_dir parameter to your picture folder.')
        else:
            self.get_logger().info(
                f'Offline image mode started. Scan dir: {self.image_dir}, publish: {self.result_topic}'
            )

    def scan_and_process_images(self) -> None:
        if not self.image_dir:
            return

        image_dir_path = Path(self.image_dir)
        if not image_dir_path.exists() or not image_dir_path.is_dir():
            self.get_logger().warn(f'image_dir not found or not a directory: {self.image_dir}')
            return

        image_paths = self.list_image_files(image_dir_path)
        pending = [p for p in image_paths if str(p) not in self.processed_files]

        if not pending:
            self.empty_rounds += 1
            if self.empty_rounds % 5 == 0:
                self.get_logger().info('No new images found. Waiting for new files...')
            if self.auto_shutdown_after_done and self.processed_files:
                self.get_logger().info('All images processed. Auto shutdown enabled, exiting node.')
                self.destroy_node()
                rclpy.shutdown()
            return

        self.empty_rounds = 0

        for img_path in pending:
            frame = cv2.imread(str(img_path))
            if frame is None:
                self.get_logger().warn(f'Failed to read image: {img_path.name}')
                self.processed_files.add(str(img_path))
                continue

            pred, confidence, quad = self.classify_frame(frame)

            if pred is None or confidence < self.min_confidence:
                self.get_logger().warn(
                    f'{img_path.name}: target not confident enough, conf={confidence if confidence is not None else 0.0:.3f}'
                )
                self.show_debug(frame, quad, pred, confidence)
                self.save_debug_image(frame, img_path.name, quad, pred, confidence)
                self.processed_files.add(str(img_path))
                continue

            out = Int32()
            out.data = pred
            self.result_pub.publish(out)

            self.get_logger().info(
                f'{img_path.name} -> class={pred} ({self.class_names[pred]}), conf={confidence:.3f}'
            )

            self.show_debug(frame, quad, pred, confidence)
            self.save_debug_image(frame, img_path.name, quad, pred, confidence)
            self.processed_files.add(str(img_path))

    def classify_frame(self, frame: np.ndarray) -> Tuple[Optional[int], Optional[float], Optional[np.ndarray]]:
        candidates = self.extract_candidate_rois(frame)

        if not candidates:
            return None, None, None

        best = None
        for quad, roi, roi_score in candidates:
            cls_idx, cls_conf, cls_scores = self.classify_roi(roi)
            fused_conf = 0.8 * cls_conf + 0.2 * roi_score

            if best is None or fused_conf > best['confidence']:
                best = {
                    'class_idx': cls_idx,
                    'confidence': fused_conf,
                    'scores': cls_scores,
                    'quad': quad,
                }

        if best is None:
            return None, None, None

        return int(best['class_idx']), float(best['confidence']), best['quad']

    # Camera callback reserved for future camera mode.
    # def image_callback(self, msg: Image) -> None:
    #     try:
    #         frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #     except Exception as exc:
    #         self.get_logger().error(f'Failed to convert ROS Image: {exc}')
    #         return
    #
    #     pred, confidence, quad = self.classify_frame(frame)
    #     if pred is None or confidence is None:
    #         self.show_debug(frame, None, None, None)
    #         return
    #
    #     out = Int32()
    #     out.data = pred
    #     self.result_pub.publish(out)
    #     self.show_debug(frame, quad, pred, confidence)

    def extract_candidate_rois(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        h, w = frame.shape[:2]
        min_area = max(1.0, self.min_roi_area_ratio * float(h * w))
        max_area = max(min_area * 1.5, self.max_roi_area_ratio * float(h * w))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 180)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            quad = approx.reshape(4, 2).astype(np.float32)
            ordered = self.order_points(quad)

            warp_size = 220
            dst = np.array(
                [[0, 0], [warp_size - 1, 0], [warp_size - 1, warp_size - 1], [0, warp_size - 1]],
                dtype=np.float32,
            )
            mat = cv2.getPerspectiveTransform(ordered, dst)
            warped = cv2.warpPerspective(frame, mat, (warp_size, warp_size))

            # Prefer near-square, larger quads.
            x, y, bw, bh = cv2.boundingRect(approx)
            ratio = bw / float(max(1, bh))
            square_score = max(0.0, 1.0 - abs(1.0 - ratio))

            # Center prior: target is usually near image center area in robot run scenes.
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            dx = abs(cx - (w * 0.5)) / max(1.0, w * 0.5)
            dy = abs(cy - (h * 0.6)) / max(1.0, h * 0.6)
            center_score = max(0.0, 1.0 - 0.7 * dx - 0.3 * dy)

            # Print density: valid cube face usually contains icon + Chinese text strokes.
            icon_region = warped[: int(warped.shape[0] * 0.9), :]
            icon_gray = cv2.cvtColor(icon_region, cv2.COLOR_BGR2GRAY)
            _, ink_mask = cv2.threshold(icon_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            ink_mask = cv2.morphologyEx(
                ink_mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                iterations=1,
            )
            ink_ratio = float(cv2.countNonZero(ink_mask)) / float(max(1, ink_mask.size))
            print_score = min(1.0, ink_ratio / 0.22)

            # Area preference around medium-size target; suppress giant wrong quads.
            area_ratio = area / float(h * w)
            area_pref = max(0.0, 1.0 - abs(area_ratio - 0.10) / 0.10)

            roi_score = 0.35 * square_score + 0.25 * center_score + 0.25 * area_pref + 0.15 * print_score

            candidates.append((ordered.astype(int), warped, roi_score))

        # Fallback: if no square-like contour is found, use a smaller center crop.
        if not candidates:
            size = int(min(h, w) * 0.35)
            cx, cy = w // 2, h // 2
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, x1 + size)
            y2 = min(h, y1 + size)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=int)
                roi = cv2.resize(roi, (220, 220), interpolation=cv2.INTER_AREA)
                candidates.append((quad, roi, 0.10))

        return candidates

    def classify_roi(self, roi: np.ndarray):
        # Ignore bottom text area; focus on icon + upper cube faces.
        icon_region = roi[: int(roi.shape[0] * 0.80), :]

        hsv = cv2.cvtColor(icon_region, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(icon_region, cv2.COLOR_BGR2GRAY)

        # Build an ink mask so white background does not dominate statistics.
        _, ink_mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        ink_mask = cv2.morphologyEx(
            ink_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        ink_pixels = float(max(1, cv2.countNonZero(ink_mask)))
        total = float(icon_region.shape[0] * icon_region.shape[1])
        ink_ratio = ink_pixels / total

        # Color masks in HSV.
        green = cv2.inRange(hsv, (35, 50, 40), (90, 255, 255))
        blue = cv2.inRange(hsv, (90, 50, 40), (135, 255, 255))
        red1 = cv2.inRange(hsv, (0, 50, 40), (12, 255, 255))
        red2 = cv2.inRange(hsv, (165, 50, 40), (179, 255, 255))
        red = cv2.bitwise_or(red1, red2)

        # Tools class uses gray ink; remove bright white background from this mask.
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        gray_mask = ((sat < 45) & (val > 35) & (val < 195) & (ink_mask > 0)).astype(np.uint8) * 255

        green_ratio = float(cv2.countNonZero(cv2.bitwise_and(green, ink_mask))) / ink_pixels
        blue_ratio = float(cv2.countNonZero(cv2.bitwise_and(blue, ink_mask))) / ink_pixels
        red_ratio = float(cv2.countNonZero(cv2.bitwise_and(red, ink_mask))) / ink_pixels
        gray_ratio = float(cv2.countNonZero(gray_mask)) / ink_pixels

        # Edge density helps avoid over-trusting weak color cues on blank regions.
        edge = cv2.Canny(gray_img, 80, 180)
        edge_ratio = float(cv2.countNonZero(cv2.bitwise_and(edge, ink_mask))) / ink_pixels

        # Extra cue: class-colored border often appears near ROI edges for 0/2/3.
        hh, ww = icon_region.shape[:2]
        band = max(4, int(min(hh, ww) * 0.08))
        ring_mask = np.zeros((hh, ww), dtype=np.uint8)
        ring_mask[:band, :] = 255
        ring_mask[-band:, :] = 255
        ring_mask[:, :band] = 255
        ring_mask[:, -band:] = 255
        ring_total = float(max(1, cv2.countNonZero(ring_mask)))
        green_ring = float(cv2.countNonZero(cv2.bitwise_and(green, ring_mask))) / ring_total
        blue_ring = float(cv2.countNonZero(cv2.bitwise_and(blue, ring_mask))) / ring_total
        red_ring = float(cv2.countNonZero(cv2.bitwise_and(red, ring_mask))) / ring_total

        scores = {
            0: 1.30 * green_ratio + 0.20 * green_ring + 0.08 * edge_ratio,
            1: 1.10 * gray_ratio + 0.10 * edge_ratio - 0.75 * max(green_ratio, blue_ratio, red_ratio),
            2: 1.30 * blue_ratio + 0.20 * blue_ring + 0.08 * edge_ratio,
            3: 1.30 * red_ratio + 0.20 * red_ring + 0.08 * edge_ratio,
        }

        # If colored evidence is strong, explicitly suppress tools fallback.
        max_color = max(green_ratio, blue_ratio, red_ratio)
        if max_color > 0.22 or (green_ring > 0.05 or blue_ring > 0.05 or red_ring > 0.05):
            scores[1] -= 0.35

        # If almost no print exists, reduce confidence for all classes.
        if ink_ratio < 0.02:
            for k in scores:
                scores[k] *= 0.6

        # Use margin between top-1 and top-2 as confidence.
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_idx, best_score = sorted_items[0]
        second_score = sorted_items[1][1]
        confidence = max(0.0, best_score - second_score)

        return int(best_idx), float(confidence), scores

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        # Return in order: top-left, top-right, bottom-right, bottom-left.
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)

        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(diff)]
        ordered[3] = pts[np.argmax(diff)]
        return ordered

    def show_debug(
        self,
        frame: np.ndarray,
        quad: Optional[np.ndarray],
        pred: Optional[int],
        confidence: Optional[float],
    ) -> None:
        if not self.enable_debug_window:
            return

        vis = frame.copy()
        if quad is not None:
            q = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [q], True, (0, 255, 0), 2)

        if pred is not None and confidence is not None:
            label = f'pred={pred} {self.class_names[pred]} conf={confidence:.3f}'
            cv2.putText(vis, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

        cv2.imshow('Scene Target Classifier', vis)
        cv2.waitKey(1)

    def save_debug_image(
        self,
        frame: np.ndarray,
        image_name: str,
        quad: Optional[np.ndarray],
        pred: Optional[int],
        confidence: Optional[float],
    ) -> None:
        if not self.save_debug_dir:
            return

        vis = frame.copy()
        if quad is not None:
            q = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [q], True, (0, 255, 0), 2)

        if pred is not None and confidence is not None:
            label = f'pred={pred} {self.class_names[pred]} conf={confidence:.3f}'
            cv2.putText(vis, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

        out_path = Path(self.save_debug_dir) / f'pred_{Path(image_name).stem}.jpg'
        cv2.imwrite(str(out_path), vis)

    @staticmethod
    def list_image_files(image_dir: Path) -> List[Path]:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        return sorted(files, key=lambda p: p.name.lower())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SceneTargetClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
