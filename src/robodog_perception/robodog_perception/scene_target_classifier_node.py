from pathlib import Path
from typing import List, Optional, Tuple
from collections import Counter, deque
import time
import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge


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
        self.declare_parameter('input_mode', 'ros_image')
        self.declare_parameter('image_dir', '')
        self.declare_parameter('scan_period_sec', 1.0)
        self.declare_parameter('auto_shutdown_after_done', False)

        # Camera mode parameter kept for future reuse.
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('local_camera_id', 0)
        self.declare_parameter('auto_select_camera_id', True)
        self.declare_parameter('camera_probe_max_id', 10)
        self.declare_parameter('local_camera_width', 640)
        self.declare_parameter('local_camera_height', 480)
        self.declare_parameter('process_every_n_frames', 2)
        self.declare_parameter('consensus_window', 5)
        self.declare_parameter('consensus_min_votes', 3)
        self.declare_parameter('publish_cooldown_sec', 0.6)
        self.declare_parameter('no_frame_warn_sec', 2.0)

        self.declare_parameter('result_topic', '/perception/scene_target_result')
        self.declare_parameter('enable_debug_window', True)
        self.declare_parameter('min_confidence', 0.10)
        self.declare_parameter('always_publish_best', False)
        self.declare_parameter('min_roi_area_ratio', 0.01)
        self.declare_parameter('max_roi_area_ratio', 0.45)
        self.declare_parameter('relative_area_keep_ratio', 0.50)
        self.declare_parameter('min_print_score', 0.08)
        self.declare_parameter('min_white_bg_ratio', 0.32)
        self.declare_parameter('temporal_iou_weight', 0.22)
        self.declare_parameter('enable_center_fallback', False)
        self.declare_parameter('foreground_bottom_weight', 0.18)
        self.declare_parameter('use_relaxed_fallback_candidates', True)
        self.declare_parameter('enable_template_matching', True)
        self.declare_parameter('template_dir', '')
        self.declare_parameter('auto_template_from_image_dir', True)
        self.declare_parameter('templates_per_class', 3)
        self.declare_parameter('template_weight', 0.12)
        self.declare_parameter('template_instrument_boost_threshold', 0.55)
        self.declare_parameter('enable_prediction_guards', True)
        self.declare_parameter('instrument_min_blue_evi', 0.23)
        self.declare_parameter('instrument_min_blue_advantage', 0.02)
        self.declare_parameter('medicine_min_red_evi', 0.20)
        self.declare_parameter('medicine_blue_rescue_margin', 0.02)
        self.declare_parameter('save_debug_dir', '')

        self.input_mode = str(self.get_parameter('input_mode').value).strip().lower()
        self.image_dir = str(self.get_parameter('image_dir').value)
        self.scan_period_sec = float(self.get_parameter('scan_period_sec').value)
        self.auto_shutdown_after_done = bool(self.get_parameter('auto_shutdown_after_done').value)
        self.image_topic = self.get_parameter('image_topic').value
        self.local_camera_id = int(self.get_parameter('local_camera_id').value)
        self.auto_select_camera_id = bool(self.get_parameter('auto_select_camera_id').value)
        self.camera_probe_max_id = max(0, int(self.get_parameter('camera_probe_max_id').value))
        self.local_camera_width = int(self.get_parameter('local_camera_width').value)
        self.local_camera_height = int(self.get_parameter('local_camera_height').value)
        self.process_every_n_frames = max(1, int(self.get_parameter('process_every_n_frames').value))
        self.consensus_window = max(1, int(self.get_parameter('consensus_window').value))
        self.consensus_min_votes = max(1, int(self.get_parameter('consensus_min_votes').value))
        self.publish_cooldown_sec = float(self.get_parameter('publish_cooldown_sec').value)
        self.no_frame_warn_sec = float(self.get_parameter('no_frame_warn_sec').value)

        self.result_topic = self.get_parameter('result_topic').value
        self.enable_debug_window = self.get_parameter('enable_debug_window').value
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.always_publish_best = bool(self.get_parameter('always_publish_best').value)
        self.min_roi_area_ratio = float(self.get_parameter('min_roi_area_ratio').value)
        self.max_roi_area_ratio = float(self.get_parameter('max_roi_area_ratio').value)
        self.relative_area_keep_ratio = float(self.get_parameter('relative_area_keep_ratio').value)
        self.min_print_score = float(self.get_parameter('min_print_score').value)
        self.min_white_bg_ratio = float(self.get_parameter('min_white_bg_ratio').value)
        self.temporal_iou_weight = float(self.get_parameter('temporal_iou_weight').value)
        self.enable_center_fallback = bool(self.get_parameter('enable_center_fallback').value)
        self.foreground_bottom_weight = float(self.get_parameter('foreground_bottom_weight').value)
        self.use_relaxed_fallback_candidates = bool(self.get_parameter('use_relaxed_fallback_candidates').value)
        self.enable_template_matching = bool(self.get_parameter('enable_template_matching').value)
        self.template_dir = str(self.get_parameter('template_dir').value)
        self.auto_template_from_image_dir = bool(self.get_parameter('auto_template_from_image_dir').value)
        self.templates_per_class = int(self.get_parameter('templates_per_class').value)
        self.template_weight = float(self.get_parameter('template_weight').value)
        self.template_instrument_boost_threshold = float(
            self.get_parameter('template_instrument_boost_threshold').value
        )
        self.enable_prediction_guards = bool(self.get_parameter('enable_prediction_guards').value)
        self.instrument_min_blue_evi = float(self.get_parameter('instrument_min_blue_evi').value)
        self.instrument_min_blue_advantage = float(self.get_parameter('instrument_min_blue_advantage').value)
        self.medicine_min_red_evi = float(self.get_parameter('medicine_min_red_evi').value)
        self.medicine_blue_rescue_margin = float(self.get_parameter('medicine_blue_rescue_margin').value)
        self.save_debug_dir = str(self.get_parameter('save_debug_dir').value)

        if self.save_debug_dir:
            Path(self.save_debug_dir).mkdir(parents=True, exist_ok=True)

        self.result_pub = self.create_publisher(Int32, self.result_topic, 10)

        self.bridge = CvBridge()
        self.image_sub = None
        self.cap = None

        self.processed_files = set()
        self.empty_rounds = 0
        self.last_quad = None
        self.timer = None
        self.frame_counter = 0
        self.vote_history = deque(maxlen=self.consensus_window)
        self.last_publish_ts = 0.0
        self.last_frame_ts = time.monotonic()
        self.last_no_frame_warn_ts = 0.0
        self.realtime_timer = None
        self.heartbeat_timer = None

        self.class_names = {
            0: 'food',
            1: 'tools',
            2: 'instrument',
            3: 'medicine',
        }

        self.templates = {0: [], 1: [], 2: [], 3: []}
        self.initialize_templates()

        if self.input_mode == 'ros_image':
            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
            self.heartbeat_timer = self.create_timer(1.0, self.stream_heartbeat_timer_callback)
            self.get_logger().info(
                f'Realtime mode started. Subscribe: {self.image_topic}, publish: {self.result_topic}'
            )
        elif self.input_mode == 'local_camera':
            self.start_local_camera()
        else:
            # Offline batch path is intentionally disabled for competition realtime usage.
            # self.timer = self.create_timer(self.scan_period_sec, self.scan_and_process_images)
            self.get_logger().warn(
                'input_mode=image_dir is currently disabled. Set input_mode:=ros_image or local_camera.'
            )

    def scan_and_process_images(self) -> None:
        # Legacy offline batch path is disabled for realtime-only runs.
        return

        # ---- legacy code below (kept for rollback) ----
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

            if pred is None:
                self.get_logger().warn(
                    f'{img_path.name}: target not found.'
                )
                self.show_debug(frame, quad, pred, confidence)
                self.save_debug_image(frame, img_path.name, quad, pred, confidence)
                self.processed_files.add(str(img_path))
                continue

            if confidence < self.min_confidence and not self.always_publish_best:
                self.get_logger().warn(
                    f'{img_path.name}: target not confident enough, conf={confidence:.3f}'
                )
                self.show_debug(frame, quad, pred, confidence)
                self.save_debug_image(frame, img_path.name, quad, pred, confidence)
                self.processed_files.add(str(img_path))
                continue

            out = Int32()
            out.data = pred
            self.result_pub.publish(out)

            if confidence < self.min_confidence:
                self.get_logger().warn(
                    f'{img_path.name} -> low-confidence class={pred} ({self.class_names[pred]}), conf={confidence:.3f}'
                )
            else:
                self.get_logger().info(
                    f'{img_path.name} -> class={pred} ({self.class_names[pred]}), conf={confidence:.3f}'
                )

            self.show_debug(frame, quad, pred, confidence)
            self.save_debug_image(frame, img_path.name, quad, pred, confidence)
            self.processed_files.add(str(img_path))

    def classify_frame(self, frame: np.ndarray) -> Tuple[Optional[int], Optional[float], Optional[np.ndarray]]:
        candidates = self.extract_candidate_rois(frame, strict=True)
        if not candidates and self.use_relaxed_fallback_candidates:
            candidates = self.extract_candidate_rois(frame, strict=False)

        if not candidates:
            return None, None, None

        # Keep candidates close to the largest visible square target to suppress small side boxes.
        max_area_ratio = max(item[3] for item in candidates)
        area_threshold = max_area_ratio * max(0.0, min(1.0, self.relative_area_keep_ratio))
        filtered = [item for item in candidates if item[3] >= area_threshold]
        if not filtered:
            filtered = candidates

        best = None
        for quad, roi, roi_score, area_ratio in filtered:
            temporal_bonus = 0.0
            if self.last_quad is not None:
                temporal_bonus = self.quad_iou(np.array(quad), np.array(self.last_quad)) * self.temporal_iou_weight

            stabilized_roi_score = roi_score + temporal_bonus
            cls_idx, cls_conf, cls_scores = self.classify_roi(roi)
            # Favor both classification confidence and geometrically reliable ROI.
            fused_conf = 0.75 * cls_conf + 0.25 * stabilized_roi_score

            if best is None or fused_conf > best['confidence']:
                best = {
                    'class_idx': cls_idx,
                    'confidence': fused_conf,
                    'scores': cls_scores,
                    'quad': quad,
                    'area_ratio': area_ratio,
                }

        if best is None:
            return None, None, None

        self.last_quad = np.array(best['quad']).copy()
        return int(best['class_idx']), float(best['confidence']), best['quad']

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert ROS Image: {exc}')
            return

        self.last_frame_ts = time.monotonic()
        self.process_live_frame(frame, source='ros_image')

    def start_local_camera(self) -> None:
        self.log_linux_video_devices()
        available_ids = self.probe_local_camera_ids(self.camera_probe_max_id)
        if not available_ids:
            self.get_logger().error(
                'No available local camera IDs found. '
                'Close other apps that occupy camera and check camera permissions.'
            )
            return

        selected_id = self.local_camera_id
        if self.auto_select_camera_id and selected_id not in available_ids:
            self.get_logger().warn(
                f'Configured local_camera_id={selected_id} is unavailable. '
                f'Auto-selecting id={available_ids[0]} from {available_ids}.'
            )
            selected_id = available_ids[0]

        if selected_id not in available_ids:
            self.get_logger().error(
                f'Failed to open local camera id={selected_id}. Available IDs: {available_ids}. '
                'Please set local_camera_id to one of available IDs.'
            )
            return

        if os.name == 'posix':
            self.cap = cv2.VideoCapture(selected_id, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(selected_id)
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.local_camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.local_camera_height)

        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error(
                f'Failed to open local camera id={selected_id}. Available IDs: {available_ids}.'
            )
            return

        self.local_camera_id = selected_id

        self.realtime_timer = self.create_timer(0.033, self.local_camera_timer_callback)
        self.get_logger().info(
            f'Local camera opened: id={self.local_camera_id}, '
            f'requested={self.local_camera_width}x{self.local_camera_height}, publish={self.result_topic}. '
            f'Available IDs: {available_ids}'
        )

    def probe_local_camera_ids(self, max_id: int) -> List[int]:
        available: List[int] = []
        for cam_id in range(max_id + 1):
            if os.name == 'posix':
                cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(cam_id)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                continue

            ok, _ = cap.read()
            cap.release()
            if ok:
                available.append(cam_id)

        self.get_logger().info(f'Camera probe result (0..{max_id}): {available}')
        return available

    def log_linux_video_devices(self) -> None:
        if os.name != 'posix':
            return

        dev_nodes = sorted(str(p) for p in Path('/dev').glob('video*'))
        if dev_nodes:
            self.get_logger().info(f'Detected video device nodes: {dev_nodes}')
        else:
            self.get_logger().warn('No /dev/video* nodes detected on system.')

    def local_camera_timer_callback(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            now = time.monotonic()
            if now - self.last_no_frame_warn_ts >= 1.0:
                self.get_logger().warn('Local camera is opened but no frame received.')
                self.last_no_frame_warn_ts = now
            return

        self.last_frame_ts = time.monotonic()
        self.process_live_frame(frame, source='local_camera')

    def stream_heartbeat_timer_callback(self) -> None:
        if self.input_mode != 'ros_image':
            return
        now = time.monotonic()
        if now - self.last_frame_ts > self.no_frame_warn_sec:
            self.get_logger().warn(
                f'No frames from topic {self.image_topic} for {self.no_frame_warn_sec:.1f}s. '
                'Check camera node/topic.'
            )

    def process_live_frame(self, frame: np.ndarray, source: str) -> None:
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n_frames != 0:
            # Always preview current camera frame so user can confirm stream is alive.
            self.show_debug(frame, None, None, None, f'source={source} preview')
            return

        pred, confidence, quad = self.classify_frame(frame)

        if pred is not None and confidence is not None and confidence >= self.min_confidence:
            self.vote_history.append(pred)
        else:
            self.vote_history.append(-1)

        valid_votes = [v for v in self.vote_history if v >= 0]
        vote_text = 'vote=none'
        winner = None
        winner_count = 0
        if valid_votes:
            winner, winner_count = Counter(valid_votes).most_common(1)[0]
            vote_text = f'vote={winner} {winner_count}/{self.consensus_window}'

        if winner is not None and winner_count >= self.consensus_min_votes:
            now = time.monotonic()
            if now - self.last_publish_ts >= self.publish_cooldown_sec:
                out = Int32()
                out.data = int(winner)
                self.result_pub.publish(out)
                self.last_publish_ts = now
                self.get_logger().info(
                    f'Realtime publish class={winner} ({self.class_names[int(winner)]}), '
                    f'votes={winner_count}/{self.consensus_window}'
                )

        self.show_debug(frame, quad, pred, confidence, f'source={source} {vote_text}')

    def extract_candidate_rois(self, frame: np.ndarray, strict: bool = True) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        h, w = frame.shape[:2]
        min_area = max(1.0, self.min_roi_area_ratio * float(h * w))
        max_area = max(min_area * 1.5, self.max_roi_area_ratio * float(h * w))
        area_span = max(1e-6, self.max_roi_area_ratio - self.min_roi_area_ratio)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        norm = cv2.equalizeHist(blur)
        edges = cv2.Canny(norm, 45, 150)

        adaptive = cv2.adaptiveThreshold(
            norm,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            4,
        )
        adaptive = cv2.bitwise_not(adaptive)
        edges = cv2.bitwise_or(edges, adaptive)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw < 25 or rh < 25:
                continue

            ratio = max(rw, rh) / float(max(1.0, min(rw, rh)))
            if ratio > 1.45:
                continue

            rect_area = float(max(1.0, rw * rh))
            extent = area / rect_area
            if extent < 0.55:
                continue

            box = cv2.boxPoints(rect).astype(np.float32)
            ordered = self.order_points(box)

            warp_size = 220
            dst = np.array(
                [[0, 0], [warp_size - 1, 0], [warp_size - 1, warp_size - 1], [0, warp_size - 1]],
                dtype=np.float32,
            )
            mat = cv2.getPerspectiveTransform(ordered, dst)
            warped = cv2.warpPerspective(frame, mat, (warp_size, warp_size))

            # Prefer near-square, high-extent quads and valid print density.
            square_score = max(0.0, 1.0 - abs(1.0 - ratio))

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

            # White-board prior: target face is mostly white background with printed icon/text.
            icon_hsv = cv2.cvtColor(icon_region, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(icon_hsv, (0, 0, 125), (179, 70, 255))
            white_bg_ratio = float(cv2.countNonZero(white_mask)) / float(max(1, white_mask.size))

            # Area score without fixed-size preference (target distance can change a lot).
            area_ratio = area / float(h * w)
            area_score = max(0.0, min(1.0, (area_ratio - self.min_roi_area_ratio) / area_span))

            print_threshold = self.min_print_score if strict else (self.min_print_score * 0.55)
            white_threshold = self.min_white_bg_ratio if strict else (self.min_white_bg_ratio * 0.55)

            if print_score < print_threshold:
                continue

            if white_bg_ratio < white_threshold:
                continue

            # Foreground prior: active target is usually on ground/front area, not high in the background.
            quad_int = ordered.astype(np.int32)
            bx, by, bw, bh = cv2.boundingRect(quad_int)
            cy = by + 0.5 * bh
            bottom_score = max(0.0, min(1.0, (cy / float(max(1, h)) - 0.20) / 0.80))

            roi_score = (
                0.22 * square_score
                + 0.23 * extent
                + 0.20 * print_score
                + 0.15 * area_score
                + 0.10 * white_bg_ratio
                + self.foreground_bottom_weight * bottom_score
            )

            if not strict:
                # Relaxed mode downweights weak-looking rectangles but keeps them for "must detect" behavior.
                roi_score *= 0.88

            candidates.append((ordered.astype(int), warped, roi_score, area_ratio))

        if candidates:
            candidates.sort(key=lambda item: item[2], reverse=True)
            candidates = candidates[:8]

        # Optional fallback: when enabled, use center crop to avoid returning no target.
        if not candidates and self.enable_center_fallback:
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
                area_ratio = float((x2 - x1) * (y2 - y1)) / float(max(1, h * w))
                candidates.append((quad, roi, 0.02, area_ratio))

        return candidates

    def classify_roi(self, roi: np.ndarray):
        # Ignore bottom text area; focus on icon + upper cube faces.
        icon_region = roi[: int(roi.shape[0] * 0.80), :]
        icon_region = self.gray_world_white_balance(icon_region)

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
        green = cv2.inRange(hsv, (30, 35, 25), (105, 255, 255))
        # Blue is tuned using your measured color (~RGB 2,64,138): broaden around deep azure.
        blue = cv2.inRange(hsv, (86, 28, 22), (150, 255, 255))
        deep_blue = cv2.inRange(hsv, (98, 35, 18), (132, 255, 210))
        red1 = cv2.inRange(hsv, (0, 35, 25), (14, 255, 255))
        red2 = cv2.inRange(hsv, (160, 35, 25), (179, 255, 255))
        red = cv2.bitwise_or(red1, red2)

        # Tools class uses gray ink; remove bright white background from this mask.
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        gray_mask = ((sat < 45) & (val > 35) & (val < 195) & (ink_mask > 0)).astype(np.uint8) * 255

        green_ratio = float(cv2.countNonZero(cv2.bitwise_and(green, ink_mask))) / ink_pixels
        blue_ratio = float(cv2.countNonZero(cv2.bitwise_and(blue, ink_mask))) / ink_pixels
        deep_blue_ratio = float(cv2.countNonZero(cv2.bitwise_and(deep_blue, ink_mask))) / ink_pixels
        red_ratio = float(cv2.countNonZero(cv2.bitwise_and(red, ink_mask))) / ink_pixels
        gray_ratio = float(cv2.countNonZero(gray_mask)) / ink_pixels
        colorful_ratio = float(cv2.countNonZero(cv2.bitwise_or(cv2.bitwise_or(green, blue), red))) / float(max(1, total))

        # Channel-dominance features on ink pixels improve stability under tinted lighting.
        bgr = icon_region.astype(np.float32)
        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]
        ink_bool = ink_mask > 0
        if np.any(ink_bool):
            green_dom = float(np.mean((np.maximum(0.0, g - r) + 0.5 * np.maximum(0.0, g - b))[ink_bool])) / 255.0
            blue_dom = float(np.mean((np.maximum(0.0, b - r) + 0.5 * np.maximum(0.0, b - g))[ink_bool])) / 255.0
            red_dom = float(np.mean((np.maximum(0.0, r - g) + 0.5 * np.maximum(0.0, r - b))[ink_bool])) / 255.0
            chroma = (np.max(bgr, axis=2) - np.min(bgr, axis=2)) / 255.0
            low_chroma_ratio = float(np.mean((chroma < 0.12)[ink_bool]))
        else:
            green_dom = 0.0
            blue_dom = 0.0
            red_dom = 0.0
            low_chroma_ratio = 0.0

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

        green_evi = green_ratio + 0.7 * green_dom + 0.25 * green_ring
        blue_evi = blue_ratio + 0.45 * deep_blue_ratio + 0.7 * blue_dom + 0.25 * blue_ring
        red_evi = red_ratio + 0.7 * red_dom + 0.25 * red_ring

        scores = {
            0: 1.05 * green_evi + 0.06 * edge_ratio,
            1: 0.85 * gray_ratio + 0.55 * low_chroma_ratio + 0.05 * edge_ratio - 0.90 * max(green_evi, blue_evi, red_evi),
            2: 1.05 * blue_evi + 0.06 * edge_ratio,
            3: 1.05 * red_evi + 0.06 * edge_ratio,
        }

        t_scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        if self.enable_template_matching:
            t_scores = self.template_match_scores(roi)
            for cls_id in scores:
                scores[cls_id] += self.template_weight * t_scores.get(cls_id, 0.0)

            # Instrument-specific template gate.
            inst_t = t_scores.get(2, 0.0)
            other_t = max(t_scores.get(0, 0.0), t_scores.get(1, 0.0), t_scores.get(3, 0.0))
            if (
                inst_t > self.template_instrument_boost_threshold
                and inst_t > other_t + 0.06
                and blue_evi > max(green_evi, red_evi) + 0.01
            ):
                scores[2] += 0.12
                scores[1] -= 0.08

        # If colored evidence is strong, explicitly suppress tools fallback.
        if max(green_evi, blue_evi, red_evi) > 0.28:
            scores[1] -= 0.25

        # Strong color dominance should force non-tools classes.
        if colorful_ratio > 0.06 and max(green_evi, blue_evi, red_evi) > 0.20:
            scores[1] -= 0.20

        # Instrument-specific boost: when blue clearly dominates, bias to class 2.
        if blue_evi > max(green_evi, red_evi) + 0.05 and blue_evi > 0.16:
            scores[2] += 0.20
            scores[1] -= 0.10

        # Direct blue-vs-green disambiguation for instrument vs food.
        if blue_evi > green_evi + 0.04:
            scores[2] += 0.12
            scores[0] -= 0.08
        elif green_evi > blue_evi + 0.04:
            scores[0] += 0.12

        # If almost no print exists, reduce confidence for all classes.
        if ink_ratio < 0.02:
            for k in scores:
                scores[k] *= 0.6

        # Confidence combines margin and top score to avoid frequent "not confident" on hard frames.
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_idx, best_score = sorted_items[0]
        second_score = sorted_items[1][1]

        if self.enable_prediction_guards:
            # Guard 1: prevent food/tool from being aggressively over-called as instrument.
            if best_idx == 2:
                blue_ok = (
                    blue_evi >= self.instrument_min_blue_evi
                    and blue_evi >= (max(green_evi, red_evi) + self.instrument_min_blue_advantage)
                )
                if not blue_ok:
                    # Keep class, but force it to low-confidence so node emits WARN instead of wrong publish.
                    best_score *= 0.25

            # Guard 2: rescue blue-ish instrument from medicine misclassification.
            if best_idx == 3:
                red_weak = red_evi < self.medicine_min_red_evi
                blue_strong = blue_evi > red_evi + self.medicine_blue_rescue_margin
                inst_template_not_worse = t_scores.get(2, 0.0) >= (t_scores.get(3, 0.0) - 0.03)
                if red_weak and blue_strong and inst_template_not_worse:
                    best_idx = 2
                    best_score = scores[2]
                    second_score = max(scores[0], scores[1], scores[3])

        margin = max(0.0, best_score - second_score)
        top_norm = min(1.0, max(0.0, best_score) / 1.25)
        confidence = 0.60 * margin + 0.40 * top_norm

        return int(best_idx), float(confidence), scores

    def initialize_templates(self) -> None:
        if not self.enable_template_matching:
            return

        source_files = []

        if self.template_dir:
            tdir = Path(self.template_dir)
            if tdir.exists() and tdir.is_dir():
                source_files.extend(self.list_image_files(tdir))

        if (not source_files) and self.auto_template_from_image_dir and self.image_dir:
            idir = Path(self.image_dir)
            if idir.exists() and idir.is_dir():
                source_files.extend(self.list_image_files(idir))

        if not source_files:
            self.get_logger().info('Template matching enabled but no template source found.')
            return

        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for fp in source_files:
            cls_id = self.infer_class_from_filename(fp.name)
            if cls_id is None:
                continue
            if counts[cls_id] >= self.templates_per_class:
                continue

            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            cands = self.extract_candidate_rois(frame, strict=False)
            if not cands:
                continue
            cands.sort(key=lambda item: item[2], reverse=True)
            roi = cands[0][1]

            tmpl = self.make_template_feature(roi)
            if tmpl is None:
                continue

            self.templates[cls_id].append(tmpl)
            counts[cls_id] += 1

        total_templates = sum(len(v) for v in self.templates.values())
        if total_templates > 0:
            self.get_logger().info(f'Template matching loaded {total_templates} templates: {counts}')
        else:
            self.get_logger().info('Template matching enabled, but no valid templates were built.')

    def template_match_scores(self, roi: np.ndarray) -> dict:
        roi_feat = self.make_template_feature(roi)
        if roi_feat is None:
            return {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        out = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for cls_id, tmpls in self.templates.items():
            if not tmpls:
                continue
            best = -1.0
            for t in tmpls:
                score = cv2.matchTemplate(roi_feat, t, cv2.TM_CCOEFF_NORMED)[0, 0]
                if score > best:
                    best = score
            out[cls_id] = float(max(0.0, best))
        return out

    @staticmethod
    def make_template_feature(roi: np.ndarray):
        if roi is None or roi.size == 0:
            return None
        icon = roi[: int(roi.shape[0] * 0.80), :]
        gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(
            bw,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        feat = cv2.resize(bw, (96, 96), interpolation=cv2.INTER_AREA)
        return feat

    @staticmethod
    def infer_class_from_filename(name: str) -> Optional[int]:
        low = name.lower()
        if low.startswith('food'):
            return 0
        if low.startswith('tool'):
            return 1
        if low.startswith('instrument'):
            return 2
        if low.startswith('medicine'):
            return 3
        return None

    @staticmethod
    def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
        imgf = img.astype(np.float32)
        avg = np.mean(imgf.reshape(-1, 3), axis=0)
        mean_gray = float(np.mean(avg)) + 1e-6
        scale = mean_gray / (avg + 1e-6)
        balanced = imgf * scale.reshape(1, 1, 3)
        return np.clip(balanced, 0, 255).astype(np.uint8)

    @staticmethod
    def quad_iou(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
        ax1, ay1, aw, ah = cv2.boundingRect(quad_a.astype(np.int32))
        bx1, by1, bw, bh = cv2.boundingRect(quad_b.astype(np.int32))
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = float(inter_w * inter_h)

        area_a = float(max(1, aw * ah))
        area_b = float(max(1, bw * bh))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

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
        status_text: Optional[str] = None,
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

        if status_text:
            cv2.putText(vis, status_text, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 220, 0), 2)

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
        if hasattr(node, 'cap') and node.cap is not None and node.cap.isOpened():
            node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
