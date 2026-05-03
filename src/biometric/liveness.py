"""
Liveness detection module for anti-spoofing protection.

This module implements multiple liveness detection techniques to prevent
spoofing attacks using photos, videos, masks, or other fake presentations.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum
try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - runtime dependency guard
    ort = None  # type: ignore[assignment]

from core.logging import (get_logger, get_security_logger,
                            SecurityEventType, PerformanceTimer)

logger = get_logger(__name__)
security_logger = get_security_logger()


class LivenessMethod(Enum):
    """Types of liveness detection methods."""
    DEPTH_ANALYSIS = "depth_analysis"
    TEXTURE_ANALYSIS = "texture_analysis"
    MOTION_ANALYSIS = "motion_analysis"
    CHALLENGE_RESPONSE = "challenge_response"
    MULTIMODAL = "multimodal"


class SpoofingType(Enum):
    """Types of spoofing attacks."""
    PHOTO_ATTACK = "photo_attack"
    VIDEO_ATTACK = "video_attack"
    MASK_ATTACK = "mask_attack"
    SCREEN_ATTACK = "screen_attack"
    DEEPFAKE_ATTACK = "deepfake_attack"


@dataclass
class LivenessConfig:
    """Configuration for liveness detection."""
    enable_depth_analysis: bool = False
    enable_texture_analysis: bool = False
    enable_motion_analysis: bool = True
    enable_challenge_response: bool = False
    liveness_threshold: float = 0.2
    depth_threshold: float = 0.3
    texture_threshold: float = 0.6
    motion_threshold: float = 0.4
    challenge_timeout: float = 5.0
    frame_buffer_size: int = 10


@dataclass
class LivenessResult:
    """Result of liveness detection analysis."""
    is_live: bool
    confidence: float
    method_scores: Dict[str, float]
    spoofing_type: Optional[SpoofingType] = None
    processing_time: float = 0.0
    frame_count: int = 0


class LivenessDetector:
    """
    Multi-modal liveness detection system.

    Combines multiple detection methods:
    - 3D depth analysis using stereo cameras or structured light
    - Texture analysis for material detection
    - Motion analysis for natural eye/head movement
    - Challenge-response for interactive verification
    """

    def __init__(self, config: Optional[LivenessConfig] = None):
        """
        Initialize liveness detector.

        Args:
            config: Liveness detection configuration
        """
        self.config = config or LivenessConfig()
        
        # Frame buffer for temporal analysis
        self.frame_buffer: List[np.ndarray] = []
        self.depth_buffer: List[np.ndarray] = []

        # Model sessions for different detection methods
        self.texture_model = None
        self.depth_model = None

        # Motion analysis state
        self.previous_frame = None
        self.previous_points: Optional[np.ndarray] = None
        self.motion_history = []

        # Challenge-response state
        self.current_challenge = None
        self.challenge_start_time = None

        self._initialize_models()

        logger.info("LivenessDetector initialized")

    def _initialize_models(self):
        """Initialize AI models for liveness detection."""
        try:
            # Initialize texture analysis model
            if self.config.enable_texture_analysis:
                self._load_texture_model()

            # Initialize depth estimation model
            if self.config.enable_depth_analysis:
                self._load_depth_model()

        except Exception as e:
            logger.warning(f"Some liveness models failed to load: {e}")
            logger.info("Using rule-based liveness detection fallback")

    def _load_texture_model(self):
        """Load texture analysis model for material detection."""
        try:
            if ort is None:
                logger.warning(
                    "ONNX Runtime unavailable; texture analysis disabled")
                return

            model_path = "models/liveness/anti_spoof.onnx"

            # Check if model exists
            import os
            if not os.path.exists(model_path):
                logger.warning(f"Texture model not found: {model_path}")
                return

            providers = ['CPUExecutionProvider']
            try:
                if ort.get_device() == 'GPU':
                    providers = ['CUDAExecutionProvider',
                                 'CPUExecutionProvider']
            except (AttributeError, RuntimeError):
                pass

            self.texture_model = ort.InferenceSession(
                model_path, providers=providers)
            logger.debug("Texture analysis model loaded")

        except Exception as e:
            logger.error(f"Failed to load texture model: {e}")
            self.texture_model = None

    def _load_depth_model(self):
        """Load depth estimation model."""
        try:
            if ort is None:
                logger.warning(
                    "ONNX Runtime unavailable; depth model disabled")
                return

            model_path = "models/liveness/depth_estimation.onnx"

            # Check if model exists
            import os
            if not os.path.exists(model_path):
                logger.warning(f"Depth model not found: {model_path}")
                return

            providers = ['CPUExecutionProvider']
            try:
                if ort.get_device() == 'GPU':
                    providers = ['CUDAExecutionProvider',
                                 'CPUExecutionProvider']
            except (AttributeError, RuntimeError):
                pass

            self.depth_model = ort.InferenceSession(
                model_path, providers=providers)
            logger.debug("Depth estimation model loaded")

        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            self.depth_model = None

    def detect_liveness(self, frame: np.ndarray,
                        face_bbox: Tuple[int, int, int, int],
                        depth_frame: Optional[np.ndarray] = None) -> float:
        """
        Detect liveness in a single frame.

        Args:
            frame: RGB frame from camera
            face_bbox: Face bounding box (x, y, width, height)
            depth_frame: Optional depth frame from depth camera

        Returns:
            Liveness confidence score [0, 1]
        """
        try:
            with PerformanceTimer("liveness_detection", target_ms=100):
                # Update frame buffer
                self._update_frame_buffer(frame)

                # Extract face region
                x, y, w, h = face_bbox
                face_region = frame[y:y+h, x:x+w]

                if not self._is_valid_face_region(face_region):
                    logger.warning(
                        "Invalid face region extracted for liveness analysis")
                    return 0.0

                method_scores = {}

                # Depth analysis
                if self.config.enable_depth_analysis:
                    depth_score = self._analyze_depth(face_region, depth_frame)
                    method_scores["depth"] = depth_score

                # Texture analysis
                if self.config.enable_texture_analysis:
                    texture_score = self._analyze_texture(face_region)
                    method_scores["texture"] = texture_score

                # Motion analysis
                if self.config.enable_motion_analysis:
                    motion_score = self._analyze_motion(face_region)
                    method_scores["motion"] = motion_score

                # Combine scores
                final_score = self._combine_scores(method_scores)

                logger.debug(
                    f"Liveness scores: {method_scores}, "
                    f"final: {final_score:.3f}")
                return final_score

        except Exception as e:
            logger.error(f"Liveness detection failed: {e}")
            return 0.0  # Conservative: assume fake if detection fails

    def detect_liveness_sequence(
        self, frames: List[np.ndarray],
        face_bboxes: List[Tuple[int, int, int, int]],
        depth_frames: Optional[List[np.ndarray]] = None
    ) -> LivenessResult:
        """
        Detect liveness using a sequence of frames for temporal analysis.

        Args:
            frames: List of RGB frames
            face_bboxes: List of face bounding boxes for each frame
            depth_frames: Optional list of depth frames

        Returns:
            LivenessResult with detailed analysis
        """
        start_time = time.time()

        try:
            if len(frames) != len(face_bboxes):
                raise ValueError(
                    "Number of frames and bounding boxes must match")

            method_scores: Dict[str, List[float]] = {}
            if self.config.enable_depth_analysis:
                method_scores["depth"] = []
            if self.config.enable_texture_analysis:
                method_scores["texture"] = []
            if self.config.enable_motion_analysis:
                method_scores["motion"] = []

            # Process each frame
            for i, (frame, bbox) in enumerate(zip(frames, face_bboxes)):
                depth_frame = depth_frames[i] if depth_frames else None

                # Store individual method scores for temporal analysis
                # (This would be extracted from the individual frame analysis)
                if self.config.enable_depth_analysis and "depth" in method_scores:
                    method_scores["depth"].append(
                        self._analyze_depth_frame(frame, bbox, depth_frame))
                if self.config.enable_texture_analysis and "texture" in method_scores:
                    method_scores["texture"].append(
                        self._analyze_texture_frame(frame, bbox))
                if self.config.enable_motion_analysis and "motion" in method_scores:
                    method_scores["motion"].append(
                        self._analyze_motion_frame(frame, bbox, i > 0))

            # Temporal consistency analysis
            temporal_scores = self._analyze_temporal_consistency(method_scores)

            # Final decision
            mean_components = [np.mean(scores)
                               for scores in temporal_scores.values() if scores]
            final_confidence = float(
                np.mean(mean_components)) if mean_components else 0.0

            is_live = final_confidence >= self.config.liveness_threshold

            # Detect potential spoofing type
            spoofing_type = self._detect_spoofing_type(
                method_scores) if not is_live else None

            processing_time = time.time() - start_time

            result = LivenessResult(
                is_live=bool(is_live),
                confidence=float(final_confidence),
                method_scores={k: (float(v[-1]) if v else 0.0)
                               for k, v in temporal_scores.items()},
                spoofing_type=spoofing_type,
                processing_time=processing_time,
                frame_count=len(frames)
            )

            # Log security event
            event_type = (SecurityEventType.LIVENESS_DETECTION_PASS if is_live
                          else SecurityEventType.LIVENESS_DETECTION_FAIL)

            security_logger.log_security_event(
                event_type,
                additional_data={
                    "confidence": final_confidence,
                    "spoofing_type": (spoofing_type.value
                                      if spoofing_type else None),
                    "frame_count": len(frames)
                },
                success=bool(is_live)
            )

            return result

        except Exception as e:
            logger.error(f"Sequence liveness detection failed: {e}")
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                method_scores={},
                processing_time=time.time() - start_time,
                frame_count=len(frames)
            )

    def _analyze_depth(self, face_region: np.ndarray,
                       depth_frame: Optional[np.ndarray] = None) -> float:
        """Analyze depth information for 3D liveness."""
        try:
            if depth_frame is not None:
                # Use actual depth data from depth camera
                return self._analyze_real_depth(face_region, depth_frame)
            else:
                # Estimate depth using monocular depth estimation
                return self._estimate_depth(face_region)

        except Exception as e:
            logger.error(f"Depth analysis failed: {e}")
            return 0.5  # Neutral score if depth analysis fails

    def _analyze_real_depth(self, face_region: np.ndarray,
                            depth_frame: np.ndarray) -> float:
        """Analyze depth using real depth camera data."""
        try:
            # Extract depth values for face region
            # This assumes depth_frame is aligned with RGB frame
            h, w = face_region.shape[:2]
            face_depth = cv2.resize(depth_frame, (w, h))

            # Calculate depth statistics
            valid_depth = face_depth[face_depth > 0]
            if len(valid_depth) == 0:
                return 0.5

            mean_depth = float(np.mean(valid_depth))  # type: ignore
            depth_variance = float(np.var(valid_depth))  # type: ignore

            # Real faces should have:
            # - Reasonable depth range (not too close/far)
            # - Some depth variation (nose protruding, etc.)

            # Check if depth is in reasonable range (30cm to 1m)
            if mean_depth < 300 or mean_depth > 1000:  # mm
                return 0.1  # Likely fake

            # Check for depth variation
            if depth_variance < 100:  # Too flat
                return 0.2  # Likely photo/screen

            # Higher score for good depth characteristics
            depth_score = min(1.0, depth_variance / 1000.0 + 0.3)
            return depth_score

        except Exception as e:
            logger.error(f"Real depth analysis failed: {e}")
            return 0.5

    def _estimate_depth(self, face_region: np.ndarray) -> float:
        """Estimate depth using monocular depth estimation model."""
        try:
            if self.depth_model is None:
                # Fallback: simple heuristics
                return self._depth_heuristics(face_region)

            # Preprocess for depth model
            input_image = self._preprocess_for_depth(face_region)
            input_name = self.depth_model.get_inputs()[0].name

            # Run depth estimation
            outputs = self.depth_model.run(None, {input_name: input_image})
            depth_map = np.asarray(outputs[0])

            # Analyze depth map characteristics
            depth_score = self._analyze_depth_map(depth_map)
            return float(depth_score)

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return self._depth_heuristics(face_region)

    def _depth_heuristics(self, face_region: np.ndarray) -> float:
        """Simple depth analysis using image heuristics."""
        try:
            # Convert to grayscale
            gray = self._to_grayscale(face_region)

            # Calculate gradients (real faces have more complex gradients)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Real faces typically have more gradient variation
            gradient_variance = np.var(gradient_magnitude)

            # Normalize to [0, 1] range
            depth_score = min(1.0, gradient_variance / 10000.0)

            return float(depth_score)

        except Exception as e:
            logger.error(f"Depth heuristics failed: {e}")
            return 0.5

    def _analyze_texture(self, face_region: np.ndarray) -> float:
        """Analyze texture patterns to detect fake materials."""
        try:
            if self.texture_model is None:
                # Fallback: texture heuristics
                return self._texture_heuristics(face_region)

            # Preprocess for texture model
            input_image = self._preprocess_for_texture(face_region)
            input_name = self.texture_model.get_inputs()[0].name

            # Run texture analysis
            outputs = self.texture_model.run(None, {input_name: input_image})
            texture_array = np.asarray(outputs[0])
            texture_score = float(texture_array.reshape(-1)[0])

            return float(texture_score)

        except Exception as e:
            logger.error(f"Texture analysis failed: {e}")
            return self._texture_heuristics(face_region)

    def _texture_heuristics(self, face_region: np.ndarray) -> float:
        """Simple texture analysis using image statistics."""
        try:
            # Convert to different color spaces for analysis
            bgr_face = self._ensure_bgr(face_region)
            gray = self._to_grayscale(bgr_face)
            hsv = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2HSV)

            # Calculate Local Binary Pattern (LBP) for texture analysis
            lbp = self._calculate_lbp(gray)
            lbp_variance = np.var(lbp)

            # Real skin has characteristic texture patterns
            # Photos/screens often have different texture characteristics

            # Calculate color distribution in HSV
            hue_variance = float(np.var(hsv[:, :, 0]))  # type: ignore
            saturation_mean = float(np.mean(hsv[:, :, 1]))  # type: ignore

            # Combine features
            texture_score = (
                min(1.0, lbp_variance / 1000.0) * 0.5 +
                min(1.0, hue_variance / 500.0) * 0.3 +
                min(1.0, saturation_mean / 255.0) * 0.2
            )

            return float(texture_score)

        except Exception as e:
            logger.error(f"Texture heuristics failed: {e}")
            return 0.5

    def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern for texture analysis."""
        # Simplified LBP calculation
        rows, cols = gray_image.shape
        if rows < 3 or cols < 3:
            return np.zeros((1, 1), dtype=np.uint8)

        lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)

        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = gray_image[i, j]
                code = 0

                # Compare with 8 neighbors
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j],
                    gray_image[i-1, j+1], gray_image[i, j+1],
                    gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]

                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)

                lbp[i-1, j-1] = code

        return lbp

    def _analyze_motion(self, face_region: np.ndarray) -> float:
        """Analyze motion patterns for natural movement detection."""
        try:
            gray_current = self._to_grayscale(face_region)

            if self.previous_frame is None or self.previous_points is None:
                self.previous_frame = face_region.copy()
                self.previous_points = cv2.goodFeaturesToTrack(
                    gray_current,
                    maxCorners=200,
                    qualityLevel=0.01,
                    minDistance=7
                )
                return 0.5  # Neutral score for first frame

            gray_previous = self._to_grayscale(self.previous_frame)

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                gray_previous,
                gray_current,
                self.previous_points,
                self.previous_points
            )

            motion_score = 0.3
            if next_points is not None and status is not None:
                status_mask = status.reshape(-1) == 1
                prev_pts = self.previous_points.reshape(-1, 2)[status_mask]
                next_pts = next_points.reshape(-1, 2)[status_mask]

                if len(prev_pts) > 0 and len(next_pts) > 0:
                    motion_vectors = next_pts - prev_pts
                    motion_magnitude = np.linalg.norm(motion_vectors, axis=1)

                    motion_variance = float(np.var(motion_magnitude))
                    motion_mean = float(np.mean(motion_magnitude))

                    if motion_mean < 0.5:
                        motion_score = 0.3
                    elif motion_mean > 10:
                        motion_score = 0.4
                    else:
                        motion_score = float(
                            min(1.0, motion_variance / 5.0 + 0.5))

            # Update previous tracking state
            self.previous_frame = face_region.copy()
            self.previous_points = next_points if next_points is not None else None

            if self.previous_points is None:
                self.previous_points = cv2.goodFeaturesToTrack(
                    gray_current,
                    maxCorners=200,
                    qualityLevel=0.01,
                    minDistance=7
                )

            # Update motion history
            self.motion_history.append(float(motion_score))
            if len(self.motion_history) > 10:
                self.motion_history.pop(0)

            return float(np.mean(self.motion_history))

        except Exception as e:
            logger.error(f"Motion analysis failed: {e}")
            return 0.5

    def _combine_scores(self, method_scores: Dict[str, float]) -> float:
        """Combine scores from different liveness detection methods."""
        if not method_scores:
            return 0.0

        # Weighted combination based on method reliability
        weights = {
            "depth": 0.4,
            "texture": 0.35,
            "motion": 0.25
        }

        total_score = 0.0
        total_weight = 0.0

        for method, score in method_scores.items():
            if method in weights:
                total_score += score * weights[method]
                total_weight += weights[method]

        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0

    def _update_frame_buffer(self, frame: np.ndarray):
        """Update frame buffer for temporal analysis."""
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.config.frame_buffer_size:
            self.frame_buffer.pop(0)

    def _analyze_depth_frame(self, frame: np.ndarray,
                             bbox: Tuple[int, int, int, int],
                             depth_frame: Optional[np.ndarray]) -> float:
        """Analyze depth for a single frame."""
        x, y, w, h = bbox
        face_region = frame[y:y+h, x:x+w]
        if not self._is_valid_face_region(face_region):
            return 0.5
        return self._analyze_depth(face_region, depth_frame)

    def _analyze_texture_frame(self, frame: np.ndarray,
                               bbox: Tuple[int, int, int, int]) -> float:
        """Analyze texture for a single frame."""
        x, y, w, h = bbox
        face_region = frame[y:y+h, x:x+w]
        if not self._is_valid_face_region(face_region):
            return 0.5
        return self._analyze_texture(face_region)

    def _analyze_motion_frame(self, frame: np.ndarray,
                              bbox: Tuple[int, int, int, int],
                              has_previous: bool) -> float:
        """Analyze motion for a single frame."""
        if not has_previous:
            return 0.5

        x, y, w, h = bbox
        face_region = frame[y:y+h, x:x+w]
        if not self._is_valid_face_region(face_region):
            return 0.5
        return self._analyze_motion(face_region)

    def _analyze_temporal_consistency(self,
                                      method_scores: Dict[str, List[float]]
                                      ) -> Dict[str, List[float]]:
        """Analyze temporal consistency of liveness scores."""
        # For now, return the same scores
        # In a full implementation, this would smooth scores and
        # detect patterns
        return method_scores

    def _detect_spoofing_type(self, method_scores: Dict[str, List[float]]
                              ) -> Optional[SpoofingType]:
        """Detect the type of spoofing attack based on method scores."""
        depth_scores = method_scores.get("depth", [])
        texture_scores = method_scores.get("texture", [])
        motion_scores = method_scores.get("motion", [])

        avg_depth = np.mean(depth_scores) if depth_scores else 0.5
        avg_texture = np.mean(texture_scores) if texture_scores else 0.5
        avg_motion = np.mean(motion_scores) if motion_scores else 0.5

        # Heuristic spoofing type detection
        if avg_depth < 0.3 and avg_motion < 0.3:
            return SpoofingType.PHOTO_ATTACK
        elif avg_depth < 0.3 and avg_motion > 0.6:
            return SpoofingType.VIDEO_ATTACK
        elif avg_texture < 0.3:
            return SpoofingType.SCREEN_ATTACK
        else:
            return SpoofingType.MASK_ATTACK  # Default for unknown patterns

    def _preprocess_for_depth(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for depth estimation model."""
        image_bgr = self._ensure_bgr(image)
        # Resize and normalize for depth model
        resized = cv2.resize(image_bgr, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        input_image = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(input_image, axis=0)

    def _preprocess_for_texture(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for texture analysis model."""
        image_bgr = self._ensure_bgr(image)
        # Resize and normalize for texture model
        resized = cv2.resize(image_bgr, (112, 112))
        normalized = resized.astype(np.float32) / 255.0
        input_image = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(input_image, axis=0)

    def _analyze_depth_map(self, depth_map: np.ndarray) -> float:
        """Analyze depth map characteristics."""
        # Calculate depth statistics
        depth_variance = np.var(depth_map)
        depth_gradient = np.mean(np.abs(np.gradient(depth_map)))

        # Real faces should have reasonable depth variation
        score = min(1.0, float(depth_variance + depth_gradient) / 2.0)
        return float(score)

    def _is_valid_face_region(self, face_region: Optional[np.ndarray]) -> bool:
        """Validate that the face region is non-empty."""
        return (face_region is not None and face_region.size > 0 and
                face_region.shape[0] >= 3 and face_region.shape[1] >= 3)

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        """Ensure the image has three BGR channels."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.ndim == 3:
            if image.shape[2] == 3:
                return image
            if image.shape[2] == 1:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        raise ValueError("Unsupported image format for BGR conversion")

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert an image to grayscale safely."""
        if image.ndim == 2:
            return image

        if image.ndim == 3:
            if image.shape[2] == 1:
                return image[:, :, 0]
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        raise ValueError("Unsupported image format for grayscale conversion")


# Example usage and testing
if __name__ == "__main__":
    # Test liveness detection
    config = LivenessConfig(
        liveness_threshold=0.5,
        enable_depth_analysis=True,
        enable_texture_analysis=True,
        enable_motion_analysis=True
    )

    detector = LivenessDetector(config)

    print("Testing liveness detection...")

    # Create test data
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bbox = (100, 100, 200, 200)

    # Test single frame detection
    liveness_score = detector.detect_liveness(test_frame, test_bbox)
    print(f"Liveness score: {liveness_score:.3f}")

    # Test sequence detection
    test_frames = [test_frame] * 5
    test_bboxes: List[Tuple[int, int, int, int]] = [
        test_bbox for _ in range(5)]

    result = detector.detect_liveness_sequence(test_frames, test_bboxes)
    print(
        f"Sequence result: {result.is_live}, "
        f"confidence: {result.confidence:.3f}")

    if result.spoofing_type:
        print(f"Detected spoofing type: {result.spoofing_type.value}")

    print("Liveness detection test completed!")
