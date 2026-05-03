"""
Authentication engine for real-time biometric verification.

This module handles the complete authentication pipeline optimized for
sub-500ms latency with high accuracy and security.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from core.logging import (get_logger, get_security_logger,
                            SecurityEventType, PerformanceTimer)
from core.exceptions import CameraError
from security.encryption import SecureTemplateStorage
from biometric.face_detection import FaceDetector
from biometric.feature_extraction import FeatureExtractor
from biometric.quality_assessment import QualityAssessment
from biometric.liveness import LivenessDetector

logger = get_logger(__name__)
security_logger = get_security_logger()


class AuthenticationResult(Enum):
    """Authentication result types."""
    SUCCESS = "success"
    REJECTED = "rejected"
    LIVENESS_FAILED = "liveness_failed"
    QUALITY_INSUFFICIENT = "quality_insufficient"
    NO_FACE_DETECTED = "no_face_detected"
    TEMPLATE_NOT_FOUND = "template_not_found"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class AuthenticationConfig:
    """Configuration for authentication engine."""
    camera_id: int = 0
    target_width: int = 640
    target_height: int = 480
    fps: int = 30
    similarity_threshold: float = 0.7
    quality_threshold: float = 0.6
    max_authentication_time: int = 10  # seconds
    enable_liveness_detection: bool = True
    enable_adaptive_threshold: bool = True
    max_failed_attempts: int = 3
    lockout_duration: int = 300  # seconds
    performance_target_ms: float = 500.0


@dataclass
class AuthenticationAttempt:
    """Single authentication attempt data."""
    user_id: str
    timestamp: float
    image: np.ndarray
    face_bbox: Tuple[int, int, int, int]
    quality_score: float
    features: np.ndarray
    similarity_score: float
    liveness_score: Optional[float] = None
    result: AuthenticationResult = AuthenticationResult.ERROR


@dataclass
class AuthenticationResponse:
    """Authentication response data."""
    success: bool
    user_id: Optional[str] = None
    confidence: float = 0.0
    result: AuthenticationResult = AuthenticationResult.ERROR
    processing_time: float = 0.0
    quality_score: float = 0.0
    liveness_score: Optional[float] = None
    error_message: Optional[str] = None


class AuthenticationEngine:
    """
    High-performance biometric authentication engine.

    Features:
    - Sub-500ms authentication latency
    - Adaptive thresholding based on user history
    - Multi-threaded processing pipeline
    - Comprehensive security event logging
    - Anti-spoofing integration
    """

    def __init__(self, config: Optional[AuthenticationConfig] = None):
        """
        Initialize authentication engine.

        Args:
            config: Authentication configuration
        """
        self.config = config or AuthenticationConfig()

        # Initialize components
        self.face_detector = FaceDetector()
        self.feature_extractor = FeatureExtractor()
        self.quality_assessor = QualityAssessment()
        self.liveness_detector = LivenessDetector(
        ) if self.config.enable_liveness_detection else None
        self.template_storage = SecureTemplateStorage()

        # Camera and processing state
        self.camera = None
        self.is_authenticating = False
        self.processing_thread = None

        # User-specific adaptive thresholds
        self.user_thresholds: Dict[str, float] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.lockout_until: Dict[str, float] = {}

        # Performance optimization
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        logger.info("AuthenticationEngine initialized")

    def authenticate_user(self, user_id: str, password: str,
                          timeout: Optional[float] = None
                          ) -> AuthenticationResponse:
        """
        Authenticate a specific user with biometric data.

        Args:
            user_id: User identifier to authenticate
            password: Template decryption password
            timeout: Authentication timeout in seconds

        Returns:
            AuthenticationResponse with result and details
        """
        start_time = time.time()
        timeout = timeout or self.config.max_authentication_time

        try:
            # Check if user is locked out
            if self._is_user_locked_out(user_id):
                return AuthenticationResponse(
                    success=False,
                    user_id=user_id,
                    result=AuthenticationResult.REJECTED,
                    error_message="User account temporarily locked"
                )

            logger.info(f"Starting authentication for user: {user_id}")

            # Load user template
            with PerformanceTimer("template_loading"):
                template = self._load_user_template(user_id, password)
                if template is None:
                    return AuthenticationResponse(
                        success=False,
                        user_id=user_id,
                        result=AuthenticationResult.TEMPLATE_NOT_FOUND,
                        error_message="User template not found"
                    )

            # Initialize camera
            with PerformanceTimer("camera_initialization"):
                self._initialize_camera()

            # Perform authentication
            response = self._perform_authentication(
                user_id, template, start_time, timeout)

            # Update user statistics
            self._update_user_statistics(user_id, response.success)

            # Log security event
            security_logger.log_security_event(
                SecurityEventType.AUTHENTICATION_SUCCESS if response.success
                else SecurityEventType.AUTHENTICATION_FAILURE,
                user_id=user_id,
                additional_data={
                    "confidence": response.confidence,
                    "processing_time": response.processing_time,
                    "result": response.result.value
                },
                success=response.success
            )

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            logger.error(f"Authentication failed for user {user_id}: {e}")

            response = AuthenticationResponse(
                success=False,
                user_id=user_id,
                result=AuthenticationResult.ERROR,
                processing_time=processing_time,
                error_message=error_msg
            )

            security_logger.log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                user_id=user_id,
                additional_data={"error": error_msg},
                success=False
            )

            return response

        finally:
            self._cleanup_camera()

    def authenticate_user_from_frames(
        self, user_id: str, password: str, frames: List[np.ndarray]
    ) -> AuthenticationResponse:
        """Authenticate using pre-captured frames (no camera access)."""
        start_time = time.time()

        try:
            if not frames:
                return AuthenticationResponse(
                    success=False,
                    user_id=user_id,
                    result=AuthenticationResult.ERROR,
                    error_message="No frames provided",
                )

            if self._is_user_locked_out(user_id):
                return AuthenticationResponse(
                    success=False,
                    user_id=user_id,
                    result=AuthenticationResult.REJECTED,
                    error_message="User account temporarily locked",
                )

            template = self._load_user_template(user_id, password)
            if template is None:
                return AuthenticationResponse(
                    success=False,
                    user_id=user_id,
                    result=AuthenticationResult.TEMPLATE_NOT_FOUND,
                    error_message="User template not found",
                )

            threshold = self._get_user_threshold(user_id)
            best_similarity = 0.0
            best_quality = 0.0
            best_liveness: Optional[float] = None
            processed_faces = 0

            for frame in frames:
                faces = self.face_detector.detect_faces(frame)
                if not faces:
                    continue

                face_bbox = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face_bbox
                face_image = frame[y:y+h, x:x+w]
                if face_image.size == 0:
                    continue

                quality_score = self.quality_assessor.assess_quality(face_image)
                if quality_score < self.config.quality_threshold:
                    continue

                liveness_score = None
                if self.liveness_detector:
                    liveness_score = self.liveness_detector.detect_liveness(
                        frame, face_bbox
                    )
                    if liveness_score < 0.5:
                        continue

                features = self.feature_extractor.extract_features(face_image)
                if features is None:
                    continue

                processed_faces += 1
                similarity = self.feature_extractor.compute_similarity(
                    features, template
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_quality = quality_score
                    best_liveness = liveness_score

                if similarity >= threshold:
                    response = AuthenticationResponse(
                        success=True,
                        user_id=user_id,
                        confidence=similarity,
                        result=AuthenticationResult.SUCCESS,
                        processing_time=time.time() - start_time,
                        quality_score=quality_score,
                        liveness_score=liveness_score,
                    )
                    self._update_user_statistics(user_id, True)
                    return response

            processing_time = time.time() - start_time
            if processed_faces == 0:
                result = AuthenticationResult.NO_FACE_DETECTED
                error_message = "No valid face found in provided frames"
            else:
                result = AuthenticationResult.REJECTED
                error_message = "Authentication threshold not met"

            self._update_user_statistics(user_id, False)
            return AuthenticationResponse(
                success=False,
                user_id=user_id,
                confidence=best_similarity,
                result=result,
                processing_time=processing_time,
                quality_score=best_quality,
                liveness_score=best_liveness,
                error_message=error_message,
            )

        except Exception as e:
            logger.error(f"Frame-based authentication failed for user {user_id}: {e}")
            return AuthenticationResponse(
                success=False,
                user_id=user_id,
                result=AuthenticationResult.ERROR,
                processing_time=time.time() - start_time,
                error_message=str(e),
            )

    def authenticate_any_user(self, enrolled_users: List[str],
                              passwords: Dict[str, str],
                              timeout: Optional[float] = None
                              ) -> AuthenticationResponse:
        """
        Authenticate against multiple enrolled users (1:N matching).

        Args:
            enrolled_users: List of enrolled user IDs
            passwords: Dictionary mapping user IDs to passwords
            timeout: Authentication timeout in seconds

        Returns:
            AuthenticationResponse with best match
        """
        start_time = time.time()
        timeout = timeout or self.config.max_authentication_time

        try:
            logger.info(
                f"Starting 1:N authentication for {len(enrolled_users)} users")

            # Load all user templates
            templates = {}
            for user_id in enrolled_users:
                if user_id in passwords:
                    template = self._load_user_template(
                        user_id, passwords[user_id])
                    if template is not None:
                        templates[user_id] = template

            if not templates:
                return AuthenticationResponse(
                    success=False,
                    result=AuthenticationResult.TEMPLATE_NOT_FOUND,
                    error_message="No valid templates found"
                )

            # Initialize camera
            self._initialize_camera()

            # Capture and process image
            with PerformanceTimer("image_capture_and_processing",
                                  target_ms=self.config.performance_target_ms):
                capture_result = self._capture_and_process_image()

                if not capture_result["success"]:
                    return AuthenticationResponse(
                        success=False,
                        result=capture_result["result"],
                        error_message=capture_result["error"]
                    )

                features = capture_result["features"]
                quality_score = capture_result["quality_score"]
                liveness_score = capture_result.get("liveness_score")

            # Match against all templates
            best_match = None
            best_similarity = 0.0

            for user_id, template in templates.items():
                similarity = self.feature_extractor.compute_similarity(
                    features, template)
                threshold = self._get_user_threshold(user_id)

                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = user_id

            processing_time = time.time() - start_time

            if best_match:
                response = AuthenticationResponse(
                    success=True,
                    user_id=best_match,
                    confidence=best_similarity,
                    result=AuthenticationResult.SUCCESS,
                    processing_time=processing_time,
                    quality_score=quality_score,
                    liveness_score=liveness_score
                )

                security_logger.log_security_event(
                    SecurityEventType.AUTHENTICATION_SUCCESS,
                    user_id=best_match,
                    additional_data={
                        "confidence": best_similarity,
                        "processing_time": processing_time
                    }
                )

            else:
                response = AuthenticationResponse(
                    success=False,
                    result=AuthenticationResult.REJECTED,
                    confidence=best_similarity,
                    processing_time=processing_time,
                    quality_score=quality_score,
                    liveness_score=liveness_score
                )

                security_logger.log_security_event(
                    SecurityEventType.AUTHENTICATION_FAILURE,
                    additional_data={
                        "best_similarity": best_similarity,
                        "processing_time": processing_time
                    },
                    success=False
                )

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            logger.error(f"1:N authentication failed: {e}")

            return AuthenticationResponse(
                success=False,
                result=AuthenticationResult.ERROR,
                processing_time=processing_time,
                error_message=error_msg
            )

        finally:
            self._cleanup_camera()

    def _perform_authentication(self, user_id: str, template: np.ndarray,
                                start_time: float, timeout: float
                                ) -> AuthenticationResponse:
        """Perform the main authentication logic."""
        while time.time() - start_time < timeout:
            try:
                # Capture and process image with performance target
                with PerformanceTimer("authentication_cycle",
                                      target_ms=self.config
                                      .performance_target_ms):
                    capture_result = self._capture_and_process_image()

                if not capture_result["success"]:
                    if capture_result["result"] in [
                        AuthenticationResult.NO_FACE_DETECTED,
                        AuthenticationResult.QUALITY_INSUFFICIENT
                    ]:
                        # Try again for these recoverable errors
                        time.sleep(0.1)
                        continue
                    else:
                        # Return immediately for non-recoverable errors
                        return AuthenticationResponse(
                            success=False,
                            user_id=user_id,
                            result=capture_result["result"],
                            processing_time=time.time() - start_time,
                            error_message=capture_result["error"]
                        )

                features = capture_result["features"]
                quality_score = capture_result["quality_score"]
                liveness_score = capture_result.get("liveness_score")

                # Compute similarity
                similarity = self.feature_extractor.compute_similarity(
                    features, template)
                threshold = self._get_user_threshold(user_id)

                # Make authentication decision
                if similarity >= threshold:
                    # Successful authentication
                    processing_time = time.time() - start_time

                    return AuthenticationResponse(
                        success=True,
                        user_id=user_id,
                        confidence=similarity,
                        result=AuthenticationResult.SUCCESS,
                        processing_time=processing_time,
                        quality_score=quality_score,
                        liveness_score=liveness_score
                    )

                # Continue trying if similarity is close but not quite enough
                if similarity > threshold * 0.8:
                    time.sleep(0.1)
                    continue

                # If similarity is very low, return rejection immediately
                if similarity < threshold * 0.5:
                    break

            except Exception as e:
                logger.warning(f"Authentication cycle error: {e}")
                time.sleep(0.1)
                continue

        # Authentication failed or timed out
        processing_time = time.time() - start_time
        result = (AuthenticationResult.TIMEOUT if processing_time >= timeout
                  else AuthenticationResult.REJECTED)

        return AuthenticationResponse(
            success=False,
            user_id=user_id,
            result=result,
            processing_time=processing_time
        )

    def _capture_and_process_image(self) -> Dict[str, Any]:
        """Capture and process a single image for authentication."""
        try:
            # Capture frame
            if self.camera is None:
                return {
                    "success": False,
                    "result": AuthenticationResult.ERROR,
                    "error": "Camera not initialized"
                }
            ret, frame = self.camera.read()
            if not ret:
                return {
                    "success": False,
                    "result": AuthenticationResult.ERROR,
                    "error": "Failed to capture frame"
                }

            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            if not faces:
                return {
                    "success": False,
                    "result": AuthenticationResult.NO_FACE_DETECTED,
                    "error": "No face detected"
                }

            # Use largest face
            face_bbox = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face_bbox
            face_image = frame[y:y+h, x:x+w]

            # Check quality
            quality_score = self.quality_assessor.assess_quality(face_image)
            if quality_score < self.config.quality_threshold:
                return {
                    "success": False,
                    "result": AuthenticationResult.QUALITY_INSUFFICIENT,
                    "error": f"Quality too low: {quality_score:.3f}"
                }

            # Liveness detection
            liveness_score = None
            if self.liveness_detector:
                liveness_score = self.liveness_detector.detect_liveness(
                    frame, face_bbox)
                if liveness_score < 0.5:  # Liveness threshold
                    security_logger.log_security_event(
                        SecurityEventType.LIVENESS_DETECTION_FAIL,
                        additional_data={"liveness_score": liveness_score}
                    )
                    return {
                        "success": False,
                        "result": AuthenticationResult.LIVENESS_FAILED,
                        "error": "Liveness detection failed"
                    }

            # Extract features
            features = self.feature_extractor.extract_features(face_image)
            if features is None:
                return {
                    "success": False,
                    "result": AuthenticationResult.ERROR,
                    "error": "Feature extraction failed"
                }

            return {
                "success": True,
                "features": features,
                "quality_score": quality_score,
                "liveness_score": liveness_score,
                "face_bbox": face_bbox
            }

        except Exception as e:
            logger.error(f"Image capture and processing failed: {e}")
            return {
                "success": False,
                "result": AuthenticationResult.ERROR,
                "error": str(e)
            }

    def _load_user_template(self, user_id: str, password: str
                            ) -> Optional[np.ndarray]:
        """Load and decrypt user template."""
        try:
            template_bytes = self.template_storage.load_template(
                user_id, password)
            if template_bytes is None:
                return None

            template = np.frombuffer(template_bytes, dtype=np.float32)
            return template

        except Exception as e:
            logger.error(f"Failed to load template for user {user_id}: {e}")
            return None

    def _get_user_threshold(self, user_id: str) -> float:
        """Get adaptive threshold for user."""
        if (self.config.enable_adaptive_threshold and
                user_id in self.user_thresholds):
            return self.user_thresholds[user_id]
        return self.config.similarity_threshold

    def _update_user_statistics(self, user_id: str, success: bool):
        """Update user-specific statistics and adaptive thresholds."""
        if success:
            # Reset failed attempts on success
            self.failed_attempts[user_id] = 0
            if user_id in self.lockout_until:
                del self.lockout_until[user_id]
        else:
            # Increment failed attempts
            self.failed_attempts[user_id] = self.failed_attempts.get(
                user_id, 0) + 1

            # Apply lockout if too many failures
            if (self.failed_attempts[user_id] >=
                    self.config.max_failed_attempts):
                self.lockout_until[user_id] = time.time(
                ) + self.config.lockout_duration
                logger.warning(
                    f"User {user_id} locked out due to failed attempts")

    def _is_user_locked_out(self, user_id: str) -> bool:
        """Check if user is currently locked out."""
        if user_id not in self.lockout_until:
            return False

        if time.time() < self.lockout_until[user_id]:
            return True
        else:
            # Lockout expired
            del self.lockout_until[user_id]
            self.failed_attempts[user_id] = 0
            return False

    def _initialize_camera(self):
        """Initialize camera for authentication."""
        if self.camera is not None:
            return  # Already initialized

        try:
            self.camera = cv2.VideoCapture(self.config.camera_id)

            if not self.camera.isOpened():
                raise CameraError(
                    f"Cannot open camera {self.config.camera_id}")

            # Set camera properties for optimal performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.target_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT,
                            self.config.target_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

            # Warm up camera
            for _ in range(3):
                ret, frame = self.camera.read()
                if not ret:
                    raise CameraError("Camera warm-up failed")

            logger.debug("Camera initialized for authentication")

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise CameraError(f"Failed to initialize camera: {e}")

    def _cleanup_camera(self):
        """Clean up camera resources."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            logger.debug("Camera resources released")

    def get_authentication_statistics(self) -> Dict[str, Any]:
        """Get authentication statistics and performance metrics."""
        return {
            "failed_attempts": dict(self.failed_attempts),
            "locked_users": list(self.lockout_until.keys()),
            "user_thresholds": dict(self.user_thresholds)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test authentication engine
    config = AuthenticationConfig(
        similarity_threshold=0.6,
        quality_threshold=0.5,
        max_authentication_time=5
    )

    auth_engine = AuthenticationEngine(config)

    print("Testing authentication engine...")
    print("Note: This requires enrolled user templates and a working camera")

    # Mock authentication test
    user_id = "test_user"
    password = "secure_password_123"

    try:
        response = auth_engine.authenticate_user(
            user_id, password, timeout=3.0)

        print(f"Authentication result: {response.result.value}")
        print(f"Success: {response.success}")
        print(f"Confidence: {response.confidence:.3f}")
        print(f"Processing time: {response.processing_time:.2f}s")

        if response.error_message:
            print(f"Error: {response.error_message}")

    except Exception as e:
        print(f"Authentication test failed: {e}")

    print("Authentication test completed!")
