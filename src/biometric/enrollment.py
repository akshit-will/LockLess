"""
Biometric enrollment module for the Lockless authentication system.

This module handles the complete enrollment pipeline from camera capture
to biometric template generation and storage.
"""

from pyexpat import features
from tempfile import template

import cv2
import numpy as np
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from core.logging import (get_logger, get_security_logger,
                            SecurityEventType, PerformanceTimer)
from core.exceptions import (
    EnrollmentError, CameraError,
    FeatureExtractionError
)
from security.encryption import SecureTemplateStorage
from biometric.face_detection import FaceDetector
from biometric.feature_extraction import FeatureExtractor
from biometric.quality_assessment import QualityAssessment

logger = get_logger(__name__)
security_logger = get_security_logger()


class EnrollmentStep(Enum):
    """Enumeration of enrollment steps."""
    CAMERA_INIT = "camera_initialization"
    FACE_DETECTION = "face_detection"
    QUALITY_CHECK = "quality_assessment"
    FEATURE_EXTRACTION = "feature_extraction"
    TEMPLATE_GENERATION = "template_generation"
    TEMPLATE_STORAGE = "template_storage"
    VERIFICATION = "verification"


@dataclass
class EnrollmentConfig:
    """Configuration for biometric enrollment."""
    camera_id: int = 0
    target_width: int = 640
    target_height: int = 480
    fps: int = 30
    min_face_size: int = 100
    max_enrollment_time: int = 30  # seconds
    required_samples: int = 5
    quality_threshold: float = 0.7
    feature_dimension: int = 512
    template_storage_path: str = "templates"


@dataclass
class EnrollmentSample:
    """Single enrollment sample data."""
    image: np.ndarray
    face_bbox: Tuple[int, int, int, int]  # x, y, width, height
    quality_score: float
    features: np.ndarray
    timestamp: float


@dataclass
class EnrollmentResult:
    """Result of enrollment process."""
    success: bool
    user_id: str
    template_id: Optional[str] = None
    samples_collected: int = 0
    average_quality: float = 0.0
    error_message: Optional[str] = None
    processing_time: float = 0.0


class BiometricEnrollment:
    """
    Handles biometric enrollment with face detection and feature extraction.

    The enrollment process includes:
    1. Camera initialization and capture
    2. Face detection and tracking
    3. Image quality assessment
    4. Feature extraction and template generation
    5. Secure template storage
    """

    def __init__(self, config: Optional[EnrollmentConfig] = None):
        """
        Initialize enrollment system.

        Args:
            config: Enrollment configuration
        """
        self.config = config or EnrollmentConfig()
        self.face_detector = FaceDetector()
        self.feature_extractor = FeatureExtractor()
        self.quality_assessor = QualityAssessment()
        self.template_storage = SecureTemplateStorage(
            self.config.template_storage_path)

        self.camera = None
        self.is_enrolling = False

        logger.info("BiometricEnrollment initialized")

    def enroll_user(self, user_id: str, password: str) -> EnrollmentResult:
        """
        Enroll a new user with biometric data.

        Args:
            user_id: Unique user identifier
            password: Master password for template encryption

        Returns:
            EnrollmentResult with success status and details
        """
        start_time = time.time()
        samples = []

        try:
            logger.info(f"Starting enrollment for user: {user_id}")
            security_logger.log_security_event(
                SecurityEventType.ENROLLMENT_SUCCESS,
                user_id=user_id,
                additional_data={"stage": "start"}
            )

            # Step 1: Initialize camera
            with PerformanceTimer("camera_initialization"):
                self._initialize_camera()

            # Step 2: Collect enrollment samples
            samples = self._collect_enrollment_samples(user_id)

            if len(samples) < self.config.required_samples:
                raise EnrollmentError(
                    f"Insufficient samples collected: "
                    f"{len(samples)}/{self.config.required_samples}"
                )

            # Step 3: Generate biometric template
            with PerformanceTimer("template_generation"):
                template = self._generate_template(samples)

            # Step 4: Store encrypted template
            with PerformanceTimer("template_storage"):
                template_id = self._store_template(user_id, template, password)

            # Step 5: Verify stored template
            with PerformanceTimer("template_verification"):
                verification_success = self._verify_stored_template(
                    user_id, template, password
                )

            if not verification_success:
                raise EnrollmentError("Template verification failed")

            processing_time = time.time() - start_time
            average_quality = float(
                np.mean([s.quality_score for s in samples]))

            result = EnrollmentResult(
                success=True,
                user_id=user_id,
                template_id=template_id,
                samples_collected=len(samples),
                average_quality=average_quality,
                processing_time=processing_time
            )

            security_logger.log_security_event(
                SecurityEventType.ENROLLMENT_SUCCESS,
                user_id=user_id,
                additional_data={
                    "samples_collected": len(samples),
                    "average_quality": average_quality,
                    "processing_time": processing_time
                }
            )

            logger.info(f"Enrollment successful for user: {user_id}")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            result = EnrollmentResult(
                success=False,
                user_id=user_id,
                samples_collected=len(samples),
                error_message=error_msg,
                processing_time=processing_time
            )

            security_logger.log_security_event(
                SecurityEventType.ENROLLMENT_FAILURE,
                user_id=user_id,
                additional_data={"error": error_msg},
                success=False
            )

            logger.error(f"Enrollment failed for user {user_id}: {e}")
            return result

        finally:
            self._cleanup_camera()

    def _initialize_camera(self):
        """Initialize camera for enrollment."""
        try:
            self.camera = cv2.VideoCapture(self.config.camera_id)

            if not self.camera.isOpened():
                raise CameraError(
                    f"Cannot open camera {self.config.camera_id}")

            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.target_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT,
                            self.config.target_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)

            # Warm up camera
            for _ in range(5):
                ret, frame = self.camera.read()
                if not ret:
                    raise CameraError("Camera warm-up failed")

            logger.debug("Camera initialized successfully")

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise CameraError(f"Failed to initialize camera: {e}")
        
    def enroll_from_frames(self, user_id, password, frames):
        print("📸 Using frames from GUI")

        samples = []

        for frame in frames:
            faces = self.face_detector.detect_faces(frame)

            if not faces:
                continue

            # ✅ Get bounding box
            x, y, w, h = faces[0]

            # ✅ Crop face from frame
            face_img = frame[y:y+h, x:x+w]

            # Safety check
            if face_img is None or face_img.size == 0:
                continue

            # ✅ Quality check (PASS IMAGE)
            quality_score = self.quality_assessor.assess_quality(face_img)

            if quality_score < self.config.quality_threshold:
                continue

            # ✅ Feature extraction (PASS IMAGE)
            from time import time

            features = self.feature_extractor.extract_features(face_img)

            if features is None:
                continue

            sample = EnrollmentSample(
                image=face_img,
                face_bbox=(x, y, w, h),
                quality_score=quality_score,
                features=features,
                timestamp=time()
            )

            samples.append(sample)

        if len(samples) == 0:
            return EnrollmentResult(
                success=False,
                user_id=user_id,
                samples_collected=0,
                error_message="No valid samples collected",
                processing_time=0.0
            )

        template = self._generate_template(samples)

        # ✅ correct order
        self._store_template(user_id, template, password)

        print("✅ Enrollment successful (GUI mode)")

        return EnrollmentResult(
            success=True,
            user_id=user_id,
            template_id=None,
            samples_collected=len(samples),
            average_quality=float(
                np.mean([s.quality_score for s in samples])
            ),
            error_message=None,
            processing_time=0.0
        )

    def _collect_enrollment_samples(self, user_id: str
                                    ) -> List[EnrollmentSample]:
        """
        Collect enrollment samples from camera feed.

        Args:
            user_id: User identifier for logging

        Returns:
            List of valid enrollment samples
        """
        samples = []
        start_time = time.time()

        logger.info(f"Collecting enrollment samples for user: {user_id}")

        while (len(samples) < self.config.required_samples and
               time.time() - start_time < self.config.max_enrollment_time):

            try:
                # Capture frame
                if self.camera is None:
                    raise CameraError("Camera not initialized")
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue

                # Detect faces
                with PerformanceTimer("face_detection"):
                    faces = self.face_detector.detect_faces(frame)

                if not faces:
                    continue

                # Use the largest face
                face_bbox = max(faces, key=lambda f: f[2] * f[3])

                # Check face size
                if (face_bbox[2] < self.config.min_face_size or
                        face_bbox[3] < self.config.min_face_size):
                    continue

                # Extract face region
                x, y, w, h = face_bbox
                face_image = frame[y:y+h, x:x+w]

                # Assess quality
                with PerformanceTimer("quality_assessment"):
                    quality_score = self.quality_assessor.assess_quality(
                        face_image)

                if quality_score < self.config.quality_threshold:
                    logger.debug(f"Low quality sample: {quality_score:.3f}")
                    continue

                # Extract features
                with PerformanceTimer("feature_extraction"):
                    features = self.feature_extractor.extract_features(
                        face_image)

                if features is None:
                    logger.debug("Feature extraction failed")
                    continue

                # Create enrollment sample
                sample = EnrollmentSample(
                    image=face_image.copy(),
                    face_bbox=face_bbox,
                    quality_score=quality_score,
                    features=features,
                    timestamp=time.time()
                )

                samples.append(sample)
                logger.debug(
                    f"Collected sample {len(samples)}/"
                    f"{self.config.required_samples}")

                # Brief pause to avoid duplicate samples
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error collecting sample: {e}")
                continue

        logger.info(f"Collected {len(samples)} enrollment samples")
        return samples

    def _generate_template(self, samples: List[EnrollmentSample]
                           ) -> np.ndarray:
        """
        Generate biometric template from enrollment samples.

        Args:
            samples: List of enrollment samples

        Returns:
            Biometric template as numpy array
        """
        try:
            if not samples:
                raise FeatureExtractionError(
                    "No samples provided for template generation")

            # Extract features from all samples
            feature_vectors = [sample.features for sample in samples]
            quality_scores = [sample.quality_score for sample in samples]

            # Weighted average based on quality scores
            weights = np.array(quality_scores) / np.sum(quality_scores)
            template = np.average(feature_vectors, axis=0, weights=weights)

            # Normalize template
            template = template / np.linalg.norm(template)

            logger.debug(f"Generated template from {len(samples)} samples")
            return template.astype(np.float32)

        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            raise FeatureExtractionError(f"Failed to generate template: {e}")

    def _store_template(self, user_id: str, template: np.ndarray,
                        password: str) -> str:
        """
        Store encrypted biometric template.

        Args:
            user_id: User identifier
            template: Biometric template
            password: Encryption password

        Returns:
            Template identifier
        """
        try:
            # Convert template to bytes
            template_bytes = template.tobytes()

            # Store with metadata
            metadata = {
                "created_at": time.time(),
                "feature_dimension": len(template),
                "version": "1.0",
                "algorithm": "face_recognition"
            }

            success = self.template_storage.store_template(
                user_id, template_bytes, password, metadata
            )

            if not success:
                raise EnrollmentError("Failed to store template")

            template_id = f"template_{user_id}_{int(time.time())}"
            logger.debug(f"Template stored with ID: {template_id}")

            return template_id

        except Exception as e:
            logger.error(f"Template storage failed: {e}")
            raise EnrollmentError(f"Failed to store template: {e}")

    def _verify_stored_template(self, user_id: str,
                                original_template: np.ndarray,
                                password: str) -> bool:
        """
        Verify that stored template can be retrieved and matches original.

        Args:
            user_id: User identifier
            original_template: Original template for comparison
            password: Decryption password

        Returns:
            True if verification successful
        """
        try:
            # Load stored template
            stored_bytes = self.template_storage.load_template(
                user_id, password)
            if stored_bytes is None:
                return False

            # Convert back to array
            stored_template = np.frombuffer(stored_bytes, dtype=np.float32)

            # Compare templates
            similarity = np.dot(original_template, stored_template)

            # Should be very close to 1.0 for identical templates
            verification_threshold = 0.99
            success = similarity >= verification_threshold

            logger.debug(f"Template verification similarity: {similarity:.4f}")
            return success

        except Exception as e:
            logger.error(f"Template verification failed: {e}")
            return False

    def _cleanup_camera(self):
        """Clean up camera resources."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            logger.debug("Camera resources released")

    def get_enrollment_progress(self) -> Dict[str, Any]:
        """
        Get current enrollment progress.

        Returns:
            Dictionary with progress information
        """
        # This would be implemented with shared state in a real application
        return {
            "is_enrolling": self.is_enrolling,
            "samples_collected": 0,  # Would track actual progress
            "required_samples": self.config.required_samples,
            "quality_threshold": self.config.quality_threshold
        }


# Example usage and testing
if __name__ == "__main__":
    # Test enrollment with mock camera (for development)
    config = EnrollmentConfig(
        camera_id=0,
        required_samples=3,
        quality_threshold=0.5
    )

    enrollment = BiometricEnrollment(config)

    print("Testing biometric enrollment...")
    print("Note: This requires a working camera and face detection models")

    # Mock enrollment test
    user_id = "test_user"
    password = "secure_password_123"

    try:
        result = enrollment.enroll_user(user_id, password)

        if result.success:
            print("Enrollment successful!")
            print(f"  Template ID: {result.template_id}")
            print(f"  Samples collected: {result.samples_collected}")
            print(f"  Average quality: {result.average_quality:.3f}")
            print(f"  Processing time: {result.processing_time:.2f}s")
        else:
            print(f"Enrollment failed: {result.error_message}")

    except Exception as e:
        print(f"Enrollment test failed: {e}")

    print("Enrollment test completed!")
