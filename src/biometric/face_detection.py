"""
Face detection module using modern computer vision techniques.

This module provides robust face detection capabilities using
optimized CNN models for real-time performance.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Sequence, Any
from pathlib import Path
try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - runtime dependency guard
    ort = None  # type: ignore[assignment]

try:
    from core.logging import get_logger, PerformanceTimer
    from core.exceptions import FaceDetectionError, ModelLoadError
except ModuleNotFoundError:
    try:
        from src.core.logging import get_logger, PerformanceTimer
        from src.core.exceptions import FaceDetectionError, ModelLoadError
    except ModuleNotFoundError:
        from LockLess.src.core.logging import get_logger, PerformanceTimer
        from LockLess.src.core.exceptions import FaceDetectionError, ModelLoadError

logger = get_logger(__name__)


class FaceDetector:
    """
    High-performance face detector using ONNX-optimized models.

    Supports multiple detection backends:
    - RetinaFace (high accuracy)
    - MobileNet-SSD (fast inference)
    - OpenCV DNN (fallback)
    """

    def __init__(self, model_path: Optional[str] = None,
                 backend: str = "retinaface",
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize face detector.

        Args:
            model_path: Path to ONNX model file
            backend: Detection backend ("retinaface", "mobilenet", "opencv")
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
        """
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.model_path = model_path or self._get_default_model_path(backend)
        self.session = None
        self.input_name: Optional[str] = None
        self.opencv_detector: Optional[cv2.CascadeClassifier] = None
        self.input_size = (640, 640)  # Default input size

        self._initialize_detector()

        logger.info(f"FaceDetector initialized with backend: {backend}")

    def _get_default_model_path(self, backend: str) -> Optional[str]:
        """Get default model path based on backend."""
        model_paths = {
            "retinaface": "models/face_detection/retinaface.onnx",
            "mobilenet": "models/face_detection/mobilenet_v2.onnx",
            "opencv": None  # Uses built-in OpenCV models
        }
        return model_paths.get(backend)

    def _initialize_detector(self):
        """Initialize the face detection model."""
        try:
            if self.backend == "opencv":
                self._initialize_opencv_detector()
            else:
                self._initialize_onnx_detector()

        except Exception as e:
            logger.error(f"Face detector initialization failed: {e}")
            # Fallback to OpenCV if ONNX fails
            if self.backend != "opencv":
                logger.warning("Falling back to OpenCV face detector")
                self.backend = "opencv"
                self._initialize_opencv_detector()
            else:
                raise FaceDetectionError(
                    f"All face detection backends failed: {e}")

    def _initialize_onnx_detector(self):
        """Initialize ONNX-based face detector."""
        try:
            if ort is None:
                raise ModelLoadError("ONNX Runtime not available")

            # Check if model file exists
            import os
            if self.model_path is None:
                raise ModelLoadError("Model path is None")
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                raise ModelLoadError(
                    f"Model file not found: {self.model_path}")

            # Initialize ONNX Runtime session
            providers = ['CPUExecutionProvider']

            # Try to use GPU if available
            try:
                if ort.get_device() == 'GPU':
                    providers = ['CUDAExecutionProvider',
                                 'CPUExecutionProvider']
            except (AttributeError, RuntimeError):
                pass

            self.session = ort.InferenceSession(
                self.model_path, providers=providers)

            # Get input details
            input_details = self.session.get_inputs()[0]
            self.input_name = input_details.name
            input_shape = input_details.shape

            if len(input_shape) == 4:  # NCHW or NHWC
                if input_shape[1] == 3:  # NCHW
                    self.input_size = (input_shape[2], input_shape[3])
                else:  # NHWC
                    self.input_size = (input_shape[1], input_shape[2])

            logger.debug(f"ONNX model loaded: {self.model_path}")
            logger.debug(f"Input size: {self.input_size}")

        except Exception as e:
            logger.error(f"ONNX detector initialization failed: {e}")
            raise ModelLoadError(f"Failed to load ONNX model: {e}")

    def _initialize_opencv_detector(self):
        """Initialize OpenCV-based face detector."""
        try:
            # Use OpenCV's built-in Haar cascade or DNN
            haarcascade_path = self._resolve_haarcascade_path()

            self.opencv_detector = cv2.CascadeClassifier(haarcascade_path)

            if self.opencv_detector.empty():
                raise ModelLoadError("Failed to load OpenCV face cascade")

            logger.debug("OpenCV face detector initialized")

        except Exception as e:
            logger.error(f"OpenCV detector initialization failed: {e}")
            raise ModelLoadError(f"Failed to initialize OpenCV detector: {e}")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of face bounding boxes as (x, y, width, height) tuples
        """
        try:
            with PerformanceTimer("face_detection", target_ms=50):
                if self.backend == "opencv":
                    return self._detect_opencv(image)
                else:
                    return self._detect_onnx(image)

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []

    def _detect_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV."""
        try:
            if self.opencv_detector is None:
                raise ModelLoadError("OpenCV detector not initialized")

            bgr_image = self._ensure_bgr(image)
            # Convert to grayscale
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.opencv_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Filter by confidence (OpenCV doesn't provide confidence scores)
            # Apply basic filtering based on face size and aspect ratio
            filtered_faces = []
            for (x, y, w, h) in faces:
                # Check aspect ratio (faces should be roughly square)
                aspect_ratio = w / h
                if 0.7 <= aspect_ratio <= 1.3:
                    filtered_faces.append((x, y, w, h))

            return filtered_faces

        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return []

    def _detect_onnx(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using ONNX model."""
        try:
            if self.session is None:
                raise ModelLoadError("ONNX session not initialized")

            # Preprocess image
            input_image = self._preprocess_image(image)

            # Run inference
            outputs = self.session.run(None, {self.input_name: input_image})

            # Post-process results
            faces = self._postprocess_onnx_output(outputs, image.shape)

            return faces

        except Exception as e:
            logger.error(f"ONNX face detection failed: {e}")
            return []

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model."""
        # Resize image
        bgr_image = self._ensure_bgr(image)
        resized = cv2.resize(bgr_image, self.input_size)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0

        # Convert to NCHW format
        input_image = np.transpose(normalized, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)

        return input_image

    def _postprocess_onnx_output(self, outputs: Sequence[Any],
                                 original_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Post-process ONNX model output."""
        # This is a simplified post-processing function
        # Real implementation would depend on the specific model format

        try:
            # Assuming outputs contain [boxes, scores, classes]
            boxes = np.asarray(outputs[0])

            if len(outputs) >= 2:
                scores = np.asarray(outputs[1])
            else:
                # Handle single output models
                scores = np.ones((boxes.shape[0],), dtype=np.float32)

            boxes = boxes.reshape(-1, boxes.shape[-1])
            scores = scores.reshape(-1)

            # Filter by confidence
            valid_indices = scores > self.confidence_threshold
            filtered_boxes = boxes[valid_indices]
            filtered_scores = scores[valid_indices]

            # Convert normalized coordinates to pixel coordinates
            h, w = original_shape[:2]
            faces = []

            for box, score in zip(filtered_boxes, filtered_scores):
                # Assuming box format: [x1, y1, x2, y2] normalized
                x1, y1, x2, y2 = box[:4]

                # Convert to pixel coordinates
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)

                # Convert to (x, y, width, height) format
                width = x2 - x1
                height = y2 - y1

                if width > 0 and height > 0:
                    faces.append((x1, y1, width, height))

            # Apply Non-Maximum Suppression
            faces = self._apply_nms(faces, filtered_scores)

            return faces

        except Exception as e:
            logger.error(f"ONNX output post-processing failed: {e}")
            return []

    def _apply_nms(self, boxes: List[Tuple[int, int, int, int]],
                   scores: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if not boxes:
            return []

        try:
            # Convert to OpenCV NMS format
            boxes_array = np.array(boxes, dtype=np.float32)
            scores_array = np.array(scores, dtype=np.float32)

            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes_array.tolist(),
                scores_array.tolist(),
                self.confidence_threshold,
                self.nms_threshold
            )

            if len(indices) > 0:
                # Convert indices to list and flatten if needed
                if hasattr(indices, 'flatten'):
                    indices_list = indices.flatten().tolist()  # type: ignore
                else:
                    indices_list = list(indices)
                return [boxes[i] for i in indices_list]
            else:
                return []

        except Exception as e:
            logger.error(f"NMS application failed: {e}")
            return boxes  # Return original boxes if NMS fails

    def detect_largest_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face in an image.

        Args:
            image: Input image

        Returns:
            Largest face bounding box or None if no face detected
        """
        faces = self.detect_faces(image)
        if not faces:
            return None

        # Return face with largest area
        return max(faces, key=lambda f: f[2] * f[3])

    def get_face_landmarks(self, image: np.ndarray,
                           face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Get facial landmarks for a detected face.

        Args:
            image: Input image
            face_bbox: Face bounding box

        Returns:
            Facial landmarks array or None if detection fails
        """
        # Placeholder for landmark detection
        # In a full implementation, this would use a separate landmark model
        logger.debug("Facial landmark detection not implemented")
        return None

    def visualize_detections(self, image: np.ndarray,
                             faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Visualize face detections on image.

        Args:
            image: Input image
            faces: List of face bounding boxes

        Returns:
            Image with drawn bounding boxes
        """
        result_image = image.copy()

        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add confidence text (if available)
            cv2.putText(result_image, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return result_image

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        """Ensure input image is in BGR format."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 3:
                return image
            if channels == 1:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        raise ValueError("Unsupported image format for face detection")

    def _resolve_haarcascade_path(self) -> str:
        """Determine the appropriate Haar cascade file path."""
        data_module = getattr(cv2, "data", None)
        haar_dir = getattr(data_module, "haarcascades",
                           None) if data_module else None

        if isinstance(haar_dir, str):
            candidate = Path(haar_dir) / "haarcascade_frontalface_default.xml"
            if candidate.exists():
                return str(candidate)

        return "haarcascade_frontalface_default.xml"


if __name__ == "__main__":
    detector = FaceDetector(backend="opencv")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        exit()

    print("Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect_faces(frame)

        print("Faces detected:", len(faces))

        # Draw boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # SHOW WINDOW 🔥
        cv2.imshow("Face Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
