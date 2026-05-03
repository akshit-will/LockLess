"""Utility services for the Lockless desktop UI.

These helpers wrap the core biometric workflows so they can be executed
safely from the graphical interface without duplicating business logic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from biometric.authentication import AuthenticationConfig, AuthenticationEngine
from biometric.enrollment import BiometricEnrollment, EnrollmentConfig
from core.config import ConfigManager
from core.logging import get_logger
from security.encryption import SecureTemplateStorage

logger = get_logger(__name__)


@dataclass
class ServiceResponse:
    """Generic response structure returned to the UI layer."""

    success: bool
    message: str
    payload: Optional[Dict[str, Any]] = None


def _load_config(config_path: Optional[str]) -> ConfigManager:
    """Create a ConfigManager instance with optional override path."""

    try:
        return ConfigManager(config_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to load configuration: %s", exc)
        raise


def enroll_user_service(
    user_id: str, password: str, config_path: Optional[str] = None, frames=None
) -> ServiceResponse:
    """Run the biometric enrollment workflow."""

    try:
        config_manager = _load_config(config_path)
        enrollment_config = EnrollmentConfig(
            camera_id=config_manager.get("camera.device_id", 0),
            required_samples=config_manager.get(
                "enrollment.required_samples", 5),
            quality_threshold=config_manager.get(
                "enrollment.quality_threshold", 0.7),
            max_enrollment_time=config_manager.get(
                "enrollment.max_enrollment_time", 30),
        )

        enrollment = BiometricEnrollment(enrollment_config)

        # ✅ NEW: If GUI is sending frames
        if frames is not None:
            print("✅ Using GUI frames, not opening camera again")
            result = enrollment.enroll_from_frames(user_id, password, frames)
        else:
            # old behavior (CLI mode)
            result = enrollment.enroll_user(user_id, password)

        payload: Dict[str, Any] = {"result": asdict(result)}
        if result.success:
            message = (
                "Enrollment successful for user "
                f"'{result.user_id}' (samples: {result.samples_collected})"
            )
            return ServiceResponse(True, message, payload)

        message = result.error_message or "Enrollment failed"
        return ServiceResponse(False, message, payload)

    except Exception as exc:
        logger.error("Enrollment service error: %s", exc)
        return ServiceResponse(False, f"Enrollment error: {exc}")


def authenticate_user_service(
    user_id: str,
    password: str,
    config_path: Optional[str] = None, frames=None
) -> ServiceResponse:
    """Execute authentication for a specific user."""

    try:
        config_manager = _load_config(config_path)
        auth_config = AuthenticationConfig(
            camera_id=config_manager.get("camera.device_id", 0),
            similarity_threshold=config_manager.get(
                "authentication.similarity_threshold", 0.7
            ),
            quality_threshold=config_manager.get(
                "authentication.quality_threshold", 0.6),
            max_authentication_time=config_manager.get(
                "authentication.max_time", 10),
            enable_liveness_detection=config_manager.get(
                "liveness.enable_texture_analysis", True
            ),
        )

        engine = AuthenticationEngine(auth_config)
        if frames is not None:
            response = engine.authenticate_user_from_frames(
                user_id, password, frames
            )
        else:
            response = engine.authenticate_user(user_id, password)

        payload: Dict[str, Any] = {"response": asdict(response)}
        if response.success:
            msg = (
                "Authentication successful for user "
                f"'{response.user_id}' with confidence {response.confidence:.3f}"
            )
            return ServiceResponse(True, msg, payload)

        error_text = response.error_message or response.result.value
        return ServiceResponse(False, f"Authentication failed: {error_text}", payload)

    except Exception as exc:
        logger.error("Authentication service error: %s", exc)
        return ServiceResponse(False, f"Authentication error: {exc}")


def list_users_service(storage_path: Optional[str] = None) -> ServiceResponse:
    """Return the list of enrolled users."""

    try:
        storage = SecureTemplateStorage(storage_path or "templates")
        users = storage.list_users()
        message = f"Found {len(users)} enrolled user(s)."
        return ServiceResponse(True, message, {"users": users})
    except Exception as exc:
        logger.error("List users service error: %s", exc)
        return ServiceResponse(False, f"Failed to list users: {exc}")


def delete_user_service(user_id: str, storage_path: Optional[str] = None) -> ServiceResponse:
    """Delete a stored biometric template."""

    try:
        storage = SecureTemplateStorage(storage_path or "templates")
        success = storage.delete_template(user_id)
        if success:
            return ServiceResponse(True, f"User '{user_id}' deleted successfully.")
        return ServiceResponse(False, f"User '{user_id}' not found.")
    except Exception as exc:
        logger.error("Delete user service error: %s", exc)
        return ServiceResponse(False, f"Failed to delete user: {exc}")


def test_camera_service(camera_id: int = 0) -> ServiceResponse:
    """Perform a lightweight camera health check."""

    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return ServiceResponse(False, f"Cannot open camera {camera_id}")

        ret, frame = cap.read()
        cap.release()
        if not ret:
            return ServiceResponse(False, f"Cannot read frames from camera {camera_id}")

        height, width = frame.shape[:2]
        message = f"Camera {camera_id} OK ({width}x{height})"
        return ServiceResponse(True, message, {"resolution": (width, height)})

    except Exception as exc:
        logger.error("Camera test service error: %s", exc)
        return ServiceResponse(False, f"Camera test error: {exc}")


def get_logo_path() -> Path:
    """Resolve the absolute path to the application logo."""

    root = Path(__file__).resolve().parents[2]
    logo_path = root / "assets" / "lockless-logo.png"
    return logo_path


def get_system_summary(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Gather system information to display on the dashboard."""

    import platform
    import sys

    config = _load_config(config_path)
    summary = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "camera_id": config.get("camera.device_id", 0),
        "data_dir": config.get("system.data_directory"),
        "log_level": config.get("system.log_level", "INFO"),
    }

    return summary
