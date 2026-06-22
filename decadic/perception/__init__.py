"""Perceptual package: integrator, object files, and health gates."""

from decadic.perception.integration import PerceptualIntegrator
from decadic.perception.organ import PerceptionOrgan, PerceptionOrganDiagnostics, RetinotopicMap
from decadic.perception.object_files import (
    DiscoveryHealth,
    HEALTH_STATES,
    ObjectFile,
    evaluate_discovery_health,
    object_files_from_proposals,
)

__all__ = [
    "DiscoveryHealth",
    "HEALTH_STATES",
    "ObjectFile",
    "PerceptionOrgan",
    "PerceptionOrganDiagnostics",
    "PerceptualIntegrator",
    "RetinotopicMap",
    "evaluate_discovery_health",
    "object_files_from_proposals",
]
