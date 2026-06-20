"""Teacher policies for Skill Dojo V1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StandTeacher:
    """Neutral stand target for consolidation-time imitation.

    Zero normalized motor targets are the neutral body-control reference. The
    dojo never applies this as live control; it records the target as an expert
    replay/consolidation hint whose weight is set by adaptive teacher assist.
    """

    name: str = "stand_teacher"

    def motor_target(self, *, n_actuators: int, metrics: dict[str, float]) -> list[float]:
        return [0.0] * max(1, int(n_actuators))


TEACHERS = {"stand_teacher": StandTeacher()}


def get_teacher(name: str):
    return TEACHERS.get(str(name), TEACHERS["stand_teacher"])
