import os
import re
import logging
from dataclasses import dataclass, field
from typing import Iterable, Set

logger = logging.getLogger("interrupt-filter")


def _split_csv(s: str | None):
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]


@dataclass
class InterruptionPolicy:
    ignored_fillers: Set[str] = field(default_factory=lambda: {
        "uh", "um", "umm", "hmm", "mm", "ah", "er", "haan", "huh", "okay", "ok", "yeah", "uh-huh", "right"
    })
    command_keywords: Set[str] = field(default_factory=lambda: {
        "stop", "wait", "pause", "hold on", "one second",
        "no", "not that", "excuse me", "listen"
    })
    min_confidence: float = 0.55

    @classmethod
    def from_env(cls):
        return cls(
            ignored_fillers=set(_split_csv(os.getenv("IGNORED_FILLERS"))) or cls().ignored_fillers,
            command_keywords=set(_split_csv(os.getenv("INTERRUPT_COMMANDS"))) or cls().command_keywords,
            min_confidence=float(os.getenv("ASR_MIN_CONFIDENCE", cls().min_confidence)),
        )


class InterruptionDecision:
    IGNORE = "ignore"
    INTERRUPT = "interrupt"
    REGISTER = "register"


class InterruptionFilter:
    def __init__(self, policy: InterruptionPolicy):
        self.policy = policy

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']", " ", text)
        return re.sub(r"\s+", " ", text)

    def _tokens(self, text: str) -> Set[str]:
        return set(self._normalize(text).split())

    def is_fillers_only(self, text: str) -> bool:
        tokens = self._tokens(text)
        return bool(tokens) and tokens.issubset(self.policy.ignored_fillers)

    def contains_command(self, text: str) -> bool:
        norm = self._normalize(text)
        tokens = self._tokens(text)
        for cmd in self.policy.command_keywords:
            if " " in cmd and cmd in norm:
                return True
            if cmd in tokens:
                return True
        return False

    def decide(self, *, text: str, confidence: float | None, agent_speaking: bool):
        conf = confidence if confidence is not None else 1.0

        if not agent_speaking:
            return InterruptionDecision.REGISTER

        if conf < self.policy.min_confidence:
            return InterruptionDecision.IGNORE

        if self.is_fillers_only(text):
            return InterruptionDecision.IGNORE

        if self.contains_command(text):
            return InterruptionDecision.INTERRUPT

        return InterruptionDecision.INTERRUPT
