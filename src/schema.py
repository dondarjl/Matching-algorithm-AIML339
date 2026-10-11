"""
Conecta individual profile schema (Track 2: synthetic domain-transfer dataset).

Design constraints (per project proposal):
  - No nationality field, anywhere. This is a deliberate fairness choice so
    the matching engine cannot learn to cluster students by nationality,
    even indirectly.
  - Fields are the minimum needed to compute the four pairwise features
    used by Model B / the fusion model: same_cohort, arrival_date_proximity,
    course_overlap_jaccard, and (from the bio) embedding_cosine_similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class UserProfile:
    """A single Conecta user profile (real or synthetic)."""

    user_id: str
    bio: str  # free-text self-description, embedded with MiniLM for Model A
    cohort: str  # e.g. "2026-T2" -- exchange/enrollment cohort, drives same_cohort
    arrival_date: date  # drives arrival_date_proximity (exponential decay)
    courses: list[str] = field(default_factory=list)  # course codes, drives course_overlap_jaccard

    # Optional context, not used as model features (kept for display / future use)
    display_name: str | None = None
    home_institution: str | None = None

    def __post_init__(self) -> None:
        if not self.bio or not self.bio.strip():
            raise ValueError(f"UserProfile {self.user_id}: bio cannot be empty")
        if not self.cohort or not self.cohort.strip():
            raise ValueError(f"UserProfile {self.user_id}: cohort cannot be empty")
        # Normalize course codes for consistent Jaccard comparison downstream
        self.courses = sorted({c.strip().upper() for c in self.courses if c.strip()})

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "bio": self.bio,
            "cohort": self.cohort,
            "arrival_date": self.arrival_date.isoformat(),
            "courses": self.courses,
            "display_name": self.display_name,
            "home_institution": self.home_institution,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        d = dict(d)
        d["arrival_date"] = date.fromisoformat(d["arrival_date"])
        return cls(**d)


if __name__ == "__main__":
    p = UserProfile(
        user_id="u001",
        bio="Exchange student from a design background, into rock climbing and film photography.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 6),
        courses=["aiml339", "desn301"],
    )
    print(p)
    print(p.to_dict())
