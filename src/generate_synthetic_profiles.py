"""
Generates the 15-25 synthetic Conecta profiles for Track 2 (domain-transfer
demonstration). Profiles are hand-authored (not randomly generated text) so
the bios read like real self-descriptions and give the embedding model
genuine semantic content to work with -- randomly assembled sentences would
undermine the point of testing embedding-based matching.

Deliberately varied across:
  - academic field / interests (so embedding similarity has real signal)
  - cohort (some share, some don't -- for same_cohort variation)
  - arrival_date (spread across a ~10 week window -- for arrival_date_proximity variation)
  - courses (overlapping sets across some pairs, disjoint for others -- for Jaccard variation)

No nationality field anywhere, per the project's fairness constraint.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from schema import UserProfile

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "profiles.json"

PROFILES: list[UserProfile] = [
    UserProfile(
        user_id="u001",
        display_name="Mia",
        bio="Design student who spends most weekends rock climbing or shooting film photography around the city.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 6),
        courses=["DESN301", "AIML339"],
    ),
    UserProfile(
        user_id="u002",
        display_name="Theo",
        bio="Into outdoor sports and visual storytelling -- hiking most weekends, always carrying a camera.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 10),
        courses=["AIML339", "COMP307"],
    ),
    UserProfile(
        user_id="u003",
        display_name="Priya",
        bio="Data science student obsessed with machine learning research, plays chess competitively in her free time.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 8),
        courses=["AIML339", "COMP307", "MATH301"],
    ),
    UserProfile(
        user_id="u004",
        display_name="Noah",
        bio="Studying economics, spends free time reading about markets and playing five-a-side football on weekends.",
        cohort="2026-T2",
        arrival_date=date(2026, 8, 20),
        courses=["ECON301", "FCOM201"],
    ),
    UserProfile(
        user_id="u005",
        display_name="Sofia",
        bio="Marine biology student who volunteers at a local aquarium and loves scuba diving whenever possible.",
        cohort="2026-T3",
        arrival_date=date(2026, 9, 15),
        courses=["BIOL305", "ENVX201"],
    ),
    UserProfile(
        user_id="u006",
        display_name="Kai",
        bio="Environmental science major passionate about ocean conservation, does beach cleanups most weekends.",
        cohort="2026-T3",
        arrival_date=date(2026, 9, 18),
        courses=["ENVX201", "BIOL305"],
    ),
    UserProfile(
        user_id="u007",
        display_name="Aroha",
        bio="Music composition student, plays in a jazz trio and is always looking for people to jam with.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 12),
        courses=["MUSC220"],
    ),
    UserProfile(
        user_id="u008",
        display_name="Liam",
        bio="Computer science student who builds small indie games in his spare time and loves board game nights.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 5),
        courses=["COMP307", "AIML339"],
    ),
    UserProfile(
        user_id="u009",
        display_name="Elena",
        bio="Law student focused on international human rights, debates competitively and volunteers with refugee support groups.",
        cohort="2026-T3",
        arrival_date=date(2026, 9, 20),
        courses=["LAWS301", "POLS210"],
    ),
    UserProfile(
        user_id="u010",
        display_name="Marco",
        bio="Political science student interested in international relations, active in Model UN and public debate.",
        cohort="2026-T3",
        arrival_date=date(2026, 9, 22),
        courses=["POLS210", "LAWS301"],
    ),
    UserProfile(
        user_id="u011",
        display_name="Yuki",
        bio="Architecture student who sketches buildings on every walk and is slowly filling a notebook with city studies.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 14),
        courses=["ARCH250", "DESN301"],
    ),
    UserProfile(
        user_id="u012",
        display_name="Ben",
        bio="Mechanical engineering student who spends weekends fixing up an old motorbike and going on long rides.",
        cohort="2026-T2",
        arrival_date=date(2026, 8, 25),
        courses=["MECH240"],
    ),
    UserProfile(
        user_id="u013",
        display_name="Isabella",
        bio="Psychology student fascinated by cognitive science, runs a small book club focused on nonfiction.",
        cohort="2026-T3",
        arrival_date=date(2026, 9, 16),
        courses=["PSYC220", "AIML339"],
    ),
    UserProfile(
        user_id="u014",
        display_name="Daniel",
        bio="Studying cognitive science and AI, curious about how the brain and machine learning models compare.",
        cohort="2026-T3",
        arrival_date=date(2026, 9, 14),
        courses=["AIML339", "PSYC220"],
    ),
    UserProfile(
        user_id="u015",
        display_name="Grace",
        bio="Nursing student who volunteers at a community health clinic and trains for half marathons on the side.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 9),
        courses=["NURS210"],
    ),
    UserProfile(
        user_id="u016",
        display_name="Oscar",
        bio="Finance student who's into long-distance running and spreadsheets in roughly equal measure.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 7),
        courses=["FCOM201", "ECON301"],
    ),
    UserProfile(
        user_id="u017",
        display_name="Hana",
        bio="Film studies student who programs a small campus screening series and writes reviews in her spare time.",
        cohort="2026-T3",
        arrival_date=date(2026, 9, 19),
        courses=["FILM215"],
    ),
    UserProfile(
        user_id="u018",
        display_name="Lucas",
        bio="Studies statistics and loves board games, especially anything with a heavy strategic element.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 11),
        courses=["MATH301", "AIML339"],
    ),
    UserProfile(
        user_id="u019",
        display_name="Amara",
        bio="Studying public health with a focus on epidemiology, volunteers weekly at a food bank.",
        cohort="2026-T3",
        arrival_date=date(2026, 9, 17),
        courses=["ENVX201", "NURS210"],
    ),
    UserProfile(
        user_id="u020",
        display_name="Felix",
        bio="Philosophy student who runs a small podcast about ethics in technology, into rock climbing on weekends.",
        cohort="2026-T2",
        arrival_date=date(2026, 8, 22),
        courses=["PHIL200", "AIML339"],
    ),
]


def generate_and_save() -> list[UserProfile]:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = [p.to_dict() for p in PROFILES]
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[generate_synthetic_profiles] wrote {len(PROFILES)} profiles to {OUTPUT_PATH}")
    return PROFILES


def load_profiles() -> list[UserProfile]:
    with open(OUTPUT_PATH) as f:
        data = json.load(f)
    return [UserProfile.from_dict(d) for d in data]


if __name__ == "__main__":
    profiles = generate_and_save()
    cohorts = {p.cohort for p in profiles}
    print(f"cohorts represented: {sorted(cohorts)}")
    all_courses = sorted({c for p in profiles for c in p.courses})
    print(f"courses represented ({len(all_courses)}): {all_courses}")
