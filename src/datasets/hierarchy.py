"""IDD label hierarchy (``external/public-code/helpers/anue_labels.py``).

The 26 official panoptic classes are IDD level3. Each belongs to one level2
group and one level1 group; every group is purely thing or purely stuff, so
PQ is well defined at the coarse levels too.
"""
from __future__ import annotations

# level3 class name -> (level2Id, level1Id)
LEVELS: dict[str, tuple[int, int]] = {
    "road": (0, 0),
    "parking": (1, 0),
    "drivable fallback": (1, 0),
    "sidewalk": (2, 1),
    "rail track": (3, 1),
    "non-drivable fallback": (3, 1),
    "person": (4, 2),
    "animal": (4, 2),
    "rider": (5, 2),
    "motorcycle": (6, 3),
    "bicycle": (6, 3),
    "autorickshaw": (7, 3),
    "car": (7, 3),
    "truck": (8, 3),
    "bus": (8, 3),
    "caravan": (8, 3),
    "vehicle fallback": (8, 3),
    "curb": (9, 4),
    "wall": (9, 4),
    "fence": (10, 4),
    "guard rail": (10, 4),
    "billboard": (11, 4),
    "traffic sign": (11, 4),
    "traffic light": (11, 4),
    "pole": (12, 4),
    "obs-str-bar-fallback": (12, 4),
    "building": (13, 5),
    "bridge": (13, 5),
    "vegetation": (14, 5),
    "sky": (15, 6),
}

LEVEL2_NAMES = [
    "drivable", "drivable fallback", "sidewalk", "non-drivable fallback", "person",
    "rider", "two-wheeler", "vehicle", "large vehicle", "barrier", "fence/guard rail",
    "sign/light/billboard", "pole/fallback", "building/bridge", "vegetation", "sky",
]
LEVEL1_NAMES = ["drivable", "non-drivable", "living thing", "vehicle",
                "roadside object", "far object", "sky"]


def coarse_ids(class_names: list[str], level: int) -> list[int]:
    """Coarse id of each fine class, in the order of ``class_names``."""
    if level not in (1, 2):
        raise ValueError(f"hierarchy level must be 1 or 2, got {level}")
    return [LEVELS[n][2 - level] for n in class_names]


def coarse_names(level: int) -> list[str]:
    return LEVEL2_NAMES if level == 2 else LEVEL1_NAMES
