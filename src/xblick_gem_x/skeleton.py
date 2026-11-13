"""The skeletons: GEM-X's 77 SOMA joints (``SOMA77``: the SOMA 0.3.3 rig's joints without its root and its twist
joints, in the rig's order; ``check_soma77`` compares with the rig) and their mapping to xblick-azure-kinect's body
tracking (pose model K4ABT_BODY_32), so that this node's bodies go wherever the Azure Kinect's go: the
same JSON on ``<camera>/<node>/bodies``.

The mapping is by anatomy. SOMA has no nose and no ears: the nose is placed between the eyes and 4 cm ahead of them
(away from the head joint), the ears 7.5 cm out from the head joint along the eyes' axis, with confidence 1 (the
SDK's "low, predicted"); every other joint has confidence 2 ("medium"), 0 where GEM-X's projection is invalid.
Orientations are not given (``orientation_wxyz`` null): SOMA's joint frames are not the Kinect SDK's.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

SOMA77 = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye",
    *(f"Left{n}" for n in (
        "Shoulder", "Arm", "ForeArm", "Hand", "HandThumb1", "HandThumb2", "HandThumb3", "HandThumbEnd",
        "HandIndex1", "HandIndex2", "HandIndex3", "HandIndex4", "HandIndexEnd",
        "HandMiddle1", "HandMiddle2", "HandMiddle3", "HandMiddle4", "HandMiddleEnd",
        "HandRing1", "HandRing2", "HandRing3", "HandRing4", "HandRingEnd",
        "HandPinky1", "HandPinky2", "HandPinky3", "HandPinky4", "HandPinkyEnd")),
    *(f"Right{n}" for n in (
        "Shoulder", "Arm", "ForeArm", "Hand", "HandThumb1", "HandThumb2", "HandThumb3", "HandThumbEnd",
        "HandIndex1", "HandIndex2", "HandIndex3", "HandIndex4", "HandIndexEnd",
        "HandMiddle1", "HandMiddle2", "HandMiddle3", "HandMiddle4", "HandMiddleEnd",
        "HandRing1", "HandRing2", "HandRing3", "HandRing4", "HandRingEnd",
        "HandPinky1", "HandPinky2", "HandPinky3", "HandPinky4", "HandPinkyEnd")),
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
)
assert len(SOMA77) == 77

K4ABT_BODY_32 = (  # the Body Tracking SDK's joint order (k4abt_joint_id_t), the names of xblick-azure-kinect's JSON
    "Pelvis", "SpineNaval", "SpineChest", "Neck", "ClavicleLeft", "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "HandtipLeft", "ThumbLeft", "ClavicleRight", "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HandtipRight", "ThumbRight", "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft", "HipRight", "KneeRight",
    "AnkleRight", "FootRight", "Head", "Nose", "EyeLeft", "EarLeft", "EyeRight", "EarRight",
)
FROM_SOMA = {  # K4ABT joint -> SOMA joint
    "Pelvis": "Hips", "SpineNaval": "Spine1", "SpineChest": "Chest", "Neck": "Neck1", "Head": "Head",
    "EyeLeft": "LeftEye", "EyeRight": "RightEye",
    **{f"{k}{side}": f"{side}{s}" for side in ("Left", "Right") for k, s in (
        ("Clavicle", "Shoulder"), ("Shoulder", "Arm"), ("Elbow", "ForeArm"), ("Wrist", "Hand"),
        ("Hand", "HandMiddle1"), ("Handtip", "HandMiddleEnd"), ("Thumb", "HandThumbEnd"),
        ("Hip", "Leg"), ("Knee", "Shin"), ("Ankle", "Foot"), ("Foot", "ToeBase"))},
}
DERIVED = ("Nose", "EarLeft", "EarRight")
assert set(FROM_SOMA) | set(DERIVED) == set(K4ABT_BODY_32)
INDEX = {n: i for i, n in enumerate(SOMA77)}
NOSE_AHEAD, EAR_OUT = 0.04, 0.075  # [m]


def check_soma77(rig_usda: Path):
    """SOMA77 against the rig's joints (without its root and twist joints); ValueError if they differ."""
    text = Path(rig_usda).read_text()
    m = re.search(r"joints = \[(.*?)\]", text, re.S)
    if m is None:
        raise ValueError(f"{rig_usda}: no joints list")
    names = [n.split("/")[-1] for n in re.findall(r'"([^"]+)"', m.group(1))]
    public = tuple(n for n in names[1:] if "Twist" not in n)
    if public != SOMA77:
        raise ValueError(f"{rig_usda}: its joints are not the SOMA 0.3.3 rig's GEM-X was trained with")


def k4abt_joints(joints: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(77, 3) SOMA joints [m] and (77,) validity -> (32, 3) K4ABT joints [m] and (32,) confidence (SDK levels)."""
    out = np.zeros((32, 3))
    confidence = np.zeros(32, dtype=int)
    for i, name in enumerate(K4ABT_BODY_32):
        if name in FROM_SOMA:
            j = INDEX[FROM_SOMA[name]]
            out[i] = joints[j]
            confidence[i] = 2 if valid[j] else 0
    head, le, re_ = joints[INDEX["Head"]], joints[INDEX["LeftEye"]], joints[INDEX["RightEye"]]
    eyes = (le + re_) / 2
    ahead = eyes - head
    ahead -= ahead @ (le - re_) / max(float((le - re_) @ (le - re_)), 1e-12) * (le - re_)  # off the eyes' axis
    across = (le - re_) / max(float(np.linalg.norm(le - re_)), 1e-9)
    derived = {"Nose": eyes + NOSE_AHEAD * ahead / max(float(np.linalg.norm(ahead)), 1e-9),
               "EarLeft": head + EAR_OUT * across, "EarRight": head - EAR_OUT * across}
    ok = bool(valid[INDEX["Head"]] and valid[INDEX["LeftEye"]] and valid[INDEX["RightEye"]])
    for name, p in derived.items():
        i = K4ABT_BODY_32.index(name)
        out[i] = p
        confidence[i] = 1 if ok else 0
    return out, confidence


def bodies_message(stamp_ns: int, frame_id: str, latency_ms: float, bodies: list[dict]) -> bytes:
    """xblick-azure-kinect's bodies JSON (model K4ABT_BODY_32, positions in mm in ``frame_id``); per body also
    ``soma77_mm`` (GEM-X's own 77 joints, SOMA77's order) and ``depth_fit`` (the depth correction, depth.py).
    ``bodies``: dicts of id, joints (77, 3) [m], valid (77,), depth_fit (dict)."""
    out = []
    for b in bodies:
        k4a, confidence = k4abt_joints(b["joints"], b["valid"])
        out.append({
            "id": int(b["id"]),
            "joints": {name: {"position_mm": (k4a[i] * 1000).round(1).tolist(), "orientation_wxyz": None,
                              "confidence": int(confidence[i])} for i, name in enumerate(K4ABT_BODY_32)},
            "soma77_mm": (np.asarray(b["joints"]) * 1000).round(1).tolist(),
            "depth_fit": b["depth_fit"],
        })
    return json.dumps({"v": 1, "model": "K4ABT_BODY_32", "source": "gem-x", "stamp_ns": int(stamp_ns),
                       "frame_id": frame_id, "latency_ms": round(float(latency_ms), 1), "bodies": out}).encode()
