# vision.py
"""
TF-free vision utilities for Smart Attendance:
- Face registration & recognition using MediaPipe FaceMesh landmarks
- Simple emotion / engagement classification using eye + mouth openness + head turn
"""

import os
import uuid
import pickle
import itertools
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# Update encodings format:
# encodings = {
#   "role:external_id": {
#       "external_id": "...",
#       "name": "...",
#       "role": "...",
#       "image_paths": [...],
#       "embeddings": [np.ndarray (1D)],
#       "mean_embedding": np.ndarray (1D)
#   },
#   ...
# }

def _ensure_embedding_list(rec):
    """Helper: convert single 'embedding' entry to list if old format exists."""
    if rec is None:
        return rec
    if "embeddings" not in rec and "embedding" in rec:
        rec["embeddings"] = [rec.pop("embedding")]
    if "embeddings" not in rec:
        rec["embeddings"] = []
    return rec

def save_face_image(image: Image.Image, role: str, external_id: str, name: str, save_only: bool = False) -> str:
    """
    Save image file to appropriate folder. If save_only==False original behavior:
    compute embedding and store. When save_only==True we only save the file (used for multi-capture flow).
    """
    safe_name = name.replace(" ", "_")
    base_dir = STUDENTS_DIR if role == "student" else TEACHERS_DIR
    filename = f"{external_id}_{safe_name}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(base_dir, filename)

    image_rgb = image.convert("RGB")
    image_rgb.save(path)

    if save_only:
        return path

    emb = _get_embedding_from_pil(image)
    if emb is None:
        print("[WARN] No face for registering:", external_id, name)
        return path

    encodings = _load_encodings()
    key = f"{role}:{external_id}"
    rec = encodings.get(key, {"external_id": external_id, "name": name, "role": role, "image_paths": [], "embeddings": []})
    # ensure list
    rec = _ensure_embedding_list(rec)
    rec["image_paths"].append(path)
    rec["embeddings"].append(emb)
    # update mean embedding
    rec["mean_embedding"] = np.mean(np.stack(rec["embeddings"], axis=0), axis=0).astype("float32")
    encodings[key] = rec
    _save_encodings(encodings)
    return path


def add_multiple_embeddings_for_user(role: str, external_id: str, name: str, image_paths: list):
    """
    Given multiple saved image files for a user, compute embeddings for each file,
    append them to encodings, and recompute mean_embedding.
    """
    encodings = _load_encodings()
    key = f"{role}:{external_id}"
    rec = encodings.get(key, {"external_id": external_id, "name": name, "role": role, "image_paths": [], "embeddings": []})
    rec = _ensure_embedding_list(rec)

    for p in image_paths:
        try:
            pil = Image.open(p).convert("RGB")
            emb = _get_embedding_from_pil(pil)
            if emb is not None:
                rec["image_paths"].append(p)
                rec["embeddings"].append(emb)
        except Exception as e:
            print("[WARN] add_multiple_embeddings_for_user error:", e, p)

    if rec["embeddings"]:
        rec["mean_embedding"] = np.mean(np.stack(rec["embeddings"], axis=0), axis=0).astype("float32")

    encodings[key] = rec
    _save_encodings(encodings)
    return True


def recognize_person(image: Image.Image, role: Optional[str] = None) -> Optional[Tuple[str, str, float]]:
    """
    Match given image with known faces using MediaPipe landmark embeddings.
    We now compare to every saved embedding for each user and use min-distance.
    """
    encodings = _load_encodings()
    if not encodings:
        return None

    emb = _get_embedding_from_pil(image)
    if emb is None:
        return None

    best_overall = None  # (dist, meta)
    for key, rec in encodings.items():
        if role is not None and rec.get("role") != role:
            continue
        rec = _ensure_embedding_list(rec)
        # if they have embeddings list, use those; else try mean_embedding
        d = None
        if rec.get("embeddings"):
            embs = np.stack(rec["embeddings"], axis=0)
            dists = np.linalg.norm(embs - emb[None, :], axis=1)
            dist = float(np.min(dists))
        elif rec.get("mean_embedding") is not None:
            dist = float(np.linalg.norm(rec["mean_embedding"] - emb))
        else:
            continue

        if best_overall is None or dist < best_overall[0]:
            best_overall = (dist, rec)

    if best_overall is None:
        return None

    best_dist, best_meta = best_overall
    return best_meta["external_id"], best_meta["name"], best_dist

# -------- Heuristic thresholds (easy to tune) --------
# Eye openness thresholds
SLEEPY_EYE = 0.012      # < this: eyes almost closed
LOW_EYE = 0.018         # < this: low attention

# Mouth openness thresholds
SPEAKING_MOUTH = 0.045  # speaking / normal
YAWN_MOUTH = 0.085      # very open => likely yawn

# Head turn thresholds (how much left/right allowed)
ATTENTIVE_TURN = 0.06   # in Lecture, must stay close to forward
RELAXED_TURN = 0.12     # in Discussion/Activity, more freedom

# Engagement points
POINT_STRONG_POSITIVE = 2
POINT_MILD = 0
POINT_NEGATIVE = -2
POINT_VERY_NEGATIVE = -3

# -------- Paths --------
FACES_DB_DIR = "faces_db"
STUDENTS_DIR = os.path.join(FACES_DB_DIR, "students")
TEACHERS_DIR = os.path.join(FACES_DB_DIR, "teachers")
ENCODINGS_PATH = os.path.join(FACES_DB_DIR, "encodings.pkl")

os.makedirs(STUDENTS_DIR, exist_ok=True)
os.makedirs(TEACHERS_DIR, exist_ok=True)

# -------- MediaPipe setup --------
# Pylance often can't see these dynamic attributes, so we ignore type warnings.
mp_solutions = mp.solutions  # type: ignore[attr-defined]
mp_face_mesh = mp_solutions.face_mesh  # type: ignore[attr-defined]

face_mesh_solver = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)

# Landmark groups for eyes & lips (we will use their indices)
LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))   # type: ignore[attr-defined]
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE))) # type: ignore[attr-defined]
LIPS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))           # type: ignore[attr-defined]


# -------- Encodings storage --------
def _load_encodings() -> Dict[str, Any]:
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_encodings(encodings: Dict[str, Any]) -> None:
    os.makedirs(FACES_DB_DIR, exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(encodings, f)


# -------- Core embedding / geometry --------
def _pil_to_rgb_ndarray(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to BGR numpy array (for OpenCV)."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _get_face_landmarks(image_bgr: np.ndarray):
    """Run MediaPipe FaceMesh and return first face landmarks or None."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh_solver.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]


def _compute_embedding_from_landmarks(landmarks) -> Optional[np.ndarray]:
    """Flattened, normalized landmark coordinates as a 'face embedding'."""
    if landmarks is None:
        return None

    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]

    coords = np.stack([xs, ys], axis=1)  # (N, 2)
    mean = coords.mean(axis=0, keepdims=True)
    std = coords.std(axis=0, keepdims=True) + 1e-6
    coords_norm = (coords - mean) / std
    emb = coords_norm.flatten().astype("float32")
    return emb


def _get_embedding_from_pil(image: Image.Image) -> Optional[np.ndarray]:
    bgr = _pil_to_rgb_ndarray(image)
    landmarks = _get_face_landmarks(bgr)
    if landmarks is None:
        return None
    return _compute_embedding_from_landmarks(landmarks)


def _eye_mouth_openness(landmarks) -> Tuple[float, float]:
    """
    Compute simple 'open' scores for eyes and mouth based on landmark spread in Y.
    """
    if landmarks is None:
        return 0.0, 0.0

    lms = landmarks.landmark

    eye_ys = [lms[i].y for i in LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES]
    lips_ys = [lms[i].y for i in LIPS_INDEXES]

    eye_open = float(max(eye_ys) - min(eye_ys)) if eye_ys else 0.0
    mouth_open = float(max(lips_ys) - min(lips_ys)) if lips_ys else 0.0

    return eye_open, mouth_open


def _head_turn_score(landmarks) -> float:
    """
    Approximate how much the head is turned left/right.
    We use nose vs average eye x-position as a simple proxy.
    Higher value => more turned.
    """
    if landmarks is None:
        return 0.0

    lms = landmarks.landmark

    nose_idx = 1  # approximate nose tip index in FaceMesh
    eye_xs = [lms[i].x for i in LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES]
    if not eye_xs:
        return 0.0

    eye_center_x = float(sum(eye_xs) / len(eye_xs))
    nose_x = float(lms[nose_idx].x)

    turn = abs(nose_x - eye_center_x)
    return turn


# -------- Public API used by app.py --------
def save_face_image(image: Image.Image, role: str, external_id: str, name: str) -> str:
    """
    Save captured face image to faces_db/{role}/ and store its embedding
    in encodings.pkl for later recognition.
    Returns saved image path.
    """
    safe_name = name.replace(" ", "_")
    if role == "student":
        base_dir = STUDENTS_DIR
    else:
        base_dir = TEACHERS_DIR

    filename = f"{external_id}_{safe_name}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(base_dir, filename)

    image_rgb = image.convert("RGB")
    image_rgb.save(path)

    emb = _get_embedding_from_pil(image)
    if emb is None:
        print("[WARN] No face detected while registering:", external_id, name)
        return path

    encodings = _load_encodings()
    key = f"{role}:{external_id}"
    encodings[key] = {
        "external_id": external_id,
        "name": name,
        "role": role,
        "embedding": emb,
    }
    _save_encodings(encodings)

    return path


def recognize_person(
    image: Image.Image, role: Optional[str] = None
) -> Optional[Tuple[str, str, float]]:
    """
    Match given image with known faces using MediaPipe landmark embeddings.
    Returns (external_id, name, distance) or None if no good match.
    """
    encodings = _load_encodings()
    if not encodings:
        return None

    emb = _get_embedding_from_pil(image)
    if emb is None:
        return None

    keys = []
    embs = []
    meta = []

    for key, rec in encodings.items():
        if role is not None and rec.get("role") != role:
            continue
        if "embedding" not in rec:
            continue
        keys.append(key)
        embs.append(rec["embedding"])
        meta.append(rec)

    if not embs:
        return None

    embs = np.stack(embs, axis=0)  # (M, D)

    dists = np.linalg.norm(embs - emb[None, :], axis=1)
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    best_meta = meta[best_idx]

    ext_id = best_meta["external_id"]
    name = best_meta["name"]

    return ext_id, name, best_dist


def detect_emotion(image: Image.Image, mode: str = "Lecture") -> Optional[str]:
    """
    Heuristic engagement estimation based on:
    - eye openness (sleepy vs awake)
    - mouth openness (speaking / yawning)
    - head turn (looking forward vs away)
    and classroom mode (Lecture / Discussion / Activity).

    Returns labels like:
      "attentive", "attentive_speaking", "attentive_discussion",
      "sleepy", "distracted", "looking_away", "yawning", "offtask_talking", etc.
    """
    bgr = _pil_to_rgb_ndarray(image)
    landmarks = _get_face_landmarks(bgr)
    if landmarks is None:
        return None

    eye_open, mouth_open = _eye_mouth_openness(landmarks)
    head_turn = _head_turn_score(landmarks)

    # ---- Tunable thresholds ----
    SLEEPY_EYE = 0.012
    LOW_EYE = 0.018

    SPEAKING_MOUTH = 0.045
    YAWN_MOUTH = 0.085

    # How much head turn we allow
    ATTENTIVE_TURN = 0.06     # roughly looking forward
    RELAXED_TURN = 0.12       # more freedom in discussion / activity

    # ---- 1) Eye-based checks: sleepy vs awake ----
    if eye_open < SLEEPY_EYE:
        if mouth_open > YAWN_MOUTH:
            return "yawning"
        return "sleepy"
    elif eye_open < LOW_EYE:
        base_state = "low_attention"
    else:
        base_state = "awake"

    # ---- 2) Head direction: forward vs away ----
    if mode == "Lecture":
        turn_limit = ATTENTIVE_TURN
    else:
        turn_limit = RELAXED_TURN

    looking_away = head_turn > turn_limit

    # ---- 3) Mouth: speaking vs neutral ----
    if mouth_open > YAWN_MOUTH:
        mouth_state = "yawning"
    elif mouth_open > SPEAKING_MOUTH:
        mouth_state = "speaking"
    else:
        mouth_state = "neutral"

    # ---- 4) Combine into final label ----

    # If base state was sleepy-ish, keep that
    if base_state == "low_attention" and looking_away:
        return "looking_away"

    if base_state == "awake":
        if not looking_away:
            # Facing forward (or reasonable)
            if mouth_state == "speaking":
                if mode == "Discussion":
                    return "attentive_discussion"
                elif mode == "Activity":
                    return "active"
                else:
                    return "attentive_speaking"
            elif mouth_state == "yawning":
                return "yawning"
            else:
                return "attentive"
        else:
            # Awake but looking away
            if mouth_state == "speaking":
                # could be talking to friend
                return "offtask_talking"
            else:
                return "looking_away"

    # Fallback for low_attention but not too bad
    if base_state == "low_attention":
        return "low_attention"

    return "distracted"


def emotion_to_points(emotion: Optional[str]) -> int:
    """
    Map emotion/engagement label to points.

    Scheme:
      - Strong positive focus:
          "attentive", "attentive_speaking",
          "attentive_discussion", "active" -> +2
      - Mild / okay:
          "low_attention", "neutral" -> 0
      - Negative:
          "looking_away", "distracted", "offtask_talking" -> -2
      - Very negative:
          "sleepy", "yawning" -> -3
      - None/unknown -> 0
    """
    if emotion is None:
        return 0

    e = emotion.lower()

    strong_positive = {
        "attentive",
        "attentive_speaking",
        "attentive_discussion",
        "active",
    }
    mild = {
        "low_attention",
        "neutral",
    }
    negative = {
        "looking_away",
        "distracted",
        "offtask_talking",
    }
    very_negative = {
        "sleepy",
        "yawning",
    }

    if e in strong_positive:
        return POINT_STRONG_POSITIVE
    if e in mild:
        return POINT_MILD
    if e in negative:
        return POINT_NEGATIVE
    if e in very_negative:
        return POINT_VERY_NEGATIVE

    return 0
