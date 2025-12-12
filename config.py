# config.py
CLASSROOM_MODES = ["Lecture", "Discussion", "Activity"]

POINTS_STUDENT = {
    "attentive": 2,
    "neutral": 0,
    "distracted": -2,
    "sleepy": -3,
}

POINTS_TEACHER = {
    "on_time": 5,
    "late": -3,
    "idle": -2,
}

# Distance threshold for DeepFace face match
FACE_MATCH_THRESHOLD = 0.4  # smaller = stricter
