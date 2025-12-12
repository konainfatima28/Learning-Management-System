# app.py
# app.py

import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

import os
from datetime import datetime, timedelta
import io
import time
import random           

import cv2           
import pandas as pd
import streamlit as st
from PIL import Image

from db import (
    init_db,
    get_or_create_user,
    create_session,
    set_session_actual_start,
    log_attendance,
    log_teacher_metric,
    get_attendance_df,
    SessionLocal,
    Session as SessionModel,
    User,
)

from vision import save_face_image, recognize_person, detect_emotion, emotion_to_points
from config import CLASSROOM_MODES, FACE_MATCH_THRESHOLD

# ---- Engagement / attendance scoring config ----
CLASS_DURATION_MIN = 55          # total class length in minutes
SLOT_MIN = 5                     # 1 engagement sample ≈ 5 minutes
SLOT_POINTS = 2                  # max +2 per slot if attentive
MAX_SLOTS = CLASS_DURATION_MIN // SLOT_MIN  # 11 slots in 55 minutes
MAX_POINTS_PER_SESSION = MAX_SLOTS * SLOT_POINTS  # 22 points
ATTENDANCE_THRESHOLD = 0.6       # 60% required for full presence
PARTIAL_ATTENDANCE_VALUE = 0.6   # partial presence weight
FULL_ATTENDANCE_VALUE = 1.0      # full presence weight
AUTO_LOG_INTERVAL_SEC = SLOT_MIN * 60  # 5 minutes per slot (300 seconds)

# ----------------- Demo data seeding -----------------
def seed_demo_data():
    """
    Create fake teachers, students, sessions and attendance records
    so that dashboards look rich for demo.

    Returns a small summary dict with counts.
    """
    # If there is already plenty of data, avoid reseeding again and again
    existing = get_attendance_df()
    if not existing.empty and len(existing) > 80:
        return {"num_sessions": 0, "num_students": 0, "num_records": 0}

    # Demo teachers and students
    demo_teachers = [
        ("T_DEMO1", "Dr. Sharma"),
        ("T_DEMO2", "Prof. Verma"),
    ]
    demo_students = [
        ("S_DEMO01", "Aarav Gupta"),
        ("S_DEMO02", "Siya Mehta"),
        ("S_DEMO03", "Kabir Singh"),
        ("S_DEMO04", "Ananya Rao"),
        ("S_DEMO05", "Rahul Jain"),
        ("S_DEMO06", "Ishita Malhotra"),
        ("S_DEMO07", "Devansh Patel"),
        ("S_DEMO08", "Muskan Ali"),
    ]

    # Create / fetch users
    teacher_users = [
        get_or_create_user(external_id=tid, name=tname, role="teacher")
        for (tid, tname) in demo_teachers
    ]
    student_users = [
        get_or_create_user(external_id=sid, name=sname, role="student")
        for (sid, sname) in demo_students
    ]

    now = datetime.now()
    sessions = []

    # Create a few demo sessions in last 3 days
    for i in range(3):
        day = now - timedelta(days=(3 - i))
        sched = datetime(day.year, day.month, day.day, 9, 0)
        teacher = teacher_users[i % len(teacher_users)]
        sess = create_session(
            name=f"AI-ML Demo Class {i + 1}",
            scheduled_start=sched,
            teacher_id=teacher.id,
        )
        sessions.append((sess, teacher))

        # Give teacher some punctuality metric (for flavor)
        log_teacher_metric(
            session_id=sess.id,
            teacher_id=teacher.id,
            metric_type="on_time",
            points=5,
        )

    emotions_students = [
        "attentive",
        "attentive_discussion",
        "active",
        "low_attention",
        "looking_away",
        "offtask_talking",
        "sleepy",
        "yawning",
    ]
    emotions_teachers = [
        "attentive",
        "active",
        "low_attention",
    ]

    num_records = 0

    # Create attendance logs for each session
    for sess, teacher in sessions:
        # Teacher engagement samples
        for _ in range(4):
            emo = random.choice(emotions_teachers)
            pts = emotion_to_points(emo)
            log_attendance(
                session_id=sess.id,
                user_id=teacher.id,
                emotion=emo,
                engagement_points=pts,
                is_teacher=True,
            )
            num_records += 1

        # Student engagement samples
        for stu in student_users:
            num_samples = random.randint(5, 11)  # like 5–11 × 5-minute slots
            for _ in range(num_samples):
                emo = random.choice(emotions_students)
                pts = emotion_to_points(emo)
                log_attendance(
                    session_id=sess.id,
                    user_id=stu.id,
                    emotion=emo,
                    engagement_points=pts,
                    is_teacher=False,
                )
                num_records += 1

    return {
        "num_sessions": len(sessions),
        "num_students": len(student_users),
        "num_records": num_records,
    }


# ----------------- INIT -----------------
init_db()
st.set_page_config(page_title="Smart Attendance System", layout="wide")

# ----- Dark theme CSS (simple) -----
dark_css = """
<style>
.stApp {
    background-color: #0e1117;
    color: #f9fafb;
}
[data-testid="stSidebar"] {
    background-color: #020617;
}
h1, h2, h3, h4, h5, h6, label, span, p {
    color: #e5e7eb !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    color: #e5e7eb !important;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

st.title("📚 Automated Lecture Monitoring Using Deep Learning")
st.caption("Automated attendance + engagement analytics for AI-driven classrooms.")

# ----------------- Sidebar: Session Controls -----------------
st.sidebar.header("Class Session Controls")

with st.sidebar.form("session_form"):
    class_name = st.text_input("Class / Course Name", value="AI-ML 7th Sem")
    teacher_external_id = st.text_input("Teacher ID", value="T001")
    teacher_name = st.text_input("Teacher Name", value="Demo Teacher")
    scheduled_time_str = st.text_input("Scheduled Start (HH:MM)", value="09:00")
    start_session_btn = st.form_submit_button("Create / Load Session")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "teacher_user_id" not in st.session_state:
    st.session_state.teacher_user_id = None
if "classroom_mode" not in st.session_state:
    st.session_state.classroom_mode = "Lecture"
if "teacher_arrival_logged" not in st.session_state:
    st.session_state.teacher_arrival_logged = False
if "capture_buffer" not in st.session_state:
    st.session_state.capture_buffer = []  # will store raw image bytes

if start_session_btn:
    # Parse scheduled time as today + given HH:MM
    try:
        today = datetime.now().date()
        hour, minute = map(int, scheduled_time_str.split(":"))
        scheduled_dt = datetime(today.year, today.month, today.day, hour, minute)
    except Exception:
        scheduled_dt = None

    teacher_user = get_or_create_user(
        external_id=teacher_external_id,
        name=teacher_name,
        role="teacher",
    )
    sess = create_session(
        name=class_name,
        scheduled_start=scheduled_dt,
        teacher_id=teacher_user.id,
    )
    st.session_state.session_id = sess.id
    st.session_state.teacher_user_id = teacher_user.id
    st.session_state.teacher_arrival_logged = False
    st.success(f"Session created: {class_name} (ID: {sess.id})")

# Mode selector
st.sidebar.subheader("Classroom Mode")
mode = st.sidebar.radio("Select Mode", options=CLASSROOM_MODES, index=0)
st.session_state.classroom_mode = mode

st.sidebar.info(
    f"Current Mode: **{st.session_state.classroom_mode}**\n\n"
    "- Lecture: strict engagement\n"
    "- Discussion: relax teacher activity\n"
    "- Activity: no penalty for movement"
)

# -------------- Demo Mode button --------------
st.sidebar.markdown("---")
if st.sidebar.button("⚡ Demo Mode: Seed Sample Data"):
    with st.spinner("Seeding demo data into database..."):
        info = seed_demo_data()

    if info["num_sessions"] == 0:
        st.sidebar.warning("Demo data already present (skipping heavy reseed).")
    else:
        st.sidebar.success(
            f"Demo data created: {info['num_sessions']} sessions, "
            f"{info['num_students']} students, "
            f"{info['num_records']} attendance records."
        )

# ----------------- Realtime video processor -----------------
class LiveAttendanceProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.last_name = None
        self.last_id = None
        self.last_emotion = None
        self.last_points = None
        # track last auto-log time per external_id
        self._last_log_ts = {}

    def recv(self, frame):
        # Convert frame to BGR image
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Current classroom mode from Streamlit state
        mode = st.session_state.get("classroom_mode", "Lecture")

        # Try to recognize person
        try:
            recog = recognize_person(pil_img, role=None)
        except Exception:
            recog = None

        if recog is not None:
            ext_id, name, dist = recog
            emotion = detect_emotion(pil_img, mode=mode) or "unknown"
            pts = emotion_to_points(emotion)

            self.last_name = name
            self.last_id = ext_id
            self.last_emotion = emotion
            self.last_points = pts

            # -------- AUTO SLOT LOGGING (every 5 min per student) --------
            try:
                session_id = st.session_state.get("session_id")
                auto_enabled = st.session_state.get("auto_log_enabled", False)

                if session_id is not None and auto_enabled:
                    now = time.time()
                    last_ts = self._last_log_ts.get(ext_id, 0)

                    # log at most once per AUTO_LOG_INTERVAL_SEC per student
                    if now - last_ts >= AUTO_LOG_INTERVAL_SEC:
                        # assume live stream mainly tracks students
                        user = get_or_create_user(
                            external_id=ext_id,
                            name=name,
                            role="student",
                        )

                        log_attendance(
                            session_id=session_id,
                            user_id=user.id,
                            emotion=emotion,
                            engagement_points=pts,
                            is_teacher=False,
                        )

                        self._last_log_ts[ext_id] = now
            except Exception:
                # never kill the stream if DB or state has issues
                pass

            overlay = (
                f"{name} ({ext_id}) | {emotion} | {pts:+d} pts | "
                f"mode={mode}"
            )
            color = (0, 255, 0) if pts > 0 else ((0, 255, 255) if pts == 0 else (0, 0, 255))
        else:
            overlay = "No recognized face"
            color = (200, 200, 200)

        # Draw overlay on frame
        cv2.rectangle(img, (5, 5), (900, 45), (0, 0, 0), thickness=-1)
        cv2.putText(
            img,
            overlay,
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------- Tabs -----------------
tab1, tab2, tab3 = st.tabs(["➕ Register Users", "🎥 Run Class Live", "📊 Dashboard & Reports"])

# ---------- TAB 1: Register Students & Teachers ----------
with tab1:
    st.subheader("Register Students & Teachers (Face + ID)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1. User Details")
        reg_role = st.selectbox("Role", ["student", "teacher"])
        reg_name = st.text_input("Name")
        reg_external_id = st.text_input("ID (Roll No / Emp ID)")

    with col2:
        st.markdown("### 2. Capture Faces (take multiple photos for accuracy)")
        # Camera capture input (single shot each press)
        cam_image = st.camera_input("Take a photo (frontal headshot)", key="reg_cam")

        if "reg_captures" not in st.session_state:
            st.session_state.reg_captures = []  # list of image bytes

        cap_col1, cap_col2 = st.columns([3,1])
        with cap_col2:
            if st.button("➕ Add Capture"):
                if cam_image is None:
                    st.error("Take a photo first")
                else:
                    st.session_state.reg_captures.append(cam_image.getvalue())
                    st.success(f"Capture added — total: {len(st.session_state.reg_captures)}")

            if st.button("🧹 Clear Captures"):
                st.session_state.reg_captures = []
                st.info("All registration captures cleared")

        # Show thumbnails of captures
        if st.session_state.get("reg_captures"):
            thumbs = st.session_state.reg_captures
            st.markdown("**Captured images (click to remove)**")
            cols = st.columns(6)
            for idx, raw in enumerate(thumbs):
                try:
                    img = Image.open(io.BytesIO(raw))
                    # display in a small box with a remove button
                    c = cols[idx % 6]
                    c.image(img, width=100)
                    if c.button(f"Remove {idx+1}", key=f"rm_{idx}"):
                        st.session_state.reg_captures.pop(idx)
                        st.experimental_rerun()
                except Exception:
                    pass

    # Save registration using all captured images
    if st.button("Save Registration (Use all captures)"):
        if not reg_name or not reg_external_id:
            st.error("Please fill name and ID.")
        elif not st.session_state.get("reg_captures"):
            st.error("No captures. Take and Add at least 3 photos from slightly different angles.")
        else:
            # Save each capture and add embeddings to encodings via vision helper
            saved_paths = []
            success_count = 0
            for i, raw in enumerate(st.session_state.reg_captures):
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                # call helper that saves and updates encodings for a single image
                path = save_face_image(
                    image=img,
                    role=reg_role,
                    external_id=reg_external_id,
                    name=reg_name,
                    save_only=True,   # we'll tell vision to save without overwriting meta
                )
                saved_paths.append(path)
                success_count += 1

            # After saving individual images, call a helper to aggregate embeddings
            from vision import add_multiple_embeddings_for_user
            add_multiple_embeddings_for_user(
                role=reg_role,
                external_id=reg_external_id,
                name=reg_name,
                image_paths=saved_paths,
            )

            # Make or get DB user entry
            user = get_or_create_user(
                external_id=reg_external_id,
                name=reg_name,
                role=reg_role,
            )

            st.success(f"{reg_role.title()} '{reg_name}' registered with {success_count} captures.")
            st.caption(f"Files saved: {len(saved_paths)}")
            # clear captures buffer
            st.session_state.reg_captures = []

# ---------- TAB 2: Run Class Live ----------
with tab2:
    st.subheader("Live Classroom Analysis (Multi-Capture for Accuracy)")

    if st.session_state.session_id is None:
        st.warning("Create or load a session from the sidebar first.")
    else:
        st.write(f"**Active Session ID:** {st.session_state.session_id}")
        st.write(f"**Mode:** {st.session_state.classroom_mode}")
        st.info(
            "ℹ️ **How it works**:\n"
            "- Realtime preview (below) shows live face, emotion and engagement points.\n"
            "- Every time you click **Analyze Batch & Mark Attendance**, the system treats it "
            "as ~5 minutes of class and updates engagement for that student.\n"
            "- At the end, the 55-min class is evaluated based on the percentage of possible points."
        )

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### Capture Frames")
            live_image = st.camera_input("Point camera towards class", key="live_camera")

            if st.button("➕ Add Frame to Batch"):
                if live_image is None:
                    st.error("Capture a frame first.")
                else:
                    # Store raw bytes in buffer
                    st.session_state.capture_buffer.append(live_image.getvalue())
                    st.success(
                        f"Frame added. Total frames in batch: {len(st.session_state.capture_buffer)}"
                    )

            if st.button("🧹 Clear Batch"):
                st.session_state.capture_buffer = []
                st.info("Capture batch cleared.")

            st.info(f"Frames currently in batch: **{len(st.session_state.capture_buffer)}**")

        with colB:
            st.markdown("### Detection Controls")
            target_role = st.selectbox(
                "Detect:",
                ["student", "teacher", "auto (any)"],
                index=2,
            )

            analyze_btn = st.button("✅ Analyze Batch & Mark Attendance")

        # ------------- Batch analysis logic -------------
        if analyze_btn:
            if not st.session_state.capture_buffer:
                st.error("No frames in batch. Capture and add frames first.")
            else:
                # Map UI role to function role
                role_for_recog = None
                if target_role == "student":
                    role_for_recog = "student"
                elif target_role == "teacher":
                    role_for_recog = "teacher"

                per_frame_results = []

                for i, raw_bytes in enumerate(st.session_state.capture_buffer):
                    img = Image.open(io.BytesIO(raw_bytes))

                    recog_result = recognize_person(img, role=role_for_recog)
                    if recog_result is None:
                        per_frame_results.append(
                            {
                                "frame": i + 1,
                                "recognized": False,
                                "external_id": None,
                                "name": None,
                                "distance": None,
                                "emotion": None,
                                "points": 0,
                            }
                        )
                        continue

                    ext_id, name, distance = recog_result
                    emotion = detect_emotion(img, mode=st.session_state.classroom_mode)
                    points = emotion_to_points(emotion)

                    per_frame_results.append(
                        {
                            "frame": i + 1,
                            "recognized": True,
                            "external_id": ext_id,
                            "name": name,
                            "distance": distance,
                            "emotion": emotion,
                            "points": points,
                        }
                    )

                # Filter only recognized frames
                recognized_frames = [r for r in per_frame_results if r["recognized"]]
                if not recognized_frames:
                    st.warning("No face recognized in any of the frames. Try again with clearer captures.")
                else:
                    # ----- Majority vote on external_id -----
                    counts = {}
                    for r in recognized_frames:
                        counts[r["external_id"]] = counts.get(r["external_id"], 0) + 1

                    majority_id = max(counts, key=counts.get)
                    majority_name = None
                    best_distance = None

                    # Find best (lowest) distance for majority_id
                    for r in recognized_frames:
                        if r["external_id"] == majority_id:
                            if best_distance is None or (
                                r["distance"] is not None and r["distance"] < best_distance
                            ):
                                best_distance = r["distance"]
                                majority_name = r["name"]

                    # Average engagement points for that person across frames
                    num_frames_for_majority = sum(
                        1 for r in recognized_frames if r["external_id"] == majority_id
                    )
                    avg_points = sum(
                        r["points"] for r in recognized_frames if r["external_id"] == majority_id
                    ) / max(1, num_frames_for_majority)

                    # Prepare best distance string safely
                    best_dist_str = f"{best_distance:.4f}" if best_distance is not None else "N/A"

                    st.success(
                        f"Final decision: **{majority_name} ({majority_id})** | "
                        f"frames agreeing: {counts[majority_id]} / {len(recognized_frames)} | "
                        f"best distance: {best_dist_str}"
                    )

                    # ----- Create / get user in DB -----
                    assumed_role = "teacher" if target_role == "teacher" else "student"
                    user = get_or_create_user(
                        external_id=majority_id,
                        name=majority_name,
                        role=assumed_role,
                    )

                    # ----- Teacher punctuality (first time only) -----
                    if user.role == "teacher" and not st.session_state.teacher_arrival_logged:
                        set_session_actual_start(st.session_state.session_id)
                        st.session_state.teacher_arrival_logged = True

                        db_local = SessionLocal()
                        try:
                            sess_obj = db_local.query(SessionModel).get(st.session_state.session_id)
                            if sess_obj and sess_obj.scheduled_start and sess_obj.actual_start:
                                delta = sess_obj.actual_start - sess_obj.scheduled_start

                                if delta <= timedelta(minutes=0):
                                    log_teacher_metric(
                                        session_id=sess_obj.id,
                                        teacher_id=user.id,
                                        metric_type="on_time",
                                        points=5,
                                    )
                                    st.success("Teacher on time (+5 points)")
                                else:
                                    log_teacher_metric(
                                        session_id=sess_obj.id,
                                        teacher_id=user.id,
                                        metric_type="late",
                                        points=-3,
                                    )
                                    st.warning("Teacher late (-3 points)")
                        finally:
                            db_local.close()

                    # ----- Classroom mode adjustment -----
                    final_points = avg_points
                    if st.session_state.classroom_mode == "Activity" and user.role == "student":
                        final_points = max(final_points, 0)

                    is_teacher_flag = user.role == "teacher"
                    record = log_attendance(
                        session_id=st.session_state.session_id,
                        user_id=user.id,
                        emotion="mixed",  # batch result; per-frame emotions below
                        engagement_points=final_points,
                        is_teacher=is_teacher_flag,
                    )

                    st.info(
                        f"Attendance logged at {record.timestamp}. "
                        f"Final engagement score (avg over batch): {final_points:.2f}"
                    )

                    # Show per-frame table
                    df_frames = pd.DataFrame(per_frame_results)
                    st.markdown("#### Per-frame details")
                    st.dataframe(df_frames, use_container_width=True)

                st.markdown("---")
                st.markdown("### 🎥 Realtime Preview (Continuous)")

                st.caption(
                    "This live feed shows who is detected, their emotion, and engagement points in real time.\n"
                    "- When **Auto logging** is ON, the system will log one engagement slot every 5 minutes "
                    "for each recognized student.\n"
                    "- This feeds directly into the 55-min / 60% attendance logic in the Session Summary."
                )

                auto_log_enabled = st.checkbox(
                    "✅ Enable auto logging from live stream (1 slot every 5 minutes per student)",
                    value=False,
                )
                st.session_state.auto_log_enabled = auto_log_enabled

                webrtc_streamer(
                    key="live_attendance_stream",
                    video_processor_factory=LiveAttendanceProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                )


# ---------- TAB 3: Analytics & Reports ----------
with tab3:
    st.subheader("📊 Analytics & Reports")

    df = get_attendance_df()

    if df.empty:
        st.info("No attendance records yet. Run a session first.")
    else:
        # Prepare time columns
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        # Main view selector
        view_mode = st.radio(
            "Choose view",
            ["Class Overview", "Teacher Dashboard", "Student Report", "Session Summary"],
            horizontal=True,
        )

        # ========== 1) CLASS OVERVIEW ==========
        if view_mode == "Class Overview":
            st.markdown("### 🏫 Class Overview (Daily Snapshot)")

            colf1, colf2 = st.columns(2)
            with colf1:
                all_dates = sorted(df["date"].unique())
                default_date = all_dates[-1]
                selected_date = st.date_input(
                    "Select Date",
                    value=default_date,
                )

            with colf2:
                role_filter = st.radio(
                    "Role Filter",
                    ["All", "Students", "Teachers"],
                    horizontal=True,
                )

            filtered = df[df["date"] == selected_date].copy()
            if role_filter == "Students":
                filtered = filtered[filtered["role"] == "student"]
            elif role_filter == "Teachers":
                filtered = filtered[filtered["role"] == "teacher"]

            if filtered.empty:
                st.warning("No records for selected date & filter.")
            else:
                # Quick stats
                st.markdown("### 📈 Quick Stats")

                total_records = len(filtered)
                total_students = filtered[filtered["role"] == "student"]["name"].nunique()
                total_teachers = filtered[filtered["role"] == "teacher"]["name"].nunique()
                avg_engagement = filtered["points"].mean()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Records", total_records)
                c2.metric("Students Present", total_students)
                c3.metric("Avg Engagement", f"{avg_engagement:.2f}")

                st.markdown("---")

                # Charts
                st.markdown("### 📊 Visual Insights")

                cA, cB = st.columns(2)

                with cA:
                    st.caption("Attendance count per person")
                    counts = (
                        filtered.groupby("name")["id"]
                        .count()
                        .sort_values(ascending=False)
                        .head(10)
                    )
                    if not counts.empty:
                        st.bar_chart(counts)
                    else:
                        st.write("Not enough data.")

                with cB:
                    st.caption("Average engagement per person")
                    avg_pts = (
                        filtered.groupby("name")["points"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(10)
                    )
                    if not avg_pts.empty:
                        st.bar_chart(avg_pts)
                    else:
                        st.write("Not enough data.")

                st.markdown("---")

                st.markdown("### 🙂 Emotion / State Distribution")
                emo_counts = filtered["emotion"].dropna().value_counts()
                if not emo_counts.empty:
                    st.bar_chart(emo_counts)
                else:
                    st.write("No emotion data yet.")

                st.markdown("---")

                st.markdown("### 🗒 Recent Records (Selected Day)")
                show_cols = ["timestamp", "name", "role", "session", "emotion", "points"]
                st.dataframe(
                    filtered.sort_values("timestamp", ascending=False)[show_cols].head(20),
                    use_container_width=True,
                )

                csv = filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Daily Report (CSV)",
                    csv,
                    "daily_attendance_report.csv",
                    "text/csv",
                )

        # ========== 2) TEACHER DASHBOARD ==========
        elif view_mode == "Teacher Dashboard":
            st.markdown("### 👨‍🏫 Teacher Dashboard")

            teacher_df = df[df["role"] == "teacher"].copy()
            if teacher_df.empty:
                st.info("No teacher records logged yet.")
            else:
                agg = (
                    teacher_df.groupby("name")
                    .agg(
                        total_records=("id", "count"),
                        unique_sessions=("session", "nunique"),
                        avg_points=("points", "mean"),
                        first_seen=("timestamp", "min"),
                        last_seen=("timestamp", "max"),
                    )
                    .sort_values("unique_sessions", ascending=False)
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Teachers", agg.shape[0])
                c2.metric("Total Teacher Records", len(teacher_df))
                c3.metric(
                    "Avg Teacher Engagement",
                    f"{teacher_df['points'].mean():.2f}",
                )

                st.markdown("#### Teacher Summary")
                st.dataframe(agg, use_container_width=True)

                st.markdown("#### Classes handled per teacher")
                st.bar_chart(agg["unique_sessions"])

        # ========== 3) STUDENT REPORT ==========
        elif view_mode == "Student Report":
            st.markdown("### 🎓 Student Engagement Report")

            student_df = df[df["role"] == "student"].copy()
            if student_df.empty:
                st.info("No student records yet.")
            else:
                students = sorted(student_df["name"].unique().tolist())
                selected_student = st.selectbox(
                    "Select student",
                    options=students,
                )

                stu = student_df[student_df["name"] == selected_student].copy()
                if stu.empty:
                    st.warning("No records for this student.")
                else:
                    total_classes = stu["session"].nunique()
                    total_days = stu["date"].nunique()
                    total_records = len(stu)
                    avg_points = stu["points"].mean()
                    last_seen = stu["timestamp"].max()

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Classes Attended", total_classes)
                    c2.metric("Days Present", total_days)
                    c3.metric("Avg Engagement", f"{avg_points:.2f}")
                    c4.metric("Last Seen", last_seen.strftime("%d-%m %H:%M"))

                    st.markdown("---")

                    st.markdown("#### Engagement Trend Over Time")
                    if len(stu) > 1:
                        trend_df = stu.sort_values("timestamp")[["timestamp", "points"]]
                        st.line_chart(trend_df, x="timestamp", y="points")
                    else:
                        st.info("Not enough data points to draw a trend line.")

                    st.markdown("#### Emotion / State Breakdown")
                    emo_counts_s = stu["emotion"].dropna().value_counts()
                    if not emo_counts_s.empty:
                        st.bar_chart(emo_counts_s)
                    else:
                        st.write("No emotion labels yet.")

                    st.markdown("#### Detailed Records")
                    show_cols = ["timestamp", "session", "emotion", "points"]
                    st.dataframe(
                        stu.sort_values("timestamp", ascending=False)[show_cols],
                        use_container_width=True,
                    )

                    csv_s = stu.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"📥 Download report for {selected_student}",
                        csv_s,
                        f"{selected_student.replace(' ', '_')}_report.csv",
                        "text/csv",
                    )

        # ========== 4) SESSION SUMMARY ==========
        elif view_mode == "Session Summary":
            st.markdown("### 🏷 Session-wise Summary")

            sessions = sorted(df["session"].dropna().unique().tolist())
            if not sessions:
                st.info("No session names found yet.")
            else:
                selected_session = st.selectbox(
                    "Select a session / class",
                    options=sessions,
                )

                ses = df[df["session"] == selected_session].copy()
                if ses.empty:
                    st.warning("No records for this session.")
                else:
                    total_students = ses[ses["role"] == "student"]["name"].nunique()
                    total_teachers = ses[ses["role"] == "teacher"]["name"].nunique()
                    avg_points = ses["points"].mean()
                    date_range = f"{ses['date'].min()} to {ses['date'].max()}"


                    c1, c2, c3 = st.columns(3)
                    c1.metric("Students in Session", total_students)
                    c2.metric("Teachers in Session", total_teachers)
                    c3.metric("Avg Engagement", f"{avg_points:.2f}")

                    st.caption(f"Date range: {date_range}")

                    st.markdown("---")

                    st.markdown("#### Engagement per Student in this Session (average points)")
                    ses_students = ses[ses["role"] == "student"]
                    if not ses_students.empty:
                        per_student_avg = (
                            ses_students.groupby("name")["points"]
                            .mean()
                            .sort_values(ascending=False)
                        )
                        st.bar_chart(per_student_avg)
                    else:
                        st.write("No student records for this session.")

                    st.markdown("#### 🎯 Attendance Score (55 min / 5 min slots, 60% rule)")

                    if not ses_students.empty:
                        per_student_total = (
                            ses_students.groupby("name")["points"]
                            .sum()
                            .sort_values(ascending=False)
                        )

                        session_attendance_rows = []
                        max_points = MAX_POINTS_PER_SESSION  # 22 for 55 min

                        for student_name, total_pts_raw in per_student_total.items():
                            # Cap total to a reasonable range for stability
                            total_pts = max(-max_points, min(total_pts_raw, max_points))
                            pct = (total_pts / max_points) if max_points > 0 else 0.0

                            if total_pts <= 0 or pct <= 0:
                                attendance_value = 0.0
                                status = "Absent"
                            elif pct < ATTENDANCE_THRESHOLD:
                                attendance_value = PARTIAL_ATTENDANCE_VALUE  # 0.6
                                status = "Partially Present"
                            else:
                                attendance_value = FULL_ATTENDANCE_VALUE      # 1.0
                                status = "Present"

                            session_attendance_rows.append(
                                {
                                    "Student": student_name,
                                    "Total Points": round(total_pts, 2),
                                    "Max Points (55 min)": max_points,
                                    "Engagement %": round(pct * 100, 1),
                                    "Status": status,
                                    "Attendance Weight": attendance_value,
                                }
                            )

                        df_session_att = pd.DataFrame(session_attendance_rows)
                        st.dataframe(df_session_att, use_container_width=True)
                        st.caption(
                            "Legend: **Present (1.0)** = ≥ 60% engagement, "
                            "**Partially Present (0.6)** = low but non-zero engagement, "
                            "**Absent (0.0)** = no or negative engagement."
                                  )


                        csv_att = df_session_att.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            f"📥 Download session '{selected_session}' attendance scores",
                            csv_att,
                            f"session_{selected_session.replace(' ', '_')}_attendance_scores.csv",
                            "text/csv",
                        )
                    else:
                        st.write("No student data for scoring in this session.")

                    st.markdown("#### All Session Records")
                    show_cols = ["timestamp", "name", "role", "emotion", "points"]
                    st.dataframe(
                        ses.sort_values("timestamp", ascending=False)[show_cols],
                        use_container_width=True,
                    )

                    csv_ses = ses.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"📥 Download session '{selected_session}' report",
                        csv_ses,
                        f"session_{selected_session.replace(' ', '_')}.csv",
                        "text/csv",
                    )