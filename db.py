# db.py
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey, Float, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_URL = "sqlite:///attendance.db"

engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True)  # roll no / teacher id
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)  # "student" or "teacher"


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)  # e.g. "AI-ML 7th Sem"
    scheduled_start = Column(DateTime, nullable=True)
    actual_start = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    teacher = relationship("User")


class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    emotion = Column(String, nullable=True)
    engagement_points = Column(Float, default=0.0)
    is_teacher = Column(Boolean, default=False)

    user = relationship("User")
    session = relationship("Session")


class TeacherMetric(Base):
    __tablename__ = "teacher_metrics"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    teacher_id = Column(Integer, ForeignKey("users.id"))
    metric_type = Column(String)  # "on_time", "late", "idle"
    points = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)

    teacher = relationship("User")
    session = relationship("Session")


def init_db():
    Base.metadata.create_all(bind=engine)


# -------- Helper functions for app --------
def get_or_create_user(external_id: str, name: str, role: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(external_id=external_id).first()
        if user:
            return user
        user = User(external_id=external_id, name=name, role=role)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def create_session(name: str, scheduled_start=None, teacher_id=None):
    db = SessionLocal()
    try:
        sess = Session(
            name=name,
            scheduled_start=scheduled_start,
            teacher_id=teacher_id
        )
        db.add(sess)
        db.commit()
        db.refresh(sess)
        return sess
    finally:
        db.close()


def set_session_actual_start(session_id: int):
    db = SessionLocal()
    try:
        sess = db.query(Session).get(session_id)
        if sess and sess.actual_start is None:
            sess.actual_start = datetime.utcnow()
            db.commit()
    finally:
        db.close()


def log_attendance(session_id: int, user_id: int, emotion: str,
                   engagement_points: float, is_teacher: bool = False):
    db = SessionLocal()
    try:
        record = Attendance(
            session_id=session_id,
            user_id=user_id,
            emotion=emotion,
            engagement_points=engagement_points,
            is_teacher=is_teacher,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


def log_teacher_metric(session_id: int, teacher_id: int,
                       metric_type: str, points: float):
    db = SessionLocal()
    try:
        rec = TeacherMetric(
            session_id=session_id,
            teacher_id=teacher_id,
            metric_type=metric_type,
            points=points,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec
    finally:
        db.close()


def get_attendance_df():
    import pandas as pd
    db = SessionLocal()
    try:
        q = (
            db.query(
                Attendance.id,
                Attendance.timestamp,
                Attendance.emotion,
                Attendance.engagement_points,
                Attendance.is_teacher,
                User.name,
                User.external_id,
                User.role,
                Session.name.label("session_name"),
            )
            .join(User, Attendance.user_id == User.id)
            .join(Session, Attendance.session_id == Session.id)
        )
        rows = q.all()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(
            [
                {
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "emotion": r.emotion,
                    "points": r.engagement_points,
                    "is_teacher": r.is_teacher,
                    "name": r.name,
                    "external_id": r.external_id,
                    "role": r.role,
                    "session": r.session_name,
                }
                for r in rows
            ]
        )
        return df
    finally:
        db.close()
