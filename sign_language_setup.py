#!/usr/bin/env python3
"""
Ars Signorum - Sign Language Interpreter: one-file project generator.

    python sign_language_setup.py

Creates the SignLanguageAI/ project next to this file, makes a private virtual
environment, installs the packages, downloads the pretrained models and launches
the app.  Options:
    --dir NAME     project folder (default SignLanguageAI)
    --no-venv      install into the current Python instead of .venv
    --no-install   skip pip
    --no-run       generate/install only
"""
import argparse
import base64
import subprocess
import sys
import zlib
from pathlib import Path

APP_PY = r'''#!/usr/bin/env python3
"""Ars Signorum - Sign Language Interpreter.

Camera -> hand tracking (MediaPipe Tasks) -> pretrained recognisers -> words on screen.

Recognition modes
  Gestures       pretrained MediaPipe Gesture Recognizer (8 canned gestures -> your words)
  Alphabet       pretrained ASL A-Z keypoint classifier (TFLite)
  Custom Hold    your own held signs (trained inside the app)
  Custom Motion  your own moving signs (trained inside the app)

Run:  python app.py
"""
import copy
import importlib
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

try:
    import cv2
    import numpy as np
    import customtkinter as ctk
    from PIL import Image
    from tkinter import filedialog
except ImportError as exc:  # pragma: no cover
    name = getattr(exc, "name", None) or str(exc)
    hint = ""
    if "tkinter" in str(exc).lower() or name == "tkinter":
        hint = "\nOn Linux install Tk first:  sudo apt install python3-tk"
    sys.exit(f"Missing package: {name}\nRun setup again (python sign_language_setup.py) or:  "
             f"pip install -r requirements.txt{hint}")

APP_NAME = "Ars Signorum"
APP_SUB = "SIGN LANGUAGE INTERPRETER"
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
SETTINGS_PATH = BASE / "settings.json"
STATIC_FILE = DATA_DIR / "static_samples.npz"
DYNAMIC_FILE = DATA_DIR / "dynamic_samples.npz"
HISTORY_FILE = DATA_DIR / "history.log"

MODELS = {
    "gesture": {
        "file": "gesture_recognizer.task",
        "title": "Gesture Recognizer",
        "purpose": "Pretrained hand gestures (8 built-in) and hand tracking. Recommended.",
        "url": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
               "gesture_recognizer/float16/1/gesture_recognizer.task",
        "min_size": 1_000_000,
    },
    "asl": {
        "file": "keypoint_classifier.tflite",
        "title": "ASL Alphabet Classifier",
        "purpose": "Pretrained ASL letters A-Z from hand landmarks (MIT, Muhib-Mehdi).",
        "url": "https://raw.githubusercontent.com/Muhib-Mehdi/ASL-Recognition-System/main/"
               "model/keypoint_classifier/keypoint_classifier.tflite",
        "min_size": 10_000,
    },
    "hand": {
        "file": "hand_landmarker.task",
        "title": "Hand Landmarker (fallback)",
        "purpose": "Hand tracking only. Used if the Gesture Recognizer file is missing.",
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
               "hand_landmarker/float16/1/hand_landmarker.task",
        "min_size": 1_000_000,
    },
}
ASL_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MODES = ["Gestures", "Alphabet", "Custom Hold", "Custom Motion"]
CASES = ["UPPERCASE", "lowercase", "Capitalized", "As is"]
RESOLUTIONS = ["640x480", "1280x720", "1920x1080"]

DEFAULTS = {
    "appearance": "Dark",
    "mode": "Gestures",
    "camera_index": 0,
    "resolution": "1280x720",
    "mirror": True,
    "show_landmarks": True,
    "show_overlay_text": True,
    "min_detection": 0.6,
    "min_presence": 0.5,
    "min_tracking": 0.5,
    "gesture_min_score": 0.6,
    "alphabet_min_conf": 0.8,
    "hold_frames": 10,
    "cooldown": 1.0,
    "repeat_delay": 2.0,
    "auto_space_seconds": 2.0,
    "knn_k": 5,
    "static_max_distance": 1.6,
    "dynamic_max_distance": 2.2,
    "min_motion": 0.12,
    "dynamic_window": 30,
    "dynamic_resample": 16,
    "text_case": "UPPERCASE",
    "speak_on_word": False,
    "auto_download": True,
    "geometry": "1280x820",
    "gesture_words": {
        "Open_Palm": "HELLO",
        "Thumb_Up": "YES",
        "Thumb_Down": "NO",
        "Closed_Fist": "STOP",
        "Pointing_Up": "ATTENTION",
        "Victory": "PEACE",
        "ILoveYou": "I LOVE YOU",
    },
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
    (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# ------------------------------------------------------------------ palette
# (light, dark) tuples - customtkinter switches them automatically.
BG = ("#F4EDDF", "#10101A")
CARD = ("#FFFCF5", "#1A1A29")
CARD2 = ("#EBE1CB", "#232336")
FIELD = ("#FFFFFF", "#12121C")
GOLD = ("#8C6B2F", "#C9A86A")
GOLD_H = ("#745724", "#DDBE82")
BURG = ("#7A1F2B", "#8E2233")
BURG_H = ("#5F1822", "#A62A3F")
TEXT = ("#231F1A", "#F2EDE0")
SUB = ("#6D6558", "#A39B8A")
ON_GOLD = ("#FFFCF5", "#14141F")
ON_BURG = ("#FFFCF5", "#F8F3E8")
SEG_OFF = ("#7B705C", "#2C2C42")
SEG_OFF_H = ("#6A6050", "#3A3A55")
OK = ("#2E7D4F", "#5FBF85")
WARN = ("#A85F00", "#E0A040")
BAD = ("#B3261E", "#E5726B")


# ----------------------------------------------------------------- settings
def load_settings():
    cfg = copy.deepcopy(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            user = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            for key, value in user.items():
                if key == "gesture_words" and isinstance(value, dict):
                    cfg[key].update({k: str(v) for k, v in value.items() if k in cfg[key]})
                elif key in cfg and not isinstance(cfg[key], dict):
                    cfg[key] = value
        except Exception as exc:
            print(f"[warn] settings.json unreadable ({exc}); using defaults")
    if cfg["mode"] not in MODES:
        cfg["mode"] = "Gestures"
    return cfg


def save_settings(cfg):
    try:
        SETTINGS_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[warn] could not save settings: {exc}")


# ------------------------------------------------------------------- models
def model_path(key):
    return MODEL_DIR / MODELS[key]["file"]


def model_ok(key):
    p = model_path(key)
    return p.exists() and p.stat().st_size >= MODELS[key]["min_size"]


def download_model(key, progress=None):
    info = MODELS[key]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = model_path(key).with_suffix(".part")
    req = urllib.request.Request(info["url"], headers={"User-Agent": "Mozilla/5.0 ArsSignorum"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total)
    if tmp.stat().st_size < info["min_size"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded file is too small (blocked or wrong link)")
    tmp.replace(model_path(key))


def open_folder(path):
    path = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:
        print("could not open folder:", exc)


# ----------------------------------------------------------------- features
def hand_slots(hands):
    """[left_slot, right_slot]: each an (21,3) array or None."""
    slots = [None, None]
    if not hands:
        return slots
    if len(hands) == 1:
        h = hands[0]
        slots[1 if h["side"] == "Right" else 0] = h["pts"]
    else:
        ordered = sorted(hands[:2], key=lambda h: float(h["pts"][0, 0]))
        slots[0], slots[1] = ordered[0]["pts"], ordered[1]["pts"]
    return slots


def normalize_hand(pts):
    rel = pts - pts[0]
    scale = float(np.linalg.norm(rel[9]))
    return (rel / (scale if scale > 1e-6 else 1.0)).flatten()


def shape_features(slots):
    parts = [normalize_hand(p) if p is not None else np.zeros(63, np.float32) for p in slots]
    return np.concatenate(parts).astype(np.float32)


def position_features(slots):
    out = []
    for p in slots:
        out.extend([float(p[0, 0]), float(p[0, 1])] if p is not None else [0.0, 0.0])
    return np.array(out, dtype=np.float32)


def dynamic_features(frames, n):
    shapes = np.array([f[0] for f in frames], dtype=np.float32)
    poses = np.array([f[1] for f in frames], dtype=np.float32)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    shapes, poses = shapes[idx], poses[idx]
    poses = poses - poses[0]
    return (np.concatenate([shapes.flatten(), poses.flatten() * 4.0]) / np.sqrt(n)).astype(np.float32)


def motion_amount(frames):
    poses = np.array([f[1] for f in frames], dtype=np.float32)
    return float((poses.max(axis=0) - poses.min(axis=0)).max())


def asl_features(pts, w, h):
    """Exactly the preprocessing the pretrained A-Z model was trained with."""
    xy = np.stack([np.minimum((pts[:, 0] * w).astype(int), w - 1),
                   np.minimum((pts[:, 1] * h).astype(int), h - 1)], axis=1).astype(np.float32)
    flat = (xy - xy[0]).flatten()
    peak = float(np.max(np.abs(flat)))
    if peak < 1e-6:
        return None
    return (flat / peak).astype(np.float32)


# --------------------------------------------------------------- classifiers
class KeypointClassifier:
    """Pretrained ASL A-Z classifier (TFLite). Works with any of three runtimes."""

    def __init__(self, path):
        interp = None
        for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
            try:
                interp = getattr(importlib.import_module(mod), "Interpreter")
                break
            except Exception:
                continue
        if interp is None:
            raise RuntimeError("no TFLite runtime installed (pip install ai-edge-litert)")
        content = Path(path).read_bytes()
        try:
            self.it = interp(model_content=content, num_threads=2)
        except TypeError:
            self.it = interp(model_content=content)
        self.it.allocate_tensors()
        self.i_idx = self.it.get_input_details()[0]["index"]
        self.o_idx = self.it.get_output_details()[0]["index"]

    def predict(self, vec):
        self.it.set_tensor(self.i_idx, vec[None, :].astype(np.float32))
        self.it.invoke()
        out = np.squeeze(self.it.get_tensor(self.o_idx))
        i = int(np.argmax(out))
        return i, float(out[i])


class SampleStore:
    def __init__(self, path, k):
        self.path, self.k = path, k
        self.X, self.y = None, np.array([], dtype="<U64")
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as z:
                    self.X, self.y = z["X"].astype(np.float32), z["y"].astype("<U64")
            except Exception as exc:
                print(f"[warn] could not load {path.name}: {exc}")

    def save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if self.X is not None and len(self.y):
            np.savez(self.path, X=self.X, y=self.y)
        elif self.path.exists():
            self.path.unlink()

    def add(self, x, label):
        x = np.asarray(x, dtype=np.float32)[None, :]
        self.X = x if self.X is None or len(self.y) == 0 else np.vstack([self.X, x])
        self.y = np.append(self.y, np.array([label], dtype="<U64"))

    def count(self, label=None):
        return int(len(self.y) if label is None else np.sum(self.y == label))

    def labels(self):
        return sorted(set(self.y.tolist()))

    def remove(self, label):
        keep = self.y != label
        self.X = self.X[keep] if self.X is not None else None
        self.y = self.y[keep]
        self.save()

    def clear(self):
        self.X, self.y = None, np.array([], dtype="<U64")
        self.save()

    def predict(self, x, max_dist):
        if self.X is None or len(self.y) == 0:
            return None, 0.0
        d = np.linalg.norm(self.X - x[None, :], axis=1)
        k = min(self.k, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        idx = idx[d[idx] <= max_dist]
        if len(idx) == 0:
            return None, 0.0
        votes = {}
        for i in idx:
            votes[str(self.y[i])] = votes.get(str(self.y[i]), 0.0) + 1.0 / (float(d[i]) + 1e-3)
        label = max(votes, key=votes.get)
        return label, votes[label] / sum(votes.values())


class Stabilizer:
    """Turns noisy per-frame labels into one committed token per sign."""

    def __init__(self, hold, cooldown, repeat=0.0):
        self.hold, self.cooldown, self.repeat = max(1, int(hold)), cooldown, repeat
        self.current, self.count, self.gap = None, 0, 0
        self.last, self.last_t = None, 0.0

    def update(self, label, now):
        if label is None:
            self.gap += 1
            if self.gap >= 8:
                self.current, self.count, self.last = None, 0, None
            return None
        self.gap = 0
        if label == self.current:
            self.count += 1
        else:
            self.current, self.count = label, 1
        if self.count < self.hold or now - self.last_t < self.cooldown:
            return None
        if label != self.last or (self.repeat > 0 and now - self.last_t >= self.repeat):
            self.last, self.last_t = label, now
            return label
        return None

    def progress(self):
        return min(1.0, self.count / float(self.hold)) if self.current else 0.0


class Tracker:
    """MediaPipe Tasks wrapper: GestureRecognizer if available, else HandLandmarker."""

    def __init__(self, kind, path, cfg):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp, self.kind, self.last_ts = mp, kind, 0
        base = mp_python.BaseOptions(model_asset_buffer=Path(path).read_bytes())
        common = dict(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=float(cfg["min_detection"]),
            min_hand_presence_confidence=float(cfg["min_presence"]),
            min_tracking_confidence=float(cfg["min_tracking"]),
        )
        if kind == "gesture":
            self.impl = vision.GestureRecognizer.create_from_options(
                vision.GestureRecognizerOptions(**common))
        else:
            self.impl = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(**common))

    def process(self, rgb):
        ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
        self.last_ts = ts
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.kind == "gesture":
            res = self.impl.recognize_for_video(img, ts)
        else:
            res = self.impl.detect_for_video(img, ts)
        hands = []
        for i, lm in enumerate(res.hand_landmarks or []):
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            side = None
            try:
                side = res.handedness[i][0].category_name
            except Exception:
                pass
            name, score = None, 0.0
            if self.kind == "gesture":
                try:
                    cat = res.gestures[i][0]
                    name, score = cat.category_name, float(cat.score)
                except Exception:
                    pass
            hands.append({"pts": pts, "side": side, "gesture": name, "gscore": score})
        return hands

    def close(self):
        try:
            self.impl.close()
        except Exception:
            pass


# ------------------------------------------------------------------- camera
class FakeCam:
    """Synthetic camera (SIGNLANG_FAKECAM=1) for testing the UI without hardware."""

    def __init__(self):
        self.t = 0

    def isOpened(self):
        return True

    def read(self):
        self.t += 1
        img = np.zeros((480, 640, 3), np.uint8)
        img[:] = (60 + int(40 * np.sin(self.t / 20)), 40, 50)
        cv2.putText(img, "SYNTHETIC CAMERA", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)
        time.sleep(0.03)
        return True, img

    def set(self, *a):
        return True

    def release(self):
        pass


def _backends():
    return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]


def open_camera(index, resolution):
    if os.environ.get("SIGNLANG_FAKECAM"):
        return FakeCam()
    try:
        w, h = (int(v) for v in resolution.split("x"))
    except Exception:
        w, h = 1280, 720
    for backend in _backends():
        cap = cv2.VideoCapture(int(index), backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def probe_cameras(active, limit=6):
    if os.environ.get("SIGNLANG_FAKECAM"):
        return [0, 1]
    found = []
    for i in range(limit):
        if i == active:
            found.append(i)
            continue
        for backend in _backends()[:1]:
            cap = cv2.VideoCapture(i, backend)
            try:
                if cap.isOpened() and cap.read()[0]:
                    found.append(i)
            finally:
                cap.release()
    return found


# ------------------------------------------------------------------- engine
class Engine(threading.Thread):
    """Capture + recognition worker. The UI only reads snapshots and queued events."""

    def __init__(self, cfg, out_q):
        super().__init__(daemon=True, name="engine")
        self.cfg, self.q = cfg, out_q
        self.lock = threading.RLock()
        self.stop_flag = False
        self.cam, self.cam_dirty = None, True
        self.tracker, self.tracker_dirty = None, True
        self.tracker_error = self.asl_error = ""
        self.asl = None
        self.static = SampleStore(STATIC_FILE, int(cfg["knn_k"]))
        self.dynamic = SampleStore(DYNAMIC_FILE, int(cfg["knn_k"]))
        self.mode = cfg["mode"]
        self.frame, self.frame_id, self.info = None, 0, {"status": "Starting..."}
        self.stab = None
        self.window = deque(maxlen=int(cfg["dynamic_window"]))
        self.missing = 0
        self.dyn_candidate, self.dyn_block = None, 0.0
        self.rec = {"active": False}
        self.last_hand_t = time.time()
        self.space_sent = True
        self.fps, self.t_prev = 0.0, time.time()
        self.bad_reads, self.retry_at = 0, 0.0
        self.frame_no = 0
        self._new_stab()

    # ---- control API (called from UI thread)
    def stop(self):
        self.stop_flag = True

    def request_camera(self):
        self.cam_dirty = True

    def reload_models(self):
        self.tracker_dirty = True

    def set_mode(self, mode):
        with self.lock:
            self.mode = mode
            self.rec = {"active": False}
            self.window.clear()
            self.dyn_candidate = None
            self._new_stab()

    def apply_config(self):
        with self.lock:
            self.window = deque(self.window, maxlen=int(self.cfg["dynamic_window"]))
            self.static.k = self.dynamic.k = int(self.cfg["knn_k"])
            self._new_stab()

    def start_hold(self, label):
        with self.lock:
            self.rec = {"active": True, "kind": "hold", "label": label, "n": 0}

    def stop_hold(self):
        with self.lock:
            if self.rec.get("active") and self.rec.get("kind") == "hold":
                label = self.rec["label"]
                self.rec = {"active": False}
                self.static.save()
                self.q.put(("samples",))
                self.q.put(("flash", f"Saved '{label}' ({self.static.count(label)} samples)"))

    def start_motion(self, label):
        with self.lock:
            self.rec = {"active": True, "kind": "motion", "label": label, "frames": [],
                        "start": time.time() + 1.5}

    def cancel_rec(self):
        with self.lock:
            self.rec = {"active": False}

    def delete_label(self, kind, label):
        with self.lock:
            (self.static if kind == "Hold" else self.dynamic).remove(label)
        self.q.put(("samples",))

    def clear_samples(self):
        with self.lock:
            self.static.clear()
            self.dynamic.clear()
        self.q.put(("samples",))

    def sample_summary(self):
        with self.lock:
            rows = [("Hold", lb, self.static.count(lb)) for lb in self.static.labels()]
            rows += [("Motion", lb, self.dynamic.count(lb)) for lb in self.dynamic.labels()]
        return rows

    def snapshot(self):
        with self.lock:
            return self.frame, self.frame_id, dict(self.info)

    def shutdown_save(self):
        with self.lock:
            self.static.save()
            self.dynamic.save()

    # ---- internals
    def _new_stab(self):
        c = self.cfg
        repeat = float(c["repeat_delay"]) if self.mode == "Alphabet" else 0.0
        self.stab = Stabilizer(int(c["hold_frames"]), float(c["cooldown"]), repeat)

    def _load_models(self):
        if self.tracker is not None:
            self.tracker.close()
        self.tracker, self.tracker_error, self.asl, self.asl_error = None, "", None, ""
        kind = "gesture" if model_ok("gesture") else ("landmark" if model_ok("hand") else None)
        if kind:
            try:
                self.tracker = Tracker(kind, model_path("gesture" if kind == "gesture" else "hand"), self.cfg)
            except Exception as exc:
                self.tracker_error = f"{type(exc).__name__}: {exc}"
        if model_ok("asl"):
            try:
                self.asl = KeypointClassifier(model_path("asl"))
            except Exception as exc:
                self.asl_error = str(exc)
        self.tracker_dirty = False

    def _open_cam(self):
        if self.cam is not None:
            self.cam.release()
            self.cam = None
        self.cam_dirty = False
        self.cam = open_camera(self.cfg["camera_index"], self.cfg["resolution"])
        self.bad_reads = 0
        if self.cam is None:
            self.retry_at = time.time() + 3
            self.q.put(("flash", f"Camera {self.cfg['camera_index']} could not be opened"))

    def _emit(self, token, kind, conf):
        self.q.put(("token", token, kind, conf, self.mode))

    def run(self):
        while not self.stop_flag:
            try:
                self._step()
            except Exception as exc:  # keep the worker alive
                with self.lock:
                    self.info = {"status": f"Engine error: {type(exc).__name__}: {exc}"}
                time.sleep(0.5)
        if self.cam is not None:
            self.cam.release()
        if self.tracker is not None:
            self.tracker.close()

    def _step(self):
        if self.tracker_dirty:
            self._load_models()
        if self.cam_dirty or (self.cam is None and time.time() >= self.retry_at):
            self._open_cam()
        if self.cam is None:
            with self.lock:
                self.frame = None
                self.info = {"status": "No camera - choose one in Settings", "hint": ""}
            time.sleep(0.2)
            return
        ok, frame = self.cam.read()
        if not ok:
            self.bad_reads += 1
            if self.bad_reads > 40:
                self.cam.release()
                self.cam, self.retry_at = None, time.time() + 2
                self.q.put(("flash", "Camera stopped delivering frames"))
            time.sleep(0.01)
            return
        self.bad_reads = 0
        cfg, now = self.cfg, time.time()
        self.frame_no += 1
        flipped = cv2.flip(frame, 1)  # the pretrained models expect a mirrored view
        h, w = flipped.shape[:2]
        hands = []
        if self.tracker is not None:
            hands = self.tracker.process(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))

        with self.lock:
            mode = self.mode
            pred, conf, source, hint, progress = None, 0.0, "", "", 0.0
            slots = hand_slots(hands)
            present = bool(hands)
            if present:
                self.last_hand_t, self.space_sent = now, False

            if self.tracker is None:
                hint = "No hand-tracking model - open the Models tab"
                if self.tracker_error:
                    hint += f" ({self.tracker_error})"
            elif mode == "Gestures" and self.tracker.kind != "gesture":
                hint = "Gesture Recognizer file missing - open the Models tab"
            elif mode == "Alphabet" and self.asl is None:
                hint = "ASL classifier unavailable: " + (self.asl_error or "model file missing (Models tab)")
            elif mode == "Custom Hold" and self.static.count() == 0 and not self.rec["active"]:
                hint = "No trained signs yet - open the Train tab"
            elif mode == "Custom Motion" and self.dynamic.count() == 0 and not self.rec["active"]:
                hint = "No trained motion signs yet - open the Train tab"

            if mode == "Gestures":
                best = None
                for hd in hands:
                    g = hd["gesture"]
                    if g and g != "None" and (best is None or hd["gscore"] > best["gscore"]):
                        best = hd
                if best and best["gscore"] >= float(cfg["gesture_min_score"]):
                    pred = cfg["gesture_words"].get(best["gesture"], best["gesture"].replace("_", " ").upper())
                    conf, source = best["gscore"], "gesture model"
            elif mode == "Alphabet":
                if hands and self.asl is not None:
                    vec = asl_features(hands[0]["pts"], w, h)
                    if vec is not None:
                        idx, c = self.asl.predict(vec)
                        if c >= float(cfg["alphabet_min_conf"]) and idx < len(ASL_LABELS):
                            pred, conf, source = ASL_LABELS[idx], c, "ASL model"
            elif mode == "Custom Hold":
                shape = shape_features(slots)
                if present:
                    pred, conf = self.static.predict(shape, float(cfg["static_max_distance"]))
                    source = "trained"
                if self.rec["active"] and self.rec.get("kind") == "hold" and present and self.frame_no % 2 == 0:
                    lb = self.rec["label"]
                    self.static.add(shape, lb)
                    self.rec["n"] += 1
                    if self.rec["n"] % 25 == 0:
                        self.static.save()

            # --- commit logic
            rec_text = ""
            if self.rec["active"]:
                if self.rec.get("kind") == "hold":
                    rec_text = f"Recording '{self.rec['label']}' - {self.rec['n']} samples"
                elif self.rec.get("kind") == "motion":
                    if now < self.rec["start"]:
                        rec_text = f"Get ready... {self.rec['start'] - now:0.1f}s"
                    else:
                        if present:
                            self.rec["frames"].append((shape_features(slots), position_features(slots)))
                        n_need = int(cfg["dynamic_window"])
                        rec_text = f"Recording '{self.rec['label']}' {len(self.rec['frames'])}/{n_need}"
                        if len(self.rec["frames"]) >= n_need:
                            self.dynamic.add(dynamic_features(self.rec["frames"], int(cfg["dynamic_resample"])),
                                             self.rec["label"])
                            self.dynamic.save()
                            lb = self.rec["label"]
                            self.rec = {"active": False}
                            self.q.put(("samples",))
                            self.q.put(("flash", f"Saved motion '{lb}' ({self.dynamic.count(lb)} samples)"))
            elif mode == "Custom Motion":
                if present:
                    self.window.append((shape_features(slots), position_features(slots)))
                    self.missing = 0
                else:
                    self.missing += 1
                    if self.missing > 6:
                        self.window.clear()
                if (len(self.window) == self.window.maxlen and self.frame_no % 4 == 0 and now >= self.dyn_block):
                    frames = list(self.window)
                    if motion_amount(frames) >= float(cfg["min_motion"]):
                        p, c = self.dynamic.predict(dynamic_features(frames, int(cfg["dynamic_resample"])),
                                                    float(cfg["dynamic_max_distance"]))
                        if p is not None and p == self.dyn_candidate:
                            self._emit(p, "word", c)
                            self.window.clear()
                            self.dyn_candidate, self.dyn_block = None, now + 1.5
                        else:
                            self.dyn_candidate = p
                        pred, conf, source = p, c, "trained"
                    else:
                        self.dyn_candidate = None
            else:
                token = self.stab.update(pred if not self.rec["active"] else None, now)
                progress = self.stab.progress()
                if token:
                    self._emit(token, "letter" if mode == "Alphabet" else "word", conf)

            # auto-space between letters when the hand leaves
            sp = float(cfg["auto_space_seconds"])
            if (mode == "Alphabet" and sp > 0 and not present and not self.space_sent
                    and now - self.last_hand_t >= sp):
                self.space_sent = True
                self.q.put(("space",))

            # --- draw overlay
            disp = flipped
            show_lm = bool(cfg["show_landmarks"])
            if show_lm:
                for hd in hands:
                    px = [(int(p[0] * w), int(p[1] * h)) for p in hd["pts"]]
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(disp, px[a], px[b], (106, 168, 201), 2, cv2.LINE_AA)
                    for p in px:
                        cv2.circle(disp, p, 4, (43, 34, 142), -1, cv2.LINE_AA)
                        cv2.circle(disp, p, 4, (200, 220, 240), 1, cv2.LINE_AA)
            mirror = bool(cfg["mirror"])
            if not mirror:
                disp = cv2.flip(disp, 1)
            if pred and hands and cfg["show_overlay_text"]:
                xs = [p[0] * w for p in hands[0]["pts"]]
                ys = [p[1] * h for p in hands[0]["pts"]]
                x1, x2 = (min(xs), max(xs)) if mirror else (w - max(xs), w - min(xs))
                y1, y2 = min(ys), max(ys)
                cv2.rectangle(disp, (int(x1) - 12, int(y1) - 12), (int(x2) + 12, int(y2) + 12), (106, 168, 201), 2)
                tag = str(pred)
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
                tx, ty = int(x1) - 12, max(int(y1) - 20, th + 12)
                cv2.rectangle(disp, (tx, ty - th - 10), (tx + tw + 14, ty + 6), (43, 34, 142), -1)
                cv2.putText(disp, tag, (tx + 7, ty - 2), cv2.FONT_HERSHEY_DUPLEX, 0.9, (235, 240, 248), 2, cv2.LINE_AA)
            if w > 960:
                disp = cv2.resize(disp, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA)

            self.fps = 0.9 * self.fps + 0.1 / max(now - self.t_prev, 1e-6)
            self.t_prev = now
            self.frame = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self.frame_id += 1
            self.info = {
                "status": "Live", "pred": pred, "conf": conf, "source": source, "hint": hint,
                "fps": self.fps, "hands": len(hands), "progress": progress, "rec_text": rec_text,
                "rec_active": self.rec["active"], "rec_kind": self.rec.get("kind"),
                "tracker": getattr(self.tracker, "kind", None), "mode": mode,
            }


# --------------------------------------------------------------------- TTS
def speak_async(text):
    def work():
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
        except Exception as exc:
            print("[tts]", exc)
    if text.strip():
        threading.Thread(target=work, daemon=True).start()


def apply_case(token, kind, current, style):
    if style == "UPPERCASE":
        return token.upper()
    if style == "lowercase":
        return token.lower()
    if style == "Capitalized":
        starts = (not current.strip()) or current.rstrip().endswith((".", "!", "?")) and current.endswith(" ")
        if kind == "letter":
            return token.upper() if starts else token.lower()
        return token.capitalize() if starts else token.lower()
    return token


# ------------------------------------------------------------------ UI parts
def gold_button(parent, text, command, width=100, **kw):
    return ctk.CTkButton(parent, text=text, command=command, width=width, fg_color=GOLD, hover_color=GOLD_H,
                         text_color=ON_GOLD, corner_radius=6, **kw)


def burg_button(parent, text, command, width=100, **kw):
    return ctk.CTkButton(parent, text=text, command=command, width=width, fg_color=BURG, hover_color=BURG_H,
                         text_color=ON_BURG, corner_radius=6, **kw)


def ghost_button(parent, text, command, width=100, **kw):
    return ctk.CTkButton(parent, text=text, command=command, width=width, fg_color="transparent",
                         hover_color=CARD2, text_color=GOLD, border_width=1, border_color=GOLD,
                         corner_radius=6, **kw)


def rule(parent):
    return ctk.CTkFrame(parent, height=1, fg_color=GOLD, corner_radius=0)


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title(f"{APP_NAME} - Settings")
        self.geometry("560x720")
        self.minsize(520, 560)
        self.configure(fg_color=BG)
        self.transient(app)
        self.after(150, self.lift)
        F = app.fonts
        head = ctk.CTkFrame(self, fg_color=CARD, border_width=1, border_color=GOLD, corner_radius=0)
        head.pack(fill="x")
        ctk.CTkLabel(head, text="Settings", font=F["h1"], text_color=GOLD).pack(side="left", padx=18, pady=12)
        ghost_button(head, "Reset to defaults", self.reset, width=130).pack(side="right", padx=14)
        self.body = ctk.CTkScrollableFrame(self, fg_color=BG)
        self.body.pack(fill="both", expand=True, padx=10, pady=10)
        self._build()

    # ---- helpers
    def section(self, title):
        ctk.CTkLabel(self.body, text=title.upper(), font=self.app.fonts["small_b"], text_color=GOLD
                     ).pack(anchor="w", padx=8, pady=(16, 2))
        rule(self.body).pack(fill="x", padx=8, pady=(0, 6))

    def row(self, label):
        r = ctk.CTkFrame(self.body, fg_color="transparent")
        r.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(r, text=label, text_color=TEXT, anchor="w").pack(side="left")
        return r

    def slider(self, label, key, lo, hi, steps, fmt="{:.2f}", cast=float, on_change=None):
        r = self.row(label)
        val = ctk.CTkLabel(r, text=fmt.format(self.app.cfg[key]), text_color=SUB, width=48)
        val.pack(side="right")

        def cb(v):
            v = cast(v)
            self.app.cfg[key] = v
            val.configure(text=fmt.format(v))
            self.app.cfg_changed(on_change)
        s = ctk.CTkSlider(r, from_=lo, to=hi, number_of_steps=steps, width=210, command=cb,
                          progress_color=GOLD, button_color=GOLD, button_hover_color=GOLD_H)
        s.set(self.app.cfg[key])
        s.pack(side="right", padx=8)

    def switch(self, label, key, on_change=None):
        r = self.row(label)
        var = ctk.BooleanVar(value=bool(self.app.cfg[key]))

        def cb():
            self.app.cfg[key] = bool(var.get())
            self.app.cfg_changed(on_change)
        ctk.CTkSwitch(r, text="", variable=var, command=cb, progress_color=GOLD, width=46).pack(side="right")

    def menu(self, label, key, values, on_change=None):
        r = self.row(label)

        def cb(v):
            self.app.cfg[key] = v
            self.app.cfg_changed(on_change)
        m = ctk.CTkOptionMenu(r, values=values, command=cb, width=170, fg_color=CARD2, button_color=GOLD,
                              button_hover_color=GOLD_H, text_color=TEXT, dropdown_fg_color=CARD,
                              dropdown_text_color=TEXT, dropdown_hover_color=CARD2)
        m.set(str(self.app.cfg[key]))
        m.pack(side="right")

    # ---- content
    def _build(self):
        for w in self.body.winfo_children():
            w.destroy()
        app = self.app
        self.section("Camera")
        r = self.row("Device")
        self.cam_menu = ctk.CTkOptionMenu(r, values=self._cam_values(), command=self._pick_cam, width=170,
                                          fg_color=CARD2, button_color=GOLD, button_hover_color=GOLD_H,
                                          text_color=TEXT, dropdown_fg_color=CARD, dropdown_text_color=TEXT,
                                          dropdown_hover_color=CARD2)
        self.cam_menu.set(f"Camera {app.cfg['camera_index']}")
        self.cam_menu.pack(side="right")
        ghost_button(r, "Scan", self._scan, width=60).pack(side="right", padx=8)
        self.menu("Resolution", "resolution", RESOLUTIONS, on_change="camera")
        self.switch("Mirror preview", "mirror")
        self.switch("Show hand skeleton", "show_landmarks")
        self.switch("Show sign label on video", "show_overlay_text")

        self.section("Recognition")
        self.slider("Hand detection confidence", "min_detection", 0.3, 0.95, 65, on_change="models")
        self.slider("Hand tracking confidence", "min_tracking", 0.3, 0.95, 65, on_change="models")
        self.slider("Gesture confidence needed", "gesture_min_score", 0.3, 0.95, 65)
        self.slider("Alphabet confidence needed", "alphabet_min_conf", 0.3, 0.99, 69)
        self.slider("Hold steady (frames)", "hold_frames", 3, 30, 27, fmt="{:.0f}", cast=int, on_change="stab")
        self.slider("Cooldown between words (s)", "cooldown", 0.2, 3.0, 28, fmt="{:.1f}", on_change="stab")
        self.slider("Repeat same letter after (s)", "repeat_delay", 0.8, 5.0, 42, fmt="{:.1f}", on_change="stab")
        self.slider("Auto-space when hand leaves (s, 0 = off)", "auto_space_seconds", 0.0, 5.0, 50, fmt="{:.1f}")

        self.section("Custom signs")
        self.slider("Hold-sign match strictness", "static_max_distance", 0.6, 3.0, 48)
        self.slider("Motion-sign match strictness", "dynamic_max_distance", 0.8, 4.0, 64)
        self.slider("Minimum hand movement", "min_motion", 0.03, 0.4, 37)

        self.section("Output")
        self.menu("Text style", "text_case", CASES)
        self.switch("Speak each new word aloud", "speak_on_word")

        self.section("Appearance")
        self.menu("Theme", "appearance", ["Dark", "Light", "System"], on_change="theme")

        self.section("Gesture vocabulary")
        ctk.CTkLabel(self.body, text="The pretrained recogniser knows 8 gestures. Choose the word each one types.",
                     text_color=SUB, wraplength=470, justify="left").pack(anchor="w", padx=8, pady=(0, 4))
        for gname in app.cfg["gesture_words"]:
            r = self.row(gname.replace("_", " "))
            e = ctk.CTkEntry(r, width=170, fg_color=FIELD, border_color=GOLD, text_color=TEXT)
            e.insert(0, app.cfg["gesture_words"][gname])
            e.pack(side="right")
            e.bind("<KeyRelease>", lambda _e, g=gname, ent=e: self._word(g, ent))

        self.section("Models")
        self.switch("Download missing models automatically", "auto_download")
        r = self.row("")
        ghost_button(r, "Open models folder", lambda: open_folder(MODEL_DIR), width=160).pack(side="right")

    def _word(self, gname, entry):
        self.app.cfg["gesture_words"][gname] = entry.get().strip() or gname
        self.app.cfg_changed()

    def _cam_values(self):
        vals = [f"Camera {i}" for i in self.app.cameras]
        cur = f"Camera {self.app.cfg['camera_index']}"
        if cur not in vals:
            vals.append(cur)
        return vals or ["Camera 0"]

    def _pick_cam(self, value):
        self.app.cfg["camera_index"] = int(value.split()[-1])
        self.app.cfg_changed("camera")

    def _scan(self):
        self.cam_menu.configure(values=["Scanning..."])
        self.app.scan_cameras(lambda: self.cam_menu.configure(values=self._cam_values()))

    def reset(self):
        keep = {k: self.app.cfg[k] for k in ("geometry",)}
        self.app.cfg.clear()
        self.app.cfg.update(copy.deepcopy(DEFAULTS))
        self.app.cfg.update(keep)
        self.app.cfg_changed("theme")
        self.app.cfg_changed("camera")
        self.app.cfg_changed("models")
        self._build()


class App(ctk.CTk):
    def __init__(self):
        self.cfg = load_settings()
        ctk.set_appearance_mode(self.cfg["appearance"])
        ctk.set_default_color_theme("dark-blue")
        super().__init__(fg_color=BG)
        self.title(f"{APP_NAME} - Sign Language Interpreter")
        try:
            self.geometry(self.cfg["geometry"])
        except Exception:
            self.geometry("1280x820")
        self.minsize(1080, 700)
        self.fonts = {
            "title": ctk.CTkFont(family="Georgia", size=26, weight="bold"),
            "h1": ctk.CTkFont(family="Georgia", size=20, weight="bold"),
            "h2": ctk.CTkFont(family="Georgia", size=15, weight="bold"),
            "big": ctk.CTkFont(family="Georgia", size=52, weight="bold"),
            "serif": ctk.CTkFont(family="Georgia", size=17),
            "small": ctk.CTkFont(size=12),
            "small_b": ctk.CTkFont(size=11, weight="bold"),
        }
        self.q = queue.Queue()
        self.ui_q = queue.Queue()
        self.cameras = [self.cfg["camera_index"]]
        self.settings_win = None
        self.frame_seen = -1
        self.flash_until, self.hist_count = 0.0, 0
        self.dl_state = {}
        self.engine = Engine(self.cfg, self.q)

        self._build_header()
        self._build_body()
        self._build_status()
        self.bind("<Control-comma>", lambda e: self.open_settings())
        self.bind("<Control-t>", lambda e: self.toggle_theme())
        self.bind("<Control-s>", lambda e: self.save_text())
        self.bind("<F11>", lambda e: self.attributes("-fullscreen", not self.attributes("-fullscreen")))
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.engine.start()
        self.scan_cameras()
        self.refresh_models()
        self.refresh_samples()
        if self.cfg["auto_download"]:
            self.download_missing()
        self.after(60, self._tick)

    # ------------------------------------------------------------- layout
    def _card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=CARD, border_width=1, border_color=GOLD, corner_radius=10, **kw)

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=CARD, border_width=1, border_color=GOLD, corner_radius=0)
        hdr.pack(fill="x")
        left = ctk.CTkFrame(hdr, fg_color="transparent")
        left.pack(side="left", padx=22, pady=10)
        ctk.CTkLabel(left, text=APP_NAME, font=self.fonts["title"], text_color=GOLD).pack(anchor="w")
        ctk.CTkLabel(left, text=APP_SUB, font=self.fonts["small_b"], text_color=SUB).pack(anchor="w")
        right = ctk.CTkFrame(hdr, fg_color="transparent")
        right.pack(side="right", padx=18)
        ghost_button(right, "Settings", self.open_settings, width=100).pack(side="right", padx=4)
        self.theme_btn = gold_button(right, "", self.toggle_theme, width=100)
        self.theme_btn.pack(side="right", padx=4)
        self._theme_label()

    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=(14, 8))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0, minsize=430)
        body.grid_rowconfigure(0, weight=1)

        # ---- left: camera
        vid = self._card(body)
        vid.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        top = ctk.CTkFrame(vid, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(12, 6))
        ctk.CTkLabel(top, text="Speculum", font=self.fonts["h1"], text_color=GOLD).pack(side="left")
        self.mode_btn = ctk.CTkSegmentedButton(
            top, values=MODES, command=self.set_mode, selected_color=BURG, selected_hover_color=BURG_H,
            unselected_color=SEG_OFF, unselected_hover_color=SEG_OFF_H, text_color=ON_BURG,
            fg_color=SEG_OFF, corner_radius=6)
        self.mode_btn.set(self.cfg["mode"])
        self.mode_btn.pack(side="right")
        rule(vid).pack(fill="x", padx=16)
        self.video_holder = ctk.CTkFrame(vid, fg_color=("#E4DAC4", "#0B0B12"), corner_radius=8)
        self.video_holder.pack(fill="both", expand=True, padx=16, pady=12)
        self.video_holder.pack_propagate(False)
        self.video_label = ctk.CTkLabel(self.video_holder, text="Waiting for camera...", text_color=SUB,
                                        font=self.fonts["serif"])
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.hint_label = ctk.CTkLabel(vid, text="", text_color=WARN, wraplength=640, font=self.fonts["small"])
        self.hint_label.pack(pady=(0, 10))

        # ---- right: tabs
        tabs = ctk.CTkTabview(
            body, fg_color=CARD, border_width=1, border_color=GOLD, corner_radius=10,
            segmented_button_fg_color=SEG_OFF, segmented_button_selected_color=BURG,
            segmented_button_selected_hover_color=BURG_H, segmented_button_unselected_color=SEG_OFF,
            segmented_button_unselected_hover_color=SEG_OFF_H, text_color=ON_BURG)
        tabs.grid(row=0, column=1, sticky="nsew")
        self.tabs = tabs
        for name in ("Translate", "Train", "History", "Models"):
            tabs.add(name)
        self._tab_translate(tabs.tab("Translate"))
        self._tab_train(tabs.tab("Train"))
        self._tab_history(tabs.tab("History"))
        self._tab_models(tabs.tab("Models"))

    def _tab_translate(self, tab):
        ctk.CTkLabel(tab, text="DETECTED SIGN", font=self.fonts["small_b"], text_color=SUB).pack(pady=(10, 0))
        self.pred_label = ctk.CTkLabel(tab, text="-", font=self.fonts["big"], text_color=BURG)
        self.pred_label.pack(pady=(0, 2))
        self.src_label = ctk.CTkLabel(tab, text="", font=self.fonts["small"], text_color=SUB)
        self.src_label.pack()
        self.conf_bar = ctk.CTkProgressBar(tab, width=340, progress_color=GOLD, fg_color=CARD2)
        self.conf_bar.set(0)
        self.conf_bar.pack(pady=(8, 2))
        self.hold_bar = ctk.CTkProgressBar(tab, width=340, height=5, progress_color=BURG, fg_color=CARD2)
        self.hold_bar.set(0)
        self.hold_bar.pack(pady=(2, 0))
        ctk.CTkLabel(tab, text="confidence  /  hold steady to confirm", font=self.fonts["small"],
                     text_color=SUB).pack(pady=(2, 8))
        rule(tab).pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(tab, text="TRANSCRIPT", font=self.fonts["small_b"], text_color=SUB).pack(anchor="w", padx=16)
        self.text = ctk.CTkTextbox(tab, height=150, font=self.fonts["serif"], fg_color=FIELD, text_color=TEXT,
                                   border_width=1, border_color=GOLD, wrap="word")
        self.text.pack(fill="x", padx=14, pady=(2, 8))
        g = ctk.CTkFrame(tab, fg_color="transparent")
        g.pack(fill="x", padx=10)
        for c in range(3):
            g.grid_columnconfigure(c, weight=1)
        gold_button(g, "Space", lambda: self.insert_raw(" ")).grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        burg_button(g, "Backspace", self.backspace).grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        ghost_button(g, "Clear", self.clear_text).grid(row=0, column=2, padx=4, pady=3, sticky="ew")
        ghost_button(g, "Copy", self.copy_text).grid(row=1, column=0, padx=4, pady=3, sticky="ew")
        ghost_button(g, "Speak", self.speak_text).grid(row=1, column=1, padx=4, pady=3, sticky="ew")
        ghost_button(g, "Save...", self.save_text).grid(row=1, column=2, padx=4, pady=3, sticky="ew")

    def _tab_train(self, tab):
        ctk.CTkLabel(tab, text="Teach the app your own signs. Type a word or letter, then record.",
                     text_color=SUB, wraplength=380, justify="left", font=self.fonts["small"]
                     ).pack(anchor="w", padx=14, pady=(10, 6))
        self.label_entry = ctk.CTkEntry(tab, placeholder_text="Label (e.g. THANK YOU)", fg_color=FIELD,
                                        border_color=GOLD, text_color=TEXT, height=34)
        self.label_entry.pack(fill="x", padx=14, pady=4)
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=10)
        row.grid_columnconfigure((0, 1), weight=1)
        self.rec_hold_btn = gold_button(row, "Record hold sign", self.toggle_hold_record)
        self.rec_hold_btn.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        burg_button(row, "Record motion sign", self.record_motion).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        self.rec_label = ctk.CTkLabel(tab, text="Hold: show the sign, press record, move slightly, press stop.\n"
                                                "Motion: press record, wait for the countdown, perform it once.\n"
                                                "Repeat 8-15 times for motion signs.",
                                      text_color=SUB, wraplength=380, justify="left", font=self.fonts["small"])
        self.rec_label.pack(anchor="w", padx=14, pady=6)
        rule(tab).pack(fill="x", padx=14, pady=4)
        head = ctk.CTkFrame(tab, fg_color="transparent")
        head.pack(fill="x", padx=14)
        ctk.CTkLabel(head, text="TRAINED SIGNS", font=self.fonts["small_b"], text_color=SUB).pack(side="left")
        ghost_button(head, "Delete all", self.clear_all_samples, width=80, height=24).pack(side="right")
        self.samples_frame = ctk.CTkScrollableFrame(tab, fg_color=FIELD, corner_radius=6, height=200)
        self.samples_frame.pack(fill="both", expand=True, padx=14, pady=(4, 10))

    def _tab_history(self, tab):
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkLabel(top, text="RECOGNITION LOG", font=self.fonts["small_b"], text_color=SUB).pack(side="left")
        ghost_button(top, "Clear", self.clear_history, width=70, height=24).pack(side="right")
        self.hist = ctk.CTkTextbox(tab, fg_color=FIELD, text_color=TEXT, border_width=1, border_color=GOLD,
                                   font=ctk.CTkFont(family="Consolas", size=12), wrap="none")
        self.hist.pack(fill="both", expand=True, padx=14, pady=(2, 12))
        self.hist.configure(state="disabled")

    def _tab_models(self, tab):
        ctk.CTkLabel(tab, text="Pretrained models live in the models folder. Download them here, or paste the "
                               "files in yourself.", text_color=SUB, wraplength=390, justify="left",
                     font=self.fonts["small"]).pack(anchor="w", padx=14, pady=(10, 6))
        self.model_rows = {}
        for key, info in MODELS.items():
            box = ctk.CTkFrame(tab, fg_color=CARD2, corner_radius=8)
            box.pack(fill="x", padx=12, pady=5)
            head = ctk.CTkFrame(box, fg_color="transparent")
            head.pack(fill="x", padx=10, pady=(8, 0))
            ctk.CTkLabel(head, text=info["title"], font=self.fonts["h2"], text_color=TEXT).pack(side="left")
            status = ctk.CTkLabel(head, text="", font=self.fonts["small_b"])
            status.pack(side="right")
            ctk.CTkLabel(box, text=info["purpose"], text_color=SUB, wraplength=380, justify="left",
                         font=self.fonts["small"]).pack(anchor="w", padx=10)
            ctk.CTkLabel(box, text="File: models/" + info["file"], text_color=SUB,
                         font=self.fonts["small"]).pack(anchor="w", padx=10)
            btns = ctk.CTkFrame(box, fg_color="transparent")
            btns.pack(fill="x", padx=6, pady=(4, 8))
            dl = gold_button(btns, "Download", lambda k=key: self.download_one(k), width=90, height=26)
            dl.pack(side="left", padx=4)
            ghost_button(btns, "Copy link", lambda k=key: self.copy_link(k), width=80, height=26).pack(side="left")
            self.model_rows[key] = (status, dl)
        bottom = ctk.CTkFrame(tab, fg_color="transparent")
        bottom.pack(fill="x", padx=12, pady=6)
        ghost_button(bottom, "Open models folder", lambda: open_folder(MODEL_DIR), width=150).pack(side="left", padx=4)
        ghost_button(bottom, "Re-check", self.recheck_models, width=80).pack(side="left", padx=4)
        ctk.CTkLabel(tab, text="Credits: MediaPipe (Google, Apache-2.0). ASL A-Z classifier by Muhib-Mehdi (MIT). "
                               "J and Z involve motion, so the static classifier is unreliable for them.",
                     text_color=SUB, wraplength=390, justify="left", font=self.fonts["small"]
                     ).pack(anchor="w", padx=14, pady=(6, 4))

    def _build_status(self):
        bar = ctk.CTkFrame(self, fg_color=CARD, border_width=1, border_color=GOLD, corner_radius=0, height=30)
        bar.pack(fill="x", side="bottom")
        self.status_left = ctk.CTkLabel(bar, text="", text_color=SUB, font=self.fonts["small"])
        self.status_left.pack(side="left", padx=14, pady=4)
        self.status_right = ctk.CTkLabel(bar, text="", text_color=GOLD, font=self.fonts["small"])
        self.status_right.pack(side="right", padx=14)

    # ------------------------------------------------------------ actions
    def flash(self, msg, secs=4.0):
        self.status_right.configure(text=msg)
        self.flash_until = time.time() + secs

    def _theme_label(self):
        dark = ctk.get_appearance_mode() == "Dark"
        self.theme_btn.configure(text="Light mode" if dark else "Dark mode")

    def toggle_theme(self):
        new = "Light" if ctk.get_appearance_mode() == "Dark" else "Dark"
        self.cfg["appearance"] = new
        ctk.set_appearance_mode(new)
        self._theme_label()
        save_settings(self.cfg)

    def open_settings(self):
        if self.settings_win is not None and self.settings_win.winfo_exists():
            self.settings_win.lift()
            self.settings_win.focus()
            return
        self.settings_win = SettingsWindow(self)

    def cfg_changed(self, what=None):
        if what == "camera":
            self.engine.request_camera()
        elif what == "models":
            self.engine.reload_models()
        elif what == "theme":
            ctk.set_appearance_mode(self.cfg["appearance"])
            self._theme_label()
        self.engine.apply_config()
        save_settings(self.cfg)

    def set_mode(self, mode):
        self.cfg["mode"] = mode
        self.mode_btn.set(mode)
        self.engine.set_mode(mode)
        self.rec_hold_btn.configure(text="Record hold sign")
        save_settings(self.cfg)

    def scan_cameras(self, done=None):
        def work():
            found = probe_cameras(int(self.cfg["camera_index"]))
            self.ui_q.put(("cameras", found, done))
        threading.Thread(target=work, daemon=True).start()

    # ---- transcript
    def get_text(self):
        return self.text.get("1.0", "end-1c")

    def insert_raw(self, s):
        self.text.insert("end", s)
        self.text.see("end")

    def append_token(self, token, kind):
        cur = self.get_text()
        token = apply_case(token, kind, cur, self.cfg["text_case"])
        if kind == "word":
            if cur and not cur.endswith((" ", "\n")):
                token = " " + token
            token += " "
        self.insert_raw(token)

    def backspace(self):
        if self.get_text():
            self.text.delete("end-2c", "end-1c")

    def clear_text(self):
        self.text.delete("1.0", "end")

    def copy_text(self):
        self.clipboard_clear()
        self.clipboard_append(self.get_text())
        self.flash("Copied to clipboard")

    def speak_text(self):
        speak_async(self.get_text())

    def save_text(self):
        text = self.get_text().strip()
        if not text:
            self.flash("Nothing to save yet")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="translation.txt",
                                            filetypes=[("Text file", "*.txt")])
        if path:
            Path(path).write_text(text + "\n", encoding="utf-8")
            self.flash(f"Saved {Path(path).name}")

    # ---- history
    def log(self, line):
        self.hist.configure(state="normal")
        self.hist.insert("end", line + "\n")
        self.hist.see("end")
        self.hist.configure(state="disabled")
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def clear_history(self):
        self.hist.configure(state="normal")
        self.hist.delete("1.0", "end")
        self.hist.configure(state="disabled")

    # ---- training
    def _label(self):
        lb = self.label_entry.get().strip()
        if not lb:
            self.flash("Type a label first (Train tab)")
            self.label_entry.focus()
        return lb

    def toggle_hold_record(self):
        info = self.engine.snapshot()[2]
        if info.get("rec_active") and info.get("rec_kind") == "hold":
            self.engine.stop_hold()
            self.rec_hold_btn.configure(text="Record hold sign")
            return
        lb = self._label()
        if not lb:
            return
        if self.cfg["mode"] != "Custom Hold":
            self.set_mode("Custom Hold")
        self.engine.start_hold(lb)
        self.rec_hold_btn.configure(text="Stop recording")

    def record_motion(self):
        lb = self._label()
        if not lb:
            return
        if self.cfg["mode"] != "Custom Motion":
            self.set_mode("Custom Motion")
        self.engine.start_motion(lb)

    def refresh_samples(self):
        for w in self.samples_frame.winfo_children():
            w.destroy()
        rows = self.engine.sample_summary()
        if not rows:
            ctk.CTkLabel(self.samples_frame, text="Nothing trained yet.", text_color=SUB).pack(pady=14)
        for kind, label, n in rows:
            r = ctk.CTkFrame(self.samples_frame, fg_color="transparent")
            r.pack(fill="x", pady=2)
            ctk.CTkLabel(r, text=label, text_color=TEXT, font=self.fonts["h2"], anchor="w").pack(side="left", padx=6)
            ctk.CTkLabel(r, text=f"{kind} - {n}", text_color=SUB, font=self.fonts["small"]).pack(side="left", padx=4)
            burg_button(r, "Delete", lambda k=kind, lb=label: self.engine.delete_label(k, lb),
                        width=56, height=22).pack(side="right", padx=4)

    def clear_all_samples(self):
        self.engine.clear_samples()

    # ---- models
    def refresh_models(self):
        for key, (status, dl) in self.model_rows.items():
            if model_ok(key):
                status.configure(text="INSTALLED", text_color=OK)
                dl.configure(text="Re-download")
            else:
                state = self.dl_state.get(key)
                if state == "busy":
                    status.configure(text="DOWNLOADING...", text_color=WARN)
                elif state == "fail":
                    status.configure(text="DOWNLOAD FAILED - paste file manually", text_color=BAD)
                else:
                    status.configure(text="MISSING", text_color=BAD)
                dl.configure(text="Download")

    def recheck_models(self):
        self.dl_state.clear()
        self.refresh_models()
        self.engine.reload_models()
        self.flash("Models re-checked")

    def copy_link(self, key):
        self.clipboard_clear()
        self.clipboard_append(MODELS[key]["url"])
        self.flash(f"Link copied - save the file as models/{MODELS[key]['file']}")

    def download_one(self, key):
        if self.dl_state.get(key) == "busy":
            return
        self.dl_state[key] = "busy"
        self.refresh_models()

        def work():
            try:
                download_model(key)
                self.dl_state[key] = "ok"
                self.ui_q.put(("model_done", key, ""))
            except Exception as exc:
                self.dl_state[key] = "fail"
                self.ui_q.put(("model_done", key, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def download_missing(self):
        need = [k for k in ("gesture", "asl") if not model_ok(k)]
        if need and not model_ok("gesture"):
            pass
        for k in need:
            self.download_one(k)

    # ---------------------------------------------------------------- loop
    def _tick(self):
        try:
            self._drain()
            frame, fid, info = self.engine.snapshot()
            self._show_frame(frame, fid, info)
            self._show_info(info)
        except Exception as exc:
            print("[ui]", type(exc).__name__, exc)
        if time.time() > self.flash_until and self.status_right.cget("text"):
            self.status_right.configure(text="")
        self.after(33, self._tick)

    def _drain(self):
        while True:
            try:
                ev = self.q.get_nowait()
            except queue.Empty:
                break
            kind = ev[0]
            if kind == "token":
                _, tok, tkind, conf, mode = ev
                self.append_token(tok, tkind)
                self.log(f"[{datetime.now():%H:%M:%S}] {mode:<13} {tok}  ({conf * 100:.0f}%)")
                if self.cfg["speak_on_word"] and tkind == "word":
                    speak_async(tok)
            elif kind == "space":
                cur = self.get_text()
                if cur and not cur.endswith((" ", "\n")):
                    self.insert_raw(" ")
            elif kind == "samples":
                self.refresh_samples()
                self.rec_hold_btn.configure(text="Record hold sign")
            elif kind == "flash":
                self.flash(ev[1])
        while True:
            try:
                ev = self.ui_q.get_nowait()
            except queue.Empty:
                break
            if ev[0] == "cameras":
                self.cameras = ev[1] or [self.cfg["camera_index"]]
                if ev[2]:
                    ev[2]()
            elif ev[0] == "model_done":
                _, key, err = ev
                if err:
                    self.flash(f"Download failed: {err}", 8)
                else:
                    self.flash(f"{MODELS[key]['title']} installed")
                self.refresh_models()
                if model_ok(key):
                    self.engine.reload_models()

    def _show_frame(self, frame, fid, info):
        if frame is None:
            if self.frame_seen != -2:
                self.video_label.configure(image=None, text=info.get("status", "Waiting for camera..."))
                self.frame_seen = -2
            return
        if fid == self.frame_seen:
            return
        self.frame_seen = fid
        hw, hh = self.video_holder.winfo_width() - 8, self.video_holder.winfo_height() - 8
        if hw < 50 or hh < 50:
            return
        fh, fw = frame.shape[:2]
        scale = min(hw / fw, hh / fh)
        size = (max(1, int(fw * scale)), max(1, int(fh * scale)))
        img = Image.fromarray(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        photo = ctk.CTkImage(light_image=img, dark_image=img, size=size)
        self.video_label.configure(image=photo, text="")
        self.video_label._photo = photo  # keep reference

    def _show_info(self, info):
        pred = info.get("pred")
        self.pred_label.configure(text=pred if pred else "-")
        self.conf_bar.set(max(0.0, min(1.0, info.get("conf", 0.0))))
        self.hold_bar.set(info.get("progress", 0.0))
        self.src_label.configure(text=info.get("source", "") or "")
        hint = info.get("hint", "")
        rec = info.get("rec_text", "")
        self.hint_label.configure(text=rec or hint, text_color=BAD if rec else WARN)
        if info.get("rec_active") is False and self.rec_hold_btn.cget("text") == "Stop recording":
            self.rec_hold_btn.configure(text="Record hold sign")
        cam = self.cfg["camera_index"]
        tr = info.get("tracker") or "none"
        self.status_left.configure(
            text=f"{info.get('status', '')}   |   Camera {cam}   |   {info.get('fps', 0):.0f} FPS   |   "
                 f"hands: {info.get('hands', 0)}   |   tracker: {tr}   |   mode: {self.cfg['mode']}")

    def on_close(self):
        try:
            self.cfg["geometry"] = self.geometry().split("+")[0]
        except Exception:
            pass
        save_settings(self.cfg)
        self.engine.stop()
        self.engine.join(timeout=2)
        self.engine.shutdown_save()
        self.destroy()


def check():
    import tkinter
    print("Python", sys.version.split()[0], "|", platform.system())
    print("OpenCV", cv2.__version__, "| NumPy", np.__version__, "| customtkinter", ctk.__version__,
          "| Tk", tkinter.TkVersion)
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    assert hasattr(vision, "GestureRecognizer") and hasattr(vision, "HandLandmarker")
    print("MediaPipe", mp.__version__)
    for rt in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
        try:
            importlib.import_module(rt)
            print("TFLite runtime:", rt)
            break
        except Exception:
            pass
    else:
        print("TFLite runtime: none (ASL Alphabet mode disabled; run: pip install ai-edge-litert)")
    print("Setup OK.")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        app = App()
        if "--selftest" in sys.argv:
            app.after(4000, app.on_close)
        app.mainloop()
'''

ASL_MODEL_B64 = "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"

REQUIREMENTS = """opencv-python>=4.8
mediapipe>=0.10.14
numpy>=1.24
customtkinter>=5.2.2
pillow>=10.0
"""
OPTIONAL = ["ai-edge-litert", "pyttsx3"]

README = r'''# Ars Signorum - Sign Language Interpreter

Camera -> hand tracking -> pretrained recognisers -> words.

## Run
    python app.py      (or run.bat / run.sh)

## Modes (top of the camera panel)
- Gestures       pretrained Google Gesture Recognizer: open palm, thumbs up/down, fist, pointing up,
                 victory, I-love-you. Each maps to a word you can change in Settings.
- Alphabet       pretrained ASL letters A-Z. Hold a letter steady to type it. Take your hand out of view
                 to insert a space. (J and Z need motion, so they are unreliable.)
- Custom Hold    signs you teach it (held hand shapes).
- Custom Motion  signs you teach it (moving gestures).

## Models  (models/ folder, or the Models tab)
    gesture_recognizer.task   https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
    keypoint_classifier.tflite (ASL A-Z; bundled)  https://raw.githubusercontent.com/Muhib-Mehdi/ASL-Recognition-System/main/model/keypoint_classifier/keypoint_classifier.tflite
    hand_landmarker.task      (fallback tracker)   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
If a download is blocked, open the link in a browser and save the file with that exact name into models/.

## Shortcuts
Ctrl+, settings | Ctrl+T theme | Ctrl+S save transcript | F11 fullscreen

## Credits
MediaPipe (Google, Apache-2.0). ASL A-Z keypoint classifier by Muhib-Mehdi (MIT).
'''

RUN_BAT = "@echo off\r\ncd /d \"%~dp0\"\r\nif exist \".venv\\Scripts\\python.exe\" (\r\n  \".venv\\Scripts\\python.exe\" app.py\r\n) else (\r\n  python app.py\r\n)\r\nif errorlevel 1 pause\r\n"
RUN_SH = "#!/usr/bin/env bash\ncd \"$(dirname \"$0\")\"\nif [ -x .venv/bin/python ]; then .venv/bin/python app.py; else python3 app.py; fi\n"


def write(path, text, binary=False):
    if binary:
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
    print("  wrote ", path.name)


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="SignLanguageAI")
    ap.add_argument("--no-venv", action="store_true")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    args = ap.parse_args()

    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required.")
    try:
        import tkinter  # noqa: F401
    except ImportError:
        sys.exit("Tk is missing from this Python.\n  Linux:  sudo apt install python3-tk\n"
                 "  Windows/macOS: reinstall Python from python.org (keep 'tcl/tk' ticked).")

    root = Path(__file__).resolve().parent / args.dir
    print("Creating project in:", root)
    for sub in ("", "data", "models"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    write(root / "app.py", APP_PY)
    write(root / "requirements.txt", REQUIREMENTS + "\n".join("# optional: " + o for o in OPTIONAL) + "\n")
    write(root / "README.md", README)
    write(root / "run.bat", RUN_BAT)
    write(root / "run.sh", RUN_SH)
    try:
        (root / "run.sh").chmod(0o755)
    except OSError:
        pass
    asl = root / "models" / "keypoint_classifier.tflite"
    if not asl.exists():
        write(asl, zlib.decompress(base64.b64decode(ASL_MODEL_B64)), binary=True)

    py = Path(sys.executable)
    if not args.no_venv:
        venv = root / ".venv"
        vpy = venv / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
        if not vpy.exists():
            print("Creating virtual environment...")
            if run([sys.executable, "-m", "venv", venv]).returncode != 0:
                sys.exit("Could not create a virtual environment. Re-run with --no-venv.")
        py = vpy

    if not args.no_install:
        print("Installing packages (first run can take a few minutes)...")
        run([py, "-m", "pip", "install", "--upgrade", "pip", "-q"])
        if run([py, "-m", "pip", "install", "-r", root / "requirements.txt"]).returncode != 0:
            sys.exit("\nInstall failed. MediaPipe/OpenCV do not always have wheels for brand-new Python "
                     "releases.\nInstall Python 3.12 or 3.13 and run this file with it "
                     "(e.g.  py -3.13 sign_language_setup.py).")
        for pkg in OPTIONAL:
            if run([py, "-m", "pip", "install", pkg, "-q"]).returncode != 0:
                print(f"  (optional package {pkg} could not be installed - related feature disabled)")

    if run([py, root / "app.py", "--check"], cwd=root).returncode != 0:
        sys.exit("Setup check failed - see the message above.")
    print("\nReady. Start any time with run.bat (Windows) or ./run.sh (Mac/Linux).")
    if not args.no_run:
        run([py, root / "app.py"], cwd=root)


if __name__ == "__main__":
    main()