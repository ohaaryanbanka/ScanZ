import json
import threading
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.neighbors import KNeighborsClassifier

DATA_FILE = Path(__file__).parent / "data" / "gestures.json"
MAX_K = 5               # how many neighbours vote
UNKNOWN_DISTANCE = 1.5  # if the closest samples are farther than this -> "unknown"
MAX_HANDS = 2

app = Flask(__name__)


def normalize_hand(points):
    """21 points of (x, y, z) -> 63 numbers that ignore hand position and size."""
    pts = np.asarray(points, dtype=float)
    if pts.shape != (21, 3):
        raise ValueError("each hand must have 21 points of (x, y, z)")
    wrist = pts[0].copy()
    pts = pts - wrist                           # wrist becomes the origin
    scale = np.linalg.norm(pts, axis=1).max()   # biggest distance from the wrist
    if scale < 1e-6:
        raise ValueError("degenerate hand")
    return (pts / scale).flatten(), wrist, scale


def build_features(hands):
    """List of 1 or 2 hands -> one fixed-length vector of 129 numbers.

    [ hand A (63) | hand B (63, zeros if absent) | offset between wrists (3) ]
    Hands are sorted left-to-right on screen so the order is always the same.
    """
    if not isinstance(hands, list) or not 1 <= len(hands) <= MAX_HANDS:
        raise ValueError("send 1 or 2 hands")
    parsed = [normalize_hand(h) for h in hands]
    parsed.sort(key=lambda item: item[1][0])    # sort by wrist x position

    vectors = [p[0] for p in parsed]
    offset = np.zeros(3)
    if len(parsed) == 2:
        # where is hand B relative to hand A, measured in hand-sizes
        mean_scale = (parsed[0][2] + parsed[1][2]) / 2
        offset = (parsed[1][1] - parsed[0][1]) / mean_scale
    while len(vectors) < MAX_HANDS:
        vectors.append(np.zeros(63))            # pad: no second hand
    return np.concatenate(vectors + [offset])


class GestureStore:
    """Keeps labelled samples, saves them to disk, and a trained KNN model."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()   # Flask can handle 2 requests at once
        self.samples = {}              # {"Hello": [[129 numbers], ...], ...}
        self.model = None
        if self.path.exists():
            self.samples = json.loads(self.path.read_text())
        self._train()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.samples))

    def _train(self):
        X, y = [], []
        for label, vectors in self.samples.items():
            X.extend(vectors)
            y.extend([label] * len(vectors))
        if not X:
            self.model = None
            return
        k = min(MAX_K, len(X))
        self.model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        self.model.fit(np.array(X), np.array(y))

    def add(self, label, hands):
        vector = build_features(hands).tolist()
        with self.lock:
            self.samples.setdefault(label, []).append(vector)
            self._train()
            self._save()

    def delete(self, label):
        with self.lock:
            removed = self.samples.pop(label, None) is not None
            self._train()
            self._save()
        return removed

    def reset(self):
        with self.lock:
            self.samples = {}
            self._train()
            self._save()

    def counts(self):
        with self.lock:
            return {label: len(v) for label, v in self.samples.items()}

    def predict(self, hands):
        vector = build_features(hands).reshape(1, -1)
        with self.lock:
            if self.model is None:
                return None
            distances, _ = self.model.kneighbors(vector)
            probs = self.model.predict_proba(vector)[0]
            best = int(np.argmax(probs))
            label = str(self.model.classes_[best])
            confidence = float(probs[best])
            mean_distance = float(distances.mean())
        if mean_distance > UNKNOWN_DISTANCE:
            return {"label": "unknown", "confidence": 0.0, "distance": mean_distance}
        return {"label": label, "confidence": confidence, "distance": mean_distance}


store = GestureStore(DATA_FILE)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/gestures")
def list_gestures():
    return jsonify(store.counts())


@app.post("/api/samples")
def add_sample():
    body = request.get_json(silent=True) or {}
    label = str(body.get("label", "")).strip()
    if not label or len(label) > 40 or label.lower() == "unknown":
        return jsonify(error="label must be 1-40 chars and not 'unknown'"), 400
    try:
        store.add(label, body.get("hands"))
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(store.counts())


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    try:
        result = store.predict(body.get("hands"))
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    if result is None:
        return jsonify(label=None, confidence=0.0, message="no gestures trained yet")
    return jsonify(result)


@app.delete("/api/gestures/<label>")
def delete_gesture(label):
    if not store.delete(label):
        return jsonify(error="no such gesture"), 404
    return jsonify(store.counts())


@app.post("/api/reset")
def reset():
    store.reset()
    return jsonify(store.counts())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)