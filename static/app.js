import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const FACE_MESH_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const FACE_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/";

// ---- settings you can tweak ----
const SAMPLES_PER_RECORDING = 60;
const SAMPLE_INTERVAL_MS = 100;    // 10 samples per second while teaching
const PREDICT_INTERVAL_MS = 100;   // ask Python 10 times per second
const MIN_CONFIDENCE = 0.6;        // ignore weak guesses
const VOTE_WINDOW = 7;             // gesture = majority of the last 7 predictions
const VOTE_NEEDED = 4;             // ...and it must win at least this many
const STABLE_COUNT = 6;            // same gesture N frames in a row before speaking it
const CAMERA_WIDTH = 960;          // lighter stream = smoother tracking
const CAMERA_HEIGHT = 540;

const FACE_INTERVAL_MS = 200;      // 5 age/sex/emotion analyses per second
const FACE_MISS_LIMIT = 4;         // keep old values for this many missed frames
const AGE_SMOOTH = 0.12;           // 0..1, smaller = smoother but slower to react
const SEX_SMOOTH = 0.2;
const EMOTION_SMOOTH = 0.3;
const BOX_SMOOTH = 0.35;           // how fast the corner brackets follow your face
const EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "disgusted", "fearful"];

// eye tracker tuning (try changing these if blinks or gaze feel off)
const BLINK_CLOSE = 0.17;          // eye openness below this = closed
const BLINK_OPEN = 0.21;           // ...and above this = open again
const GAZE_X_THRESHOLD = 0.045;
const GAZE_Y_THRESHOLD = 0.09;

// colours used on the canvas
const ACCENT = "#a9bcff";
const ACCENT_2 = "#c3b4ff";

// ---- page elements ----
const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const statusEl = $("status");
const startBtn = $("start-btn");
const recordBtn = $("record-btn");
const labelInput = $("label-input");
const gestureLabel = $("gesture-label");
const gestureConf = $("gesture-conf");
const banner = $("banner");
const speakBox = $("speak");
const tgEyes = $("tg-eyes");
const tgScan = $("tg-scan");
const tg3d = $("tg-3d");
const viewer = $("viewer3d");
const vctx = viewer.getContext("2d");

// ---- state ----
let mode = "hand";                 // "hand" or "face"
let handLandmarker = null;
let faceLandmarker = null;         // loaded the first time you switch an overlay on
let faceLandmarkerLoading = null;
let drawing = null;
let lastVideoTime = -1;

// hand mode
let latestHands = [];              // one entry per hand, each is 21 points [x, y, z]
let recording = false;
let recordedCount = 0;
let lastSampleAt = 0;
let predicting = false;
let lastPredictAt = 0;
let votes = [];
let stableLabel = null;
let stableCount = 0;
let spokenLabel = null;

// face mode (age / sex / emotion)
let faceBusy = false;
let lastFaceAt = 0;
let faceMisses = 0;
let targetBox = null;
let shownBox = null;
let smooth = null;
let emotionLabel = null;

// face mode (overlays)
let faceLM = null;                 // 478 landmarks of the current frame, or null
let eyeState = { closed: false, blinks: 0, gx: 0, gy: 0, open: 0.3, lastDom: 0 };
let viewYaw = 0, viewPitch = 0;    // manual rotation of the 3D viewer
let dragging = false, dragX = 0, dragY = 0;

// ---- talking to the Python server ----
async function api(path, method = "GET", body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  return res.json();
}

async function refreshGestureList() {
  const counts = await api("/api/gestures");
  const list = $("gesture-list");
  list.innerHTML = "";
  for (const [label, n] of Object.entries(counts)) {
    const li = document.createElement("li");

    const name = document.createElement("span");
    name.textContent = label;

    const count = document.createElement("span");
    count.className = "count";
    count.textContent = `${n} samples`;

    const del = document.createElement("button");
    del.textContent = "x";
    del.onclick = async () => {
      await api(`/api/gestures/${encodeURIComponent(label)}`, "DELETE");
      refreshGestureList();
    };

    li.append(name, count, del);
    list.append(li);
  }
}

// ---- mode switch (two "pages") ----
function setMode(newMode) {
  mode = newMode;
  $("mode-switch").dataset.mode = mode;
  $("mode-hand").classList.toggle("active", mode === "hand");
  $("mode-face").classList.toggle("active", mode === "face");
  $("panel-hand").classList.toggle("hidden", mode !== "hand");
  $("panel-face").classList.toggle("hidden", mode !== "face");
  $("hud-mode").textContent = mode === "hand" ? "HAND TRACKING" : "FACE ANALYSIS";

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  latestHands = [];
  votes = [];
  recording = false;
  faceLM = null;
  recordBtn.disabled = !handLandmarker;
  showGesture(null, 0);
  clearFace();
  if (handLandmarker) {
    statusEl.textContent = mode === "hand" ? "Show a gesture, or teach a new one." : "Look at the camera.";
  }
}
$("mode-hand").onclick = () => setMode("hand");
$("mode-face").onclick = () => setMode("face");

// ---- start everything ----
async function start() {
  startBtn.disabled = true;
  try {
    statusEl.textContent = "Asking for camera access...";
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: CAMERA_WIDTH },
        height: { ideal: CAMERA_HEIGHT },
        frameRate: { ideal: 30 },
      },
    });
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    document.documentElement.style.setProperty("--ar", video.videoWidth / video.videoHeight);
    drawing = new DrawingUtils(ctx);

    statusEl.textContent = "Loading hand model...";
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    handLandmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
    });

    statusEl.textContent = "Loading face models...";
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(FACE_MODEL_URL),
      faceapi.nets.ageGenderNet.loadFromUri(FACE_MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(FACE_MODEL_URL),
    ]);

    $("start-overlay").classList.add("hidden");
    setMode(mode);
    refreshGestureList();
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Could not start: " + err.message;
    startBtn.disabled = false;
  }
}

// ---- FPS counter shown in the corner of the video ----
let fpsFrames = 0, fpsStart = performance.now();
const MODE_NAMES = { hand: "HAND TRACKING", face: "FACE ANALYSIS" };

function countFrame(now) {
  fpsFrames++;
  if (now - fpsStart >= 1000) {
    const fps = Math.round((fpsFrames * 1000) / (now - fpsStart));
    $("hud-mode").textContent = `${MODE_NAMES[mode]} · ${fps} FPS`;
    fpsFrames = 0;
    fpsStart = now;
  }
}

// ---- runs on every video frame: only the SELECTED mode does any work ----
function loop() {
  const now = performance.now();

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    countFrame(now);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (mode === "hand") {
      const result = handLandmarker.detectForVideo(video, now);
      latestHands = result.landmarks.map((hand) => hand.map((p) => [p.x, p.y, p.z]));
      for (const hand of result.landmarks) {
        drawing.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: ACCENT_2, lineWidth: 4 });
        drawing.drawLandmarks(hand, { color: "#ffffff", radius: 4 });
      }
    } else {
      // landmarks (478 points) are only computed while an overlay is switched on
      if (faceLandmarker && (tgEyes.checked || tgScan.checked || tg3d.checked)) {
        const r = faceLandmarker.detectForVideo(video, now);
        faceLM = r.faceLandmarks.length ? r.faceLandmarks[0] : null;
      } else {
        faceLM = null;
      }

      if (faceLM && tg3d.checked) drawMesh(faceLM);
      drawFaceBox();
      if (faceLM && tgScan.checked) drawScan(faceLM, now);
      if (faceLM && tgEyes.checked) drawEyes(faceLM, now);
    }
  }

  if (mode === "hand") {
    handleRecording(now);
    handlePrediction(now);
  } else {
    handleFace(now);
    if (tg3d.checked) draw3DViewer(now);
  }
  requestAnimationFrame(loop);
}

/* =====================  HAND GESTURES  ===================== */

recordBtn.onclick = async () => {
  const label = labelInput.value.trim();
  if (!label) {
    statusEl.textContent = "Type what the gesture means first.";
    labelInput.focus();
    return;
  }
  if (recording) return;

  recordBtn.disabled = true;
  for (let s = 3; s > 0; s--) {
    statusEl.textContent = `Get ready: ${s}...`;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  if (mode !== "hand") { recordBtn.disabled = false; return; }
  recordedCount = 0;
  recording = true;
};

async function handleRecording(now) {
  if (!recording || latestHands.length === 0) return;
  if (now - lastSampleAt < SAMPLE_INTERVAL_MS) return;
  lastSampleAt = now;

  const label = labelInput.value.trim();
  const res = await api("/api/samples", "POST", { label, hands: latestHands });
  if (res.error) {
    statusEl.textContent = res.error;
    recording = false;
    recordBtn.disabled = false;
    return;
  }

  recordedCount++;
  statusEl.textContent = `Recording "${label}": ${recordedCount}/${SAMPLES_PER_RECORDING}`;
  if (recordedCount >= SAMPLES_PER_RECORDING) {
    recording = false;
    recordBtn.disabled = false;
    statusEl.textContent = `Learned "${label}". Try it!`;
    refreshGestureList();
  }
}

$("reset-btn").onclick = async () => {
  if (confirm("Delete every trained gesture?")) {
    await api("/api/reset", "POST");
    refreshGestureList();
  }
};

async function handlePrediction(now) {
  if (recording || predicting || now - lastPredictAt < PREDICT_INTERVAL_MS) return;
  lastPredictAt = now;

  if (latestHands.length === 0) {
    pushVote(null, 0);
    return;
  }

  predicting = true;
  try {
    const res = await api("/api/predict", "POST", { hands: latestHands });
    if (mode !== "hand") return;
    const good = res.label && res.label !== "unknown" && res.confidence >= MIN_CONFIDENCE;
    pushVote(good ? res.label : null, good ? res.confidence : 0);
  } finally {
    predicting = false;
  }
}

// A single prediction can flicker. Keep the last few and go with the majority.
function pushVote(label, conf) {
  votes.push({ label, conf });
  if (votes.length > VOTE_WINDOW) votes.shift();

  const tally = new Map();
  for (const v of votes) {
    if (!v.label) continue;
    const t = tally.get(v.label) || { n: 0, conf: 0 };
    t.n++; t.conf += v.conf;
    tally.set(v.label, t);
  }
  let winner = null, best = 0;
  for (const [name, t] of tally) if (t.n > best) { winner = name; best = t.n; }

  if (winner && best >= VOTE_NEEDED) showGesture(winner, tally.get(winner).conf / best);
  else showGesture(null, 0);
}

function showGesture(label, confidence) {
  gestureLabel.textContent = label ?? "-";
  gestureConf.style.width = `${Math.round(confidence * 100)}%`;
  banner.textContent = label ?? "";
  banner.classList.toggle("hidden", !label);

  if (label && label === stableLabel) stableCount++;
  else { stableLabel = label; stableCount = 1; }
  if (!label) spokenLabel = null;

  if (speakBox.checked && label && stableCount >= STABLE_COUNT && spokenLabel !== label) {
    spokenLabel = label;
    speechSynthesis.speak(new SpeechSynthesisUtterance(label));
  }
}

/* =====================  FACE: AGE / SEX / EMOTION  ===================== */

const barEls = {};
for (const name of EMOTIONS) {
  const row = document.createElement("div");
  row.className = "bar";
  row.innerHTML = `<span>${name}</span><div class="track"><div class="fill"></div></div><span class="pct">0%</span>`;
  $("emotion-bars").append(row);
  barEls[name] = { row, fill: row.querySelector(".fill"), pct: row.querySelector(".pct") };
}

const ema = (old, value, alpha) => (old == null ? value : old + (value - old) * alpha);

async function handleFace(now) {
  if (faceBusy || now - lastFaceAt < FACE_INTERVAL_MS) return;
  faceBusy = true;
  lastFaceAt = now;
  try {
    const det = await faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 }))
      .withFaceExpressions()
      .withAgeAndGender();
    if (mode !== "face") return;

    if (det) {
      faceMisses = 0;
      updateFace(det);
    } else if (++faceMisses >= FACE_MISS_LIMIT) {
      clearFace();
    }
  } finally {
    faceBusy = false;
  }
}

function updateFace(det) {
  const b = det.detection.box;
  targetBox = { x: b.x, y: b.y, width: b.width, height: b.height };
  if (!shownBox) shownBox = { ...targetBox };

  const pMale = det.gender === "male" ? det.genderProbability : 1 - det.genderProbability;
  if (!smooth) smooth = { age: null, pMale: null, expr: {} };
  smooth.age = ema(smooth.age, det.age, AGE_SMOOTH);
  smooth.pMale = ema(smooth.pMale, pMale, SEX_SMOOTH);
  for (const name of EMOTIONS) smooth.expr[name] = ema(smooth.expr[name], det.expressions[name], EMOTION_SMOOTH);

  const age = Math.round(smooth.age);
  $("face-age").textContent = `~${age}`;
  $("face-age-range").textContent = `rough range ${Math.max(1, age - 8)} to ${age + 8}`;

  const male = smooth.pMale >= 0.5;
  const sexConf = male ? smooth.pMale : 1 - smooth.pMale;
  $("face-sex").textContent = `${male ? "male" : "female"} (${Math.round(sexConf * 100)}%)`;
  $("face-sex-bar").style.width = `${Math.round(sexConf * 100)}%`;

  let top = EMOTIONS[0];
  for (const name of EMOTIONS) if (smooth.expr[name] > smooth.expr[top]) top = name;
  if (!emotionLabel || smooth.expr[top] > smooth.expr[emotionLabel] + 0.08) emotionLabel = top;
  $("face-emotion").textContent = emotionLabel;

  for (const name of EMOTIONS) {
    const p = Math.round(smooth.expr[name] * 100);
    barEls[name].fill.style.width = `${p}%`;
    barEls[name].pct.textContent = `${p}%`;
    barEls[name].row.classList.toggle("top", name === emotionLabel);
  }
}

function clearFace() {
  targetBox = null;
  shownBox = null;
  smooth = null;
  emotionLabel = null;
  faceMisses = 0;
  $("face-age").textContent = "-";
  $("face-age-range").textContent = "";
  $("face-sex").textContent = "-";
  $("face-sex-bar").style.width = "0%";
  $("face-emotion").textContent = "-";
  for (const name of EMOTIONS) {
    barEls[name].fill.style.width = "0%";
    barEls[name].pct.textContent = "0%";
    barEls[name].row.classList.remove("top");
  }
}

// four glowing corner brackets that glide after the face
function drawFaceBox() {
  if (!targetBox || !shownBox) return;
  for (const k of ["x", "y", "width", "height"]) {
    shownBox[k] += (targetBox[k] - shownBox[k]) * BOX_SMOOTH;
  }
  const { x, y, width: w, height: h } = shownBox;
  const L = Math.min(w, h) * 0.22;
  ctx.save();
  ctx.strokeStyle = ACCENT;
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.shadowColor = ACCENT;
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.moveTo(x, y + L); ctx.lineTo(x, y); ctx.lineTo(x + L, y);
  ctx.moveTo(x + w - L, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + L);
  ctx.moveTo(x + w, y + h - L); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - L, y + h);
  ctx.moveTo(x + L, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - L);
  ctx.stroke();
  ctx.restore();
}

/* =====================  FACE OVERLAYS (toggles)  ===================== */

// Load the 478-point face mesh model the first time an overlay is switched on.
async function ensureFaceLandmarker() {
  if (faceLandmarker) return;
  if (!faceLandmarkerLoading) {
    faceLandmarkerLoading = (async () => {
      statusEl.textContent = "Loading face mesh model...";
      const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
      faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MESH_MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numFaces: 1,
      });
      statusEl.textContent = "Look at the camera.";
    })().catch((err) => {
      console.error(err);
      statusEl.textContent = "Could not load face mesh: " + err.message;
      faceLandmarkerLoading = null;
    });
  }
  await faceLandmarkerLoading;
}

function onToggle() {
  $("card-eyes").classList.toggle("hidden", !tgEyes.checked);
  $("card-3d").classList.toggle("hidden", !tg3d.checked);
  if (tgEyes.checked || tgScan.checked || tg3d.checked) ensureFaceLandmarker();
}
tgEyes.onchange = tgScan.onchange = tg3d.onchange = onToggle;

// --- 3D mesh drawn on top of the video, coloured by depth ---
const DEPTH_COLORS = [
  "rgba(222, 230, 255, 0.60)",   // nearest to the camera: brightest
  "rgba(169, 188, 255, 0.40)",
  "rgba(111, 140, 255, 0.28)",
  "rgba(66, 90, 190, 0.18)",    // farthest: faintest
];

function drawDepthLines(lines, depthOf, project, target) {
  // 1) find the depth range, 2) put each line into one of 4 depth buckets, 3) stroke each bucket once
  let zmin = Infinity, zmax = -Infinity;
  for (const p of depthOf) { if (p < zmin) zmin = p; if (p > zmax) zmax = p; }
  const span = zmax - zmin || 1;
  const paths = DEPTH_COLORS.map(() => new Path2D());
  for (const { start, end } of lines) {
    const z = (depthOf[start] + depthOf[end]) / 2;
    const k = Math.min(DEPTH_COLORS.length - 1, Math.floor(((z - zmin) / span) * DEPTH_COLORS.length));
    const a = project(start), b = project(end);
    paths[k].moveTo(a.x, a.y);
    paths[k].lineTo(b.x, b.y);
  }
  target.lineWidth = 1;
  paths.forEach((path, i) => { target.strokeStyle = DEPTH_COLORS[i]; target.stroke(path); });
}

function drawMesh(lm) {
  const w = canvas.width, h = canvas.height;
  const depth = lm.map((p) => p.z);                       // smaller z = closer to the camera
  drawDepthLines(
    FaceLandmarker.FACE_LANDMARKS_TESSELATION,
    depth,
    (i) => ({ x: lm[i].x * w, y: lm[i].y * h }),
    ctx
  );
  // brighter outlines for the main features
  const bright = { color: "rgba(226, 232, 255, 0.9)", lineWidth: 2 };
  drawing.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, bright);
  drawing.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, bright);
  drawing.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, bright);
  drawing.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, bright);
}

// --- face scan: a laser line sweeps over the face and lights up the landmarks it touches ---
function drawScan(lm, now) {
  const w = canvas.width, h = canvas.height;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of lm) {
    x0 = Math.min(x0, p.x * w); x1 = Math.max(x1, p.x * w);
    y0 = Math.min(y0, p.y * h); y1 = Math.max(y1, p.y * h);
  }
  const phase = (now / 2400) % 2;                 // 0..2
  const t = phase < 1 ? phase : 2 - phase;        // 0..1..0 (ping-pong)
  const y = y0 + t * (y1 - y0);
  const band = (y1 - y0) * 0.07;
  const pad = (x1 - x0) * 0.08;

  ctx.save();
  // soft glow trail behind the laser
  const dir = phase < 1 ? -1 : 1;
  const g = ctx.createLinearGradient(0, y, 0, y + dir * (y1 - y0) * 0.25);
  g.addColorStop(0, "rgba(169, 188, 255, 0.28)");
  g.addColorStop(1, "rgba(169, 188, 255, 0)");
  ctx.fillStyle = g;
  ctx.fillRect(x0 - pad, Math.min(y, y + dir * (y1 - y0) * 0.25), x1 - x0 + 2 * pad, Math.abs((y1 - y0) * 0.25));

  // the laser line
  ctx.shadowColor = ACCENT;
  ctx.shadowBlur = 18;
  const lg = ctx.createLinearGradient(x0 - pad, 0, x1 + pad, 0);
  lg.addColorStop(0, "rgba(169, 188, 255, 0)");
  lg.addColorStop(0.5, "rgba(240, 244, 255, 1)");
  lg.addColorStop(1, "rgba(169, 188, 255, 0)");
  ctx.strokeStyle = lg;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x0 - pad, y);
  ctx.lineTo(x1 + pad, y);
  ctx.stroke();

  // landmarks near the laser light up
  ctx.shadowBlur = 8;
  for (const p of lm) {
    const d = Math.abs(p.y * h - y);
    if (d > band) continue;
    const a = 1 - d / band;
    ctx.fillStyle = `rgba(240, 244, 255, ${0.25 + 0.75 * a})`;
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 1.6 + 2.2 * a, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

// --- eye tracker (simple): a thin eyelid outline + a small ring and dot on each iris ---
const EYES = {
  right: { corners: [33, 133], top: 159, bottom: 145, iris: 468, irisRing: [469, 470, 471, 472], contour: () => FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE },
  left:  { corners: [362, 263], top: 386, bottom: 374, iris: 473, irisRing: [474, 475, 476, 477], contour: () => FaceLandmarker.FACE_LANDMARKS_LEFT_EYE },
};

function drawEyes(lm, now) {
  const w = canvas.width, h = canvas.height;
  const P = (i) => ({ x: lm[i].x * w, y: lm[i].y * h });

  for (const eye of Object.values(EYES)) {
    const iris = P(eye.iris);
    const irisR = eye.irisRing.reduce((s, i) => s + Math.hypot(P(i).x - iris.x, P(i).y - iris.y), 0) / 4;

    // 1) thin outline of the eyelids
    drawing.drawConnectors(lm, eye.contour(), { color: ACCENT, lineWidth: 1.5 });

    // 2) one ring around the iris, plus a small centre dot
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(iris.x, iris.y, irisR * 1.3, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(iris.x, iris.y, Math.max(2, irisR * 0.28), 0, Math.PI * 2);
    ctx.fill();
  }

  updateEyeStats(lm, now);
}

// where along the eye (inner corner -> outer corner) is the iris? 0.5 = middle
function irisRatio(p, inner, outer) {
  const dx = outer.x - inner.x, dy = outer.y - inner.y;
  return ((p.x - inner.x) * dx + (p.y - inner.y) * dy) / (dx * dx + dy * dy);
}

function updateEyeStats(lm, now) {
  const w = canvas.width, h = canvas.height;
  const P = (i) => ({ x: lm[i].x * w, y: lm[i].y * h });
  const R = EYES.right, L = EYES.left;

  // horizontal gaze: right eye looks toward its outer corner when you look right; left eye toward its inner corner
  const rr = irisRatio(P(R.iris), P(R.corners[1]), P(R.corners[0]));
  const lr = irisRatio(P(L.iris), P(L.corners[0]), P(L.corners[1]));
  const gx = ((rr - 0.5) + (0.5 - lr)) / 2;

  // vertical gaze: position of the iris between the upper and lower eyelid
  const vr = (P(R.iris).y - P(R.top).y) / (P(R.bottom).y - P(R.top).y || 1);
  const vl = (P(L.iris).y - P(L.top).y) / (P(L.bottom).y - P(L.top).y || 1);
  const gy = (vr + vl) / 2 - 0.5;

  // openness = eyelid gap relative to eye width (about 0.25-0.35 open, under 0.15 closed)
  const openR = Math.hypot(P(R.top).x - P(R.bottom).x, P(R.top).y - P(R.bottom).y) / Math.hypot(P(R.corners[0]).x - P(R.corners[1]).x, P(R.corners[0]).y - P(R.corners[1]).y);
  const openL = Math.hypot(P(L.top).x - P(L.bottom).x, P(L.top).y - P(L.bottom).y) / Math.hypot(P(L.corners[0]).x - P(L.corners[1]).x, P(L.corners[0]).y - P(L.corners[1]).y);
  const open = (openR + openL) / 2;

  const s = eyeState;
  s.gx = ema(s.gx, gx, 0.25);
  s.gy = ema(s.gy, gy, 0.25);
  s.open = ema(s.open, open, 0.5);

  if (!s.closed && open < BLINK_CLOSE) s.closed = true;
  else if (s.closed && open > BLINK_OPEN) { s.closed = false; s.blinks++; }

  if (now - s.lastDom < 90) return;      // update the text about 11 times a second
  s.lastDom = now;

  const horiz = s.gx > GAZE_X_THRESHOLD ? "right" : s.gx < -GAZE_X_THRESHOLD ? "left" : "";
  const vert = s.gy > GAZE_Y_THRESHOLD ? "down" : s.gy < -GAZE_Y_THRESHOLD ? "up" : "";
  $("eye-gaze").textContent = [vert, horiz].filter(Boolean).join(" ") || "center";
  $("eye-blinks").textContent = s.blinks;
  $("eye-open").textContent = `${Math.min(100, Math.round((s.open / 0.3) * 100))}%`;
  const dx = Math.max(-1, Math.min(1, s.gx / 0.15)) * 34;
  const dy = Math.max(-1, Math.min(1, s.gy / 0.2)) * 34;
  $("gaze-dot").style.transform = `translate(${dx}px, ${dy}px)`;
}

// --- the little 3D viewer: the same 478 points, rotated in 3D and drawn with depth shading ---
viewer.onpointerdown = (e) => { dragging = true; dragX = e.clientX; dragY = e.clientY; viewer.setPointerCapture(e.pointerId); };
viewer.onpointerup = () => { dragging = false; };
viewer.onpointermove = (e) => {
  if (!dragging) return;
  viewYaw += (e.clientX - dragX) * 0.01;
  viewPitch += (e.clientY - dragY) * 0.01;
  dragX = e.clientX; dragY = e.clientY;
};
viewer.ondblclick = () => { viewYaw = 0; viewPitch = 0; };

function draw3DViewer(now) {
  const W = viewer.width, H = viewer.height;
  vctx.clearRect(0, 0, W, H);

  if (!faceLM) {
    vctx.fillStyle = "rgba(125, 146, 163, 0.9)";
    vctx.font = "500 24px 'JetBrains Mono', monospace";
    vctx.textAlign = "center";
    vctx.fillText(faceLandmarker ? "waiting for a face..." : "loading mesh model...", W / 2, H / 2);
    return;
  }

  const vw = video.videoWidth, vh = video.videoHeight;
  const n = faceLM.length;
  let mx = 0, my = 0, mz = 0;
  for (const p of faceLM) { mx += p.x; my += p.y; mz += p.z; }
  mx /= n; my /= n; mz /= n;

  // gentle automatic sway + whatever you dragged
  const yaw = viewYaw + Math.sin(now / 1800) * 0.55;
  const pitch = viewPitch;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);

  const pts = new Array(n);
  let radius = 1;
  for (let i = 0; i < n; i++) {
    const p = faceLM[i];
    const X = (p.x - mx) * vw, Y = (p.y - my) * vh, Z = (p.z - mz) * vw;   // z uses the same scale as x
    const x1 = X * cy + Z * sy, z1 = -X * sy + Z * cy;                     // turn left/right
    const y2 = Y * cp - z1 * sp, z2 = Y * sp + z1 * cp;                    // tilt up/down
    pts[i] = { x: x1, y: y2, z: z2 };
    radius = Math.max(radius, Math.hypot(x1, y2));
  }
  const scale = (W * 0.46) / radius;
  const project = (i) => ({ x: W / 2 - pts[i].x * scale, y: H / 2 + pts[i].y * scale });  // x flipped to match the mirrored video

  vctx.save();
  vctx.lineCap = "round";
  drawDepthLines(
    FaceLandmarker.FACE_LANDMARKS_TESSELATION,
    pts.map((p) => p.z),
    project,
    vctx
  );

  vctx.strokeStyle = "rgba(226, 232, 255, 0.95)";
  vctx.lineWidth = 2;
  vctx.shadowColor = ACCENT;
  vctx.shadowBlur = 10;
  for (const set of [FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, FaceLandmarker.FACE_LANDMARKS_LIPS,
                     FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE]) {
    vctx.beginPath();
    for (const { start, end } of set) {
      const a = project(start), b = project(end);
      vctx.moveTo(a.x, a.y);
      vctx.lineTo(b.x, b.y);
    }
    vctx.stroke();
  }
  vctx.restore();
}

startBtn.onclick = start;