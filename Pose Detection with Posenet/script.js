// ── DOM refs ──────────────────────────────────────────────────────────────────
const video          = document.getElementById('webcam');
const canvas         = document.getElementById('overlay');
const ctx            = canvas.getContext('2d');
const fpsValueEl     = document.getElementById('fps-value');
const loadingOverlay = document.getElementById('loading-overlay');
const repCountEl     = document.getElementById('rep-count');
const phaseBadgeEl   = document.getElementById('phase-badge');
const kneeAngleEl    = document.getElementById('knee-angle');
const keypointListEl = document.getElementById('keypoint-list');
const modeStatusEl   = document.getElementById('mode-status');

// ── PoseNet skeleton connections (adjacent keypoint pairs) ───────────────────
// PoseNet keypoint indices:
// 0:nose  1:leftEye  2:rightEye  3:leftEar  4:rightEar
// 5:leftShoulder  6:rightShoulder  7:leftElbow  8:rightElbow
// 9:leftWrist  10:rightWrist  11:leftHip  12:rightHip
// 13:leftKnee  14:rightKnee  15:leftAnkle  16:rightAnkle

const SKELETON_PAIRS = [
    [0, 1], [0, 2], [1, 3], [2, 4],          // face
    [5, 6],                                    // shoulders
    [5, 7], [7, 9],                            // left arm
    [6, 8], [8, 10],                           // right arm
    [5, 11], [6, 12],                          // torso sides
    [11, 12],                                  // hips
    [11, 13], [13, 15],                        // left leg
    [12, 14], [14, 16],                        // right leg
];

// Named keypoints for sidebar display (subset of the 17)
const DISPLAY_KEYPOINTS = [
    'nose', 'leftShoulder', 'rightShoulder',
    'leftElbow', 'rightElbow',
    'leftHip', 'rightHip',
    'leftKnee', 'rightKnee',
    'leftAnkle', 'rightAnkle',
];

// ── App state ─────────────────────────────────────────────────────────────────
let net;
let detectionMode = 'single';   // 'single' | 'multi'
let isVideoReady  = false;

// FPS
let frameCount  = 0;
let lastFpsTime = performance.now();

// Squat counter state
let repCount    = 0;
let squatPhase  = 'up';          // 'up' | 'down'
const SQUAT_DOWN_ANGLE  = 120;   // knee angle threshold to enter "down" phase
const SQUAT_UP_ANGLE    = 160;   // knee angle threshold to return to "up" phase

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
    try {
        console.log('Loading PoseNet…');
        net = await posenet.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            inputResolution: { width: 640, height: 480 },
            multiplier: 0.75,
        });
        console.log('PoseNet loaded.');

        loadingOverlay.style.display = 'none';
        await setupWebcam();
        isVideoReady = true;
        predictFrame();

    } catch (err) {
        console.error('Init error:', err);
        loadingOverlay.innerHTML = `
            <p style="color:#f75c5c;font-family:'DM Sans',sans-serif;padding:20px;text-align:center;">
                ⚠ ${err.message}<br><small>Allow webcam access &amp; reload.</small>
            </p>`;
    }
}

// ── Webcam ────────────────────────────────────────────────────────────────────
async function setupWebcam() {
    return new Promise((resolve, reject) => {
        if (!navigator.mediaDevices?.getUserMedia) {
            reject(new Error('Browser does not support webcam access.'));
            return;
        }

        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false })
            .then(stream => {
                video.srcObject = stream;
                video.addEventListener('loadeddata', () => {
                    canvas.width  = video.videoWidth;
                    canvas.height = video.videoHeight;
                    resolve();
                });
            })
            .catch(reject);
    });
}

// ── Mode toggle (called from HTML buttons) ────────────────────────────────────
function setMode(mode) {
    detectionMode = mode;

    document.getElementById('btn-single').classList.toggle('active', mode === 'single');
    document.getElementById('btn-multi').classList.toggle('active',  mode === 'multi');

    modeStatusEl.textContent = mode === 'single'
        ? 'Single-pose · higher accuracy'
        : 'Multi-pose · detects multiple people';
}

// ── Main render loop ──────────────────────────────────────────────────────────
async function predictFrame() {
    if (!isVideoReady) return;

    // 1. Run inference based on current mode
    let poses = [];

    if (detectionMode === 'single') {
        const pose = await net.estimateSinglePose(video, { flipHorizontal: false });
        poses = [pose];
    } else {
        poses = await net.estimateMultiplePoses(video, {
            flipHorizontal: false,
            maxDetections: 5,
            scoreThreshold: 0.5,
            nmsRadius: 20,
        });
    }

    // 2. Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 3. Draw all detected poses
    poses.forEach((pose, i) => {
        const hue = (i * 72) % 360; // different tint per person
        drawPose(pose, hue);
    });

    // 4. Sidebar updates (use first/primary pose only)
    if (poses.length > 0 && poses[0].score > 0.2) {
        updateSquatCounter(poses[0].keypoints);
        updateKeypointSidebar(poses[0].keypoints);
    }

    // 5. FPS
    updateFPS();

    requestAnimationFrame(predictFrame);
}

// ── Draw a single pose (keypoints + skeleton) ─────────────────────────────────
function drawPose(pose, hue = 210) {
    const minConfidence = 0.4;
    const kps = pose.keypoints;

    // Color scheme: accent blue for default, hue-shifted for multi-pose
    const pointColor  = hue === 210 ? '#4f8ef7' : `hsl(${hue}, 90%, 65%)`;
    const boneColor   = hue === 210 ? 'rgba(79,142,247,0.55)' : `hsla(${hue}, 80%, 60%, 0.55)`;
    const jointColor  = '#22d3a5';

    // Draw skeleton bones
    SKELETON_PAIRS.forEach(([i, j]) => {
        const kpA = kps[i];
        const kpB = kps[j];

        if (kpA.score < minConfidence || kpB.score < minConfidence) return;

        ctx.beginPath();
        ctx.moveTo(kpA.position.x, kpA.position.y);
        ctx.lineTo(kpB.position.x, kpB.position.y);
        ctx.strokeStyle = boneColor;
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.stroke();
    });

    // Draw keypoint circles
    kps.forEach(kp => {
        if (kp.score < minConfidence) return;

        const { x, y } = kp.position;

        // Outer glow ring
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(34,211,165,0.18)`;
        ctx.fill();

        // Inner filled dot
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = jointColor;
        ctx.fill();
    });

    // Draw overall pose confidence score tag
    if (pose.score > 0.3) {
        const headKp = kps[0]; // nose
        if (headKp.score > minConfidence) {
            const label = `${(pose.score * 100).toFixed(0)}%`;
            ctx.font = '700 12px "Space Mono", monospace';
            ctx.textBaseline = 'bottom';
            const tw = ctx.measureText(label).width;

            ctx.fillStyle = pointColor;
            roundRect(ctx, headKp.position.x - tw / 2 - 8, headKp.position.y - 30, tw + 16, 20, 5);
            ctx.fill();

            ctx.fillStyle = '#0d0f14';
            ctx.fillText(label, headKp.position.x - tw / 2, headKp.position.y - 13);
        }
    }
}

// ── Squat counter ─────────────────────────────────────────────────────────────
function updateSquatCounter(keypoints) {
    // Use left knee (idx 13) angle: hip(11) → knee(13) → ankle(15)
    const hip   = keypoints[11];
    const knee  = keypoints[13];
    const ankle = keypoints[15];

    if (hip.score < 0.4 || knee.score < 0.4 || ankle.score < 0.4) {
        kneeAngleEl.textContent = '—°';
        return;
    }

    const angle = calculateAngle(hip.position, knee.position, ankle.position);
    kneeAngleEl.textContent = `${Math.round(angle)}°`;

    // State machine: detect down → up transition = 1 rep
    if (angle < SQUAT_DOWN_ANGLE && squatPhase === 'up') {
        squatPhase = 'down';
        updatePhase('down');
    } else if (angle > SQUAT_UP_ANGLE && squatPhase === 'down') {
        squatPhase = 'up';
        repCount++;
        repCountEl.textContent = repCount;
        updatePhase('up');

        // Bump animation
        repCountEl.classList.add('bump');
        setTimeout(() => repCountEl.classList.remove('bump'), 200);
    }
}

// Returns angle at point B formed by A-B-C (in degrees)
function calculateAngle(A, B, C) {
    const radians = Math.atan2(C.y - B.y, C.x - B.x) - Math.atan2(A.y - B.y, A.x - B.x);
    let angle = Math.abs(radians * (180 / Math.PI));
    if (angle > 180) angle = 360 - angle;
    return angle;
}

function updatePhase(phase) {
    phaseBadgeEl.textContent = phase.toUpperCase();
    phaseBadgeEl.className   = `phase-badge phase-${phase}`;
}

// ── Reset counter (called from HTML button) ───────────────────────────────────
function resetCounter() {
    repCount   = 0;
    squatPhase = 'up';
    repCountEl.textContent   = '0';
    updatePhase('up');
    kneeAngleEl.textContent  = '—°';
}

// ── Keypoint sidebar ──────────────────────────────────────────────────────────
function updateKeypointSidebar(keypoints) {
    const kpMap = {};
    keypoints.forEach(kp => { kpMap[kp.part] = kp.score; });

    keypointListEl.innerHTML = DISPLAY_KEYPOINTS.map(name => {
        const score = kpMap[name] ?? 0;
        const pct   = (score * 100).toFixed(0);

        // Color: green > 70%, yellow 40–70%, red <40%
        const color = score > 0.7 ? '#22d3a5' : score > 0.4 ? '#f7c948' : '#f75c5c';

        // Format label: camelCase → spaced
        const label = name.replace(/([A-Z])/g, ' $1').trim().toLowerCase();

        return `
        <div class="kp-row">
            <span class="kp-name">${label}</span>
            <div class="kp-bar-wrap">
                <div class="kp-bar-fill" style="width:${pct}%; background:${color};"></div>
            </div>
            <span class="kp-score" style="color:${color};">${pct}%</span>
        </div>`;
    }).join('');
}

// ── FPS ───────────────────────────────────────────────────────────────────────
function updateFPS() {
    frameCount++;
    const now = performance.now();

    if (now - lastFpsTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
        fpsValueEl.textContent = `FPS: ${fps}`;
        frameCount  = 0;
        lastFpsTime = now;
    }
}

// ── Utility: rounded rect path ────────────────────────────────────────────────
function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', init);
