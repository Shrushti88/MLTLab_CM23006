// ── DOM refs ──────────────────────────────────────────────────────────────────
const video             = document.getElementById('webcam');
const canvas            = document.getElementById('overlay');
const ctx               = canvas.getContext('2d');
const predictionsEl     = document.getElementById('predictions');
const fpsValueEl        = document.getElementById('fps-value');
const loadingOverlay    = document.getElementById('loading-overlay');

// ── State ─────────────────────────────────────────────────────────────────────
let model;
let isVideoReady = false;

// FPS tracking
let frameCount   = 0;
let lastFpsTime  = performance.now();
let currentFps   = 0;

// ── Color palette ─────────────────────────────────────────────────────────────
// Matches the dark-tech design tokens
const colorPalette = [
    '#4f8ef7', // accent blue
    '#22d3a5', // green
    '#f7c948', // yellow
    '#f75c5c', // red
    '#c084fc', // purple
    '#fb923c', // orange
    '#38bdf8', // sky
];
const classColors = {};

function getColor(className) {
    if (!classColors[className]) {
        const idx = Object.keys(classColors).length % colorPalette.length;
        classColors[className] = colorPalette[idx];
    }
    return classColors[className];
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
    try {
        console.log('Loading COCO-SSD (MobileNet backbone)…');
        model = await cocoSsd.load();
        console.log('Model ready.');

        loadingOverlay.style.display = 'none';

        await setupWebcam();
        isVideoReady = true;
        predictFrame();

    } catch (err) {
        console.error('Init error:', err);
        loadingOverlay.innerHTML = `
            <p style="color:#f75c5c;font-family:'DM Sans',sans-serif;padding:20px;text-align:center;">
                ⚠ ${err.message}<br><small>Please allow webcam access and reload.</small>
            </p>`;
    }
}

// ── Webcam setup ──────────────────────────────────────────────────────────────
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

// ── Main render loop ──────────────────────────────────────────────────────────
async function predictFrame() {
    if (!isVideoReady) return;

    const t0 = performance.now();

    // 1. Run inference
    const predictions = await model.detect(video);

    // 2. Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 3. Sort by confidence, show top 3 in sidebar
    const sorted = [...predictions].sort((a, b) => b.score - a.score);
    updatePredictionsUI(sorted.slice(0, 3));

    // 4. Draw every detected object
    predictions.forEach(drawBoundingBox);

    // 5. FPS
    updateFPS();

    // 6. Next frame
    requestAnimationFrame(predictFrame);
}

// ── Sidebar predictions UI ────────────────────────────────────────────────────
function updatePredictionsUI(predictions) {
    if (predictions.length === 0) {
        predictionsEl.innerHTML = `
            <div class="empty-state">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="12" cy="12" r="9"/>
                    <path d="M12 8v4M12 16h.01"/>
                </svg>
                <p>No objects detected</p>
            </div>`;
        return;
    }

    predictionsEl.innerHTML = predictions.map((p, i) => {
        const pct   = (p.score * 100).toFixed(1);
        const color = getColor(p.class);

        return `
        <div class="prediction-item">
            <!-- confidence fill bar -->
            <div class="pred-bar" style="width:${pct}%; background:${color};"></div>

            <span class="pred-rank">${String(i + 1).padStart(2, '0')}</span>
            <span class="pred-swatch" style="background:${color};"></span>
            <span class="pred-label">${p.class}</span>
            <span class="pred-score" style="color:${color};">${pct}%</span>
        </div>`;
    }).join('');
}

// ── Bounding box drawing ──────────────────────────────────────────────────────
function drawBoundingBox(prediction) {
    const [x, y, w, h] = prediction.bbox;
    const color        = getColor(prediction.class);
    const pct          = (prediction.score * 100).toFixed(1);
    const label        = `${prediction.class}  ${pct}%`;

    // --- Box ---
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.stroke();

    // Corner accents (makes it feel more "detection-UI")
    const corner = 12;
    ctx.lineWidth = 3;
    ctx.beginPath();
    // top-left
    ctx.moveTo(x, y + corner); ctx.lineTo(x, y); ctx.lineTo(x + corner, y);
    // top-right
    ctx.moveTo(x + w - corner, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + corner);
    // bottom-left
    ctx.moveTo(x, y + h - corner); ctx.lineTo(x, y + h); ctx.lineTo(x + corner, y + h);
    // bottom-right
    ctx.moveTo(x + w - corner, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - corner);
    ctx.stroke();

    // --- Label pill ---
    ctx.font         = '600 12px "DM Sans", sans-serif';
    ctx.textBaseline = 'top';
    const textW      = ctx.measureText(label).width;
    const pillH      = 22;
    const pillPad    = 8;
    const pillX      = x;
    const pillY      = y > pillH + 4 ? y - pillH - 4 : y + 4;

    // Background pill
    ctx.fillStyle    = color;
    roundRect(ctx, pillX, pillY, textW + pillPad * 2, pillH, 5);
    ctx.fill();

    // Label text
    ctx.fillStyle = '#0d0f14';
    ctx.fillText(label, pillX + pillPad, pillY + 5);
}

// Helper: rounded rectangle path
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

// ── FPS counter ───────────────────────────────────────────────────────────────
function updateFPS() {
    frameCount++;
    const now = performance.now();

    if (now - lastFpsTime >= 1000) {
        currentFps      = Math.round((frameCount * 1000) / (now - lastFpsTime));
        fpsValueEl.textContent = `FPS: ${currentFps}`;
        frameCount      = 0;
        lastFpsTime     = now;
    }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', init);
