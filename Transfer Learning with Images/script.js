// ─────────────────────────────────────────────────────────────────────────────
//  Transfer Learning with MobileNet — TensorFlow.js
//  MobileNet (frozen) → dense head (trainable) → custom categories
// ─────────────────────────────────────────────────────────────────────────────

// ── DOM ───────────────────────────────────────────────────────────────────────
const videoEl       = document.getElementById('webcam');
const canvasEl      = document.getElementById('overlay');
const loadingOverlay= document.getElementById('loading-overlay');

// ── Palette — up to 8 categories ─────────────────────────────────────────────
const CAT_COLORS = ['#4f8ef7','#22d3a5','#f7c948','#f75c5c','#c084fc','#fb923c','#38bdf8','#a3e635'];

// ── App state ─────────────────────────────────────────────────────────────────
let mobileNet       = null;   // frozen feature extractor
let headModel       = null;   // trainable dense head
let isVideoReady    = false;
let isPredicting    = false;

// Categories: [{name, color, samples:[Float32Array(1024)]}]
let categories      = [];
let activeCatIdx    = 0;

// For incremental training log
let perfHistory     = [];
let lastValAcc      = null;

// Loss / acc histories for charts
let lossHistory     = [];
let valAccHistory   = [];

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', async () => {
    disableAll();
    setPipe('pipe-load', 'active');

    try {
        // Start webcam early (parallel to model load)
        setupWebcam();

        log('train-log', 'Loading MobileNet v1 (frozen)…', 'info');
        mobileNet = await mobilenet.load({ version: 1, alpha: 1.0 });
        setPipe('pipe-load', 'done');
        log('train-log', 'MobileNet loaded ✓', 'ok');

        loadingOverlay.style.display = 'none';

        // Default 3 fruit categories
        ['Apple 🍎', 'Banana 🍌', 'Orange 🍊'].forEach(n => addCategoryNamed(n));
        renderCategoryTabs();
        updateStats();

        setBtn('btn-capture', false);
        setPipe('pipe-collect', 'active');

    } catch (e) {
        loadingOverlay.innerHTML = `<p style="color:#f75c5c;font-family:'DM Sans',sans-serif;padding:20px;text-align:center;">⚠ ${e.message}</p>`;
    }
});

// ── Webcam ────────────────────────────────────────────────────────────────────
async function setupWebcam() {
    if (!navigator.mediaDevices?.getUserMedia) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video:{ facingMode:'environment' }, audio:false });
        videoEl.srcObject = stream;
        videoEl.addEventListener('loadeddata', () => { isVideoReady = true; });
    } catch(e) {
        console.warn('Webcam unavailable:', e.message);
    }
}

// ── Category management ───────────────────────────────────────────────────────
function addCategoryNamed(name) {
    const idx = categories.length;
    if (idx >= CAT_COLORS.length) { alert('Max 8 categories supported.'); return; }
    categories.push({ name, color: CAT_COLORS[idx], samples: [] });
}

function addCategory() {
    const name = prompt('Category name (e.g. Grape 🍇):');
    if (!name || !name.trim()) return;
    if (categories.length >= CAT_COLORS.length) { alert('Max 8 categories.'); return; }
    addCategoryNamed(name.trim());
    activeCatIdx = categories.length - 1;
    renderCategoryTabs();
    updateStats();
    log('train-log', `Category "${categories[activeCatIdx].name}" added. Capture at least 10 samples.`, 'info');
    document.getElementById('train-log').style.display = 'block';
}

function renderCategoryTabs() {
    const container = document.getElementById('category-tabs');
    container.innerHTML = categories.map((cat, i) => `
        <div class="cat-tab ${i === activeCatIdx ? 'active' : ''}" onclick="selectCat(${i})">
            <span class="cat-dot" style="background:${cat.color};"></span>
            ${cat.name}
            <span class="cat-count">(${cat.samples.length})</span>
        </div>`).join('');

    // Sample pills
    const pillsEl = document.getElementById('sample-counts');
    pillsEl.innerHTML = categories.map(cat => `
        <div class="sample-pill">
            <span class="pill-dot" style="background:${cat.color};"></span>
            <span>${cat.name}</span>
            <span class="pill-count">${cat.samples.length}</span>
        </div>`).join('');
}

function selectCat(i) {
    activeCatIdx = i;
    renderCategoryTabs();
}

// ── Capture a frame ───────────────────────────────────────────────────────────
async function captureFrame() {
    if (!mobileNet || !isVideoReady) { log('train-log', 'Webcam not ready.', 'warn'); document.getElementById('train-log').style.display='block'; return; }

    // Extract 1024-d embedding from MobileNet's penultimate layer
    const embedding = tf.tidy(() => {
        const img = tf.browser.fromPixels(videoEl);
        const resized = tf.image.resizeBilinear(img, [224, 224]);
        const batched = resized.expandDims(0).toFloat().div(127).sub(1);
        // Use the internal model to get embeddings (layer before final softmax)
        const infer = mobileNet.infer(batched, true); // true = embedding layer
        return infer.squeeze();
    });

    const data = await embedding.data();
    embedding.dispose();

    categories[activeCatIdx].samples.push(new Float32Array(data));
    renderCategoryTabs();
    updateStats();

    // Enable train once every category has ≥2 samples
    const canTrain = categories.every(c => c.samples.length >= 2) && categories.length >= 2;
    setBtn('btn-train', !canTrain);

    // Flash canvas feedback
    flashCaptureFeedback(categories[activeCatIdx].color);
}

function flashCaptureFeedback(color) {
    const ctx = canvasEl.getContext('2d');
    canvasEl.width  = videoEl.videoWidth  || 320;
    canvasEl.height = videoEl.videoHeight || 240;
    ctx.strokeStyle = color;
    ctx.lineWidth = 6;
    ctx.strokeRect(4, 4, canvasEl.width - 8, canvasEl.height - 8);
    setTimeout(() => ctx.clearRect(0, 0, canvasEl.width, canvasEl.height), 300);
}

// ── Train ─────────────────────────────────────────────────────────────────────
async function trainModel() {
    clearLog('train-log');
    setBtn('btn-train', true);
    setBtn('btn-capture', true);
    lossHistory = []; valAccHistory = [];

    const epochs    = parseInt(document.getElementById('epochs').value)    || 30;
    const lr        = parseFloat(document.getElementById('lr').value)      || 0.001;
    const batchSize = parseInt(document.getElementById('batchsize').value) || 16;
    const hUnits    = parseInt(document.getElementById('hunits').value)    || 64;

    const numClasses = categories.length;

    setPipe('pipe-train', 'active');
    log('train-log', `Building head: Dense(${hUnits}) → Dropout(0.3) → Dense(${numClasses}, softmax)`, 'info');

    // ── Build a small trainable head ──────────────────────────────────────────
    if (headModel) { headModel.dispose(); headModel = null; }

    headModel = tf.sequential();
    headModel.add(tf.layers.dense({ units: hUnits, activation: 'relu', inputShape: [1024] }));
    headModel.add(tf.layers.dropout({ rate: 0.3 }));
    headModel.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
    headModel.compile({
        optimizer: tf.train.adam(lr),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    // Update sidebar
    document.getElementById('arch-dim').textContent = `→ ${numClasses} classes`;
    document.getElementById('arch-head').querySelector('.arch-sub').textContent =
        `Dense(${hUnits}) · Dropout · Softmax`;

    // ── Build tensors from captured embeddings ────────────────────────────────
    log('train-log', 'Assembling dataset from embeddings…', 'info');

    // Shuffle and split 80/20
    let allX = [], allY = [];
    categories.forEach((cat, ci) => {
        cat.samples.forEach(emb => {
            allX.push(Array.from(emb));
            const label = new Array(numClasses).fill(0);
            label[ci] = 1;
            allY.push(label);
        });
    });

    // Shuffle together
    const combined = allX.map((x, i) => ({ x, y: allY[i] }));
    shuffle(combined);

    const splitIdx = Math.floor(combined.length * 0.8);
    const trainSet = combined.slice(0, splitIdx);
    const valSet   = combined.slice(splitIdx);

    // Store val set for evaluation (as raw arrays, for confusion matrix)
    window._valSet = valSet.map(d => ({ x: d.x, yTrue: d.y.indexOf(1) }));

    const xTrain = tf.tensor2d(trainSet.map(d => d.x));
    const yTrain = tf.tensor2d(trainSet.map(d => d.y));
    const xVal   = tf.tensor2d(valSet.map(d => d.x));
    const yVal   = tf.tensor2d(valSet.map(d => d.y));

    const totalSamples = allX.length;
    log('train-log', `${trainSet.length} train / ${valSet.length} val · ${epochs} epochs`, 'info');

    // Progress bar
    const progressWrap = document.getElementById('progress-wrap');
    const progressBar  = document.getElementById('progress-bar');
    const progressLbl  = document.getElementById('progress-label');
    progressWrap.style.display = 'flex';

    const logFreq = Math.max(1, Math.floor(epochs / 8));

    await headModel.fit(xTrain, yTrain, {
        epochs,
        batchSize: Math.min(batchSize, trainSet.length),
        validationData: [xVal, yVal],
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                lossHistory.push(logs.loss);
                const va = logs.val_acc ?? logs.val_accuracy ?? 0;
                valAccHistory.push(va);
                lastValAcc = va;

                const pct = Math.round(((epoch+1) / epochs) * 100);
                progressBar.style.width = pct + '%';
                progressLbl.textContent = `Epoch ${epoch+1} / ${epochs}`;

                if ((epoch+1) % logFreq === 0 || epoch === epochs-1) {
                    log('train-log',
                        `Ep ${String(epoch+1).padStart(3)} — loss:${logs.loss.toFixed(4)} acc:${(logs.acc ?? logs.accuracy ?? 0).toFixed(3)} val_acc:${va.toFixed(3)}`,
                        'ok');
                }
            },
        },
    });

    xTrain.dispose(); yTrain.dispose(); xVal.dispose(); yVal.dispose();

    setPipe('pipe-train', 'done');
    setPipe('pipe-collect', 'done');
    log('train-log', `Training complete ✓  final val_acc: ${(lastValAcc*100).toFixed(1)}%`, 'ok');

    // Record in incremental log
    const run = perfHistory.length + 1;
    const delta = perfHistory.length > 0
        ? lastValAcc - perfHistory[perfHistory.length-1].acc
        : null;
    perfHistory.push({ run, cats: numClasses, samples: totalSamples, acc: lastValAcc, delta });
    renderPerfTable();
    document.getElementById('perf-delta-wrap').style.display = 'block';

    setBadge('model-status-badge', 'Trained', 'badge-ok');
    drawLineChart('loss-chart', 'chart-empty', lossHistory, '#4f8ef7', 'Loss');
    drawLineChart('acc-chart',  'acc-chart-empty', valAccHistory, '#22d3a5', 'Val Acc', true);

    setBtn('btn-train',      false);
    setBtn('btn-capture',    false);
    setBtn('btn-eval',       false);
    setBtn('btn-start-pred', false);
    setPipe('pipe-eval', 'active');
}

// ── Evaluate ──────────────────────────────────────────────────────────────────
async function evaluateModel() {
    clearLog('eval-log');
    setBtn('btn-eval', true);
    document.getElementById('accuracy-bars').style.display = 'none';
    document.getElementById('confusion-wrap').style.display = 'none';

    if (!headModel || !window._valSet || window._valSet.length === 0) {
        log('eval-log', 'Train the model first.', 'err');
        setBtn('btn-eval', false);
        return;
    }

    setPipe('pipe-eval', 'active');
    log('eval-log', `Evaluating on ${window._valSet.length} validation samples…`, 'info');

    const numClasses = categories.length;
    // confusion[actual][predicted]
    const confMatrix = Array.from({length:numClasses}, () => new Array(numClasses).fill(0));
    const classCorrect = new Array(numClasses).fill(0);
    const classTotal   = new Array(numClasses).fill(0);

    for (const sample of window._valSet) {
        const input  = tf.tensor2d([sample.x]);
        const predT  = headModel.predict(input);
        const predArr = await predT.data();
        input.dispose(); predT.dispose();

        const predicted = predArr.indexOf(Math.max(...predArr));
        const actual    = sample.yTrue;

        confMatrix[actual][predicted]++;
        classTotal[actual]++;
        if (predicted === actual) classCorrect[actual]++;
    }

    setPipe('pipe-eval', 'done');
    log('eval-log', 'Evaluation complete ✓', 'ok');

    // Overall accuracy
    const totalCorrect = classCorrect.reduce((a,b)=>a+b,0);
    const totalSamples = classTotal.reduce((a,b)=>a+b,0);
    const overallAcc   = totalSamples > 0 ? totalCorrect / totalSamples : 0;
    log('eval-log', `Overall val accuracy: ${(overallAcc*100).toFixed(1)}%`, 'ok');

    // Per-class accuracy bars
    const barContainer = document.getElementById('acc-bar-container');
    barContainer.innerHTML = categories.map((cat, i) => {
        const acc = classTotal[i] > 0 ? classCorrect[i] / classTotal[i] : 0;
        const pct = (acc * 100).toFixed(1);
        return `
        <div class="acc-bar-row">
            <span class="acc-bar-label">${cat.name}</span>
            <div class="acc-bar-bg">
                <div class="acc-bar-fill" style="width:${pct}%; background:${cat.color};"></div>
            </div>
            <span class="acc-bar-pct" style="color:${cat.color};">${pct}%</span>
        </div>`;
    }).join('');
    document.getElementById('accuracy-bars').style.display = 'block';

    // Confusion matrix
    renderConfusionMatrix(confMatrix);
    document.getElementById('confusion-wrap').style.display = 'block';

    setBtn('btn-eval', false);
}

function renderConfusionMatrix(matrix) {
    const n = categories.length;
    let html = `<table class="cm-table"><thead><tr><th></th>`;
    categories.forEach(c => { html += `<th title="Predicted: ${c.name}"><span style="color:${c.color}">${truncate(c.name,8)}</span></th>`; });
    html += `</tr></thead><tbody>`;

    matrix.forEach((row, i) => {
        html += `<tr><td class="cm-row-label" style="color:${categories[i].color};">${truncate(categories[i].name,8)}</td>`;
        row.forEach((val, j) => {
            const cls = val === 0 ? 'cm-zero' : (i === j ? 'cm-correct' : 'cm-wrong');
            html += `<td class="${cls}">${val}</td>`;
        });
        html += `</tr>`;
    });

    html += `</tbody></table>`;
    document.getElementById('confusion-matrix').innerHTML = html;
}

// ── Live prediction ───────────────────────────────────────────────────────────
let predLoopId = null;

function startPredicting() {
    if (!headModel || !isVideoReady) { log('pred-log', 'Train model and allow webcam first.', 'warn'); document.getElementById('pred-log').style.display='block'; return; }
    isPredicting = true;
    setBtn('btn-start-pred', true);
    setBtn('btn-stop-pred',  false);
    setPipe('pipe-predict', 'active');
    log('pred-log', 'Live prediction started ✓', 'ok');
    document.getElementById('pred-log').style.display = 'block';
    predLoop();
}

function stopPredicting() {
    isPredicting = false;
    setBtn('btn-start-pred', false);
    setBtn('btn-stop-pred',  true);
    setPipe('pipe-predict', 'done');
    document.getElementById('live-bars').innerHTML = '';
    log('pred-log', 'Prediction stopped.', 'info');
}

async function predLoop() {
    if (!isPredicting) return;

    const embedding = tf.tidy(() => {
        const img    = tf.browser.fromPixels(videoEl);
        const resized= tf.image.resizeBilinear(img, [224, 224]);
        const batched= resized.expandDims(0).toFloat().div(127).sub(1);
        return mobileNet.infer(batched, true).squeeze();
    });

    const data  = await embedding.data();
    embedding.dispose();

    const input  = tf.tensor2d([Array.from(data)]);
    const predT  = headModel.predict(input);
    const scores = Array.from(await predT.data());
    input.dispose(); predT.dispose();

    // Update live bars
    const liveBars = document.getElementById('live-bars');
    liveBars.innerHTML = categories.map((cat, i) => {
        const pct = (scores[i] * 100).toFixed(1);
        return `
        <div class="live-bar-row">
            <span class="live-bar-label">${cat.name}</span>
            <div class="live-bar-bg">
                <div class="live-bar-fill" style="width:${pct}%; background:${cat.color};"></div>
            </div>
            <span class="live-bar-pct" style="color:${cat.color};">${pct}%</span>
        </div>`;
    }).join('');

    // Overlay top prediction on canvas
    const topIdx = scores.indexOf(Math.max(...scores));
    drawPredictionOverlay(categories[topIdx].name, categories[topIdx].color, scores[topIdx]);

    requestAnimationFrame(predLoop);
}

function drawPredictionOverlay(label, color, conf) {
    const ctx = canvasEl.getContext('2d');
    canvasEl.width  = videoEl.videoWidth  || 320;
    canvasEl.height = videoEl.videoHeight || 240;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    const text = `${label}  ${(conf*100).toFixed(0)}%`;
    ctx.font = '700 14px "Space Mono", monospace';
    ctx.textBaseline = 'top';
    const tw = ctx.measureText(text).width;

    // Pill background
    ctx.fillStyle = color;
    roundRect(ctx, 10, 10, tw + 20, 28, 6);
    ctx.fill();

    // Label
    ctx.fillStyle = '#0d0f14';
    ctx.fillText(text, 20, 16);

    // Confidence border
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(2, 2, canvasEl.width-4, canvasEl.height-4);
}

// ── Incremental performance table ─────────────────────────────────────────────
function renderPerfTable() {
    const tbody = document.getElementById('perf-tbody');
    tbody.innerHTML = perfHistory.map(r => {
        let deltaStr = '—';
        let deltaCls = 'delta-neu';
        if (r.delta !== null) {
            const sign = r.delta >= 0 ? '+' : '';
            deltaStr  = `${sign}${(r.delta*100).toFixed(1)}%`;
            deltaCls  = r.delta > 0.01 ? 'delta-pos' : r.delta < -0.01 ? 'delta-neg' : 'delta-neu';
        }
        return `<tr>
            <td>#${r.run}</td>
            <td>${r.cats}</td>
            <td>${r.samples}</td>
            <td class="match-yes">${(r.acc*100).toFixed(1)}%</td>
            <td class="${deltaCls}">${deltaStr}</td>
        </tr>`;
    }).join('');
    setPipe('pipe-expand', 'done');
}

// ── Clear samples ─────────────────────────────────────────────────────────────
function clearSamples() {
    categories.forEach(c => c.samples = []);
    renderCategoryTabs();
    updateStats();
    setBtn('btn-train', true);
    setBadge('model-status-badge', 'Untrained', 'badge-idle');
    log('train-log', 'All samples cleared.', 'warn');
    document.getElementById('train-log').style.display = 'block';
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats() {
    document.getElementById('stat-cats').textContent  = categories.length;
    document.getElementById('stat-total').textContent = categories.reduce((a,c)=>a+c.samples.length,0);
}

// ── Charts ────────────────────────────────────────────────────────────────────
function drawLineChart(canvasId, emptyId, history, lineColor, yLabel, isAcc=false) {
    const canvas = document.getElementById(canvasId);
    const ctx    = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    document.getElementById(emptyId).style.display = 'none';
    ctx.clearRect(0, 0, W, H);

    const pad = { top:14, right:14, bottom:26, left:40 };
    const cw  = W - pad.left - pad.right;
    const ch  = H - pad.top  - pad.bottom;

    const minV = isAcc ? 0  : Math.min(...history);
    const maxV = isAcc ? 1  : Math.max(...history);
    const range= maxV - minV || 1;

    // Grid + axes
    ctx.strokeStyle='#252a38'; ctx.lineWidth=1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top+ch);
    ctx.lineTo(pad.left+cw, pad.top+ch);
    ctx.stroke();

    ctx.fillStyle='#5a6380'; ctx.font='9px "Space Mono"';
    ctx.textAlign='right'; ctx.textBaseline='middle';
    [0,0.5,1].forEach(t => {
        const val = minV + t*range;
        const y   = pad.top + ch - t*ch;
        ctx.fillText(isAcc ? (val*100).toFixed(0)+'%' : val.toFixed(3), pad.left-4, y);
        ctx.strokeStyle='#1a1e2a';
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left+cw, y); ctx.stroke();
    });

    ctx.fillStyle='#5a6380'; ctx.textAlign='center'; ctx.textBaseline='bottom';
    ctx.fillText('epochs', pad.left+cw/2, H-2);

    if (history.length < 2) return;

    // Curve
    ctx.strokeStyle = lineColor;
    ctx.lineWidth   = 2;
    ctx.shadowColor = lineColor.replace(')', ',0.4)').replace('rgb','rgba');
    ctx.shadowBlur  = 6;
    ctx.beginPath();
    history.forEach((v, i) => {
        const x = pad.left + (i/(history.length-1))*cw;
        const y = pad.top  + ch - ((v-minV)/range)*ch;
        i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Final dot
    const lx = pad.left+cw;
    const ly = pad.top+ch - ((history[history.length-1]-minV)/range)*ch;
    ctx.fillStyle = '#22d3a5';
    ctx.beginPath(); ctx.arc(lx, ly, 4, 0, 2*Math.PI); ctx.fill();
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function ts() {
    const d = new Date();
    return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}
function pad2(n) { return String(n).padStart(2,'0'); }

function log(boxId, msg, type='info') {
    const box = document.getElementById(boxId);
    box.style.display = 'block';
    box.innerHTML += `<div class="log-line"><span class="log-time">[${ts()}]</span><span class="log-${type}">${msg}</span></div>`;
    box.scrollTop = box.scrollHeight;
}

function clearLog(id) {
    const el = document.getElementById(id);
    el.innerHTML = ''; el.style.display = 'none';
}

function setBtn(id, disabled) {
    const el = document.getElementById(id);
    if (el) el.disabled = disabled;
}

function disableAll() {
    ['btn-capture','btn-train','btn-eval','btn-start-pred','btn-stop-pred'].forEach(id => setBtn(id, true));
}

function setBadge(id, text, cls) {
    const el = document.getElementById(id);
    el.textContent = text; el.className = `storage-badge ${cls}`;
}

function setPipe(id, state) {
    const el = document.getElementById(id);
    if (el) el.className = 'pipe-step' + (state ? ' '+state : '');
}

function shuffle(arr) {
    for (let i = arr.length-1; i>0; i--) {
        const j = Math.floor(Math.random()*(i+1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
}

function truncate(str, n) { return str.length > n ? str.slice(0,n)+'…' : str; }

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x+r, y); ctx.lineTo(x+w-r, y);
    ctx.quadraticCurveTo(x+w, y, x+w, y+r);
    ctx.lineTo(x+w, y+h-r);
    ctx.quadraticCurveTo(x+w, y+h, x+w-r, y+h);
    ctx.lineTo(x+r, y+h);
    ctx.quadraticCurveTo(x, y+h, x, y+h-r);
    ctx.lineTo(x, y+r);
    ctx.quadraticCurveTo(x, y, x+r, y);
    ctx.closePath();
}
