// ─────────────────────────────────────────────────────────────────────────────
//  Deploying a Model in Browser — TensorFlow.js
//  XOR problem: 4 samples, 2 inputs → 1 binary output
//  Covers: train → save LocalStorage → reload → verify → export files → import
// ─────────────────────────────────────────────────────────────────────────────

// ── XOR dataset ──────────────────────────────────────────────────────────────
const XOR_INPUTS  = [[0,0],[0,1],[1,0],[1,1]];
const XOR_LABELS  = [0, 1, 1, 0];            // expected outputs

// xs: shape [4,2]   ys: shape [4,1]
const xs = tf.tensor2d(XOR_INPUTS);
const ys = tf.tensor2d(XOR_LABELS, [4, 1]);

// ── State ─────────────────────────────────────────────────────────────────────
let originalModel  = null;   // freshly trained model
let originalPreds  = [];     // predictions from original model (floats)
let lossHistory    = [];     // for chart
const LS_KEY       = 'tfjs-xor-model';

// ── Pipeline step names ───────────────────────────────────────────────────────
const PIPE_STEPS = ['pipe-train','pipe-save','pipe-reload','pipe-predict','pipe-export','pipe-import'];

// ─────────────────────────────────────────────────────────────────────────────
//  UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

function ts() {
    const d = new Date();
    return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
}

function log(boxId, msg, type = 'info') {
    const box = document.getElementById(boxId);
    box.style.display = 'block';
    box.innerHTML += `<div class="log-line"><span class="log-time">[${ts()}]</span><span class="log-${type}">${msg}</span></div>`;
    box.scrollTop = box.scrollHeight;
}

function clearLog(boxId) {
    const box = document.getElementById(boxId);
    box.innerHTML = '';
    box.style.display = 'none';
}

function setPipeStep(index, state) {
    // state: 'active' | 'done' | ''
    const el = document.getElementById(PIPE_STEPS[index]);
    if (!el) return;
    el.className = 'pipe-step' + (state ? ' ' + state : '');
}

function setBadge(id, text, cls) {
    const el = document.getElementById(id);
    el.textContent = text;
    el.className   = `storage-badge ${cls}`;
}

function setBtn(id, disabled) {
    const el = document.getElementById(id);
    if (el) el.disabled = disabled;
}

// ─────────────────────────────────────────────────────────────────────────────
//  LAB 1 — BUILD + TRAIN + SAVE TO LOCALSTORAGE
// ─────────────────────────────────────────────────────────────────────────────

function buildModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 8,  activation: 'relu',    inputShape: [2] }));
    model.add(tf.layers.dense({ units: 8,  activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1,  activation: 'sigmoid' }));
    return model;
}

async function trainAndSave() {
    clearLog('train-log');
    setBtn('btn-train', true);
    lossHistory = [];

    const epochs = parseInt(document.getElementById('epochs').value) || 300;
    const lr     = parseFloat(document.getElementById('lr').value)    || 0.1;

    log('train-log', `Building model…`, 'info');
    originalModel = buildModel();
    originalModel.compile({
        optimizer: tf.train.adam(lr),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    // Show model summary in sidebar
    renderModelSummary(originalModel);

    // Progress elements
    const progressWrap = document.getElementById('progress-wrap');
    const progressBar  = document.getElementById('progress-bar');
    const progressLbl  = document.getElementById('progress-label');
    progressWrap.style.display = 'flex';

    setPipeStep(0, 'active');
    log('train-log', `Training for ${epochs} epochs, lr=${lr}…`, 'info');

    const logFreq = Math.max(1, Math.floor(epochs / 10));

    await originalModel.fit(xs, ys, {
        epochs,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                lossHistory.push(logs.loss);
                const pct = Math.round(((epoch + 1) / epochs) * 100);
                progressBar.style.width = pct + '%';
                progressLbl.textContent = `Epoch ${epoch + 1} / ${epochs}`;

                if ((epoch + 1) % logFreq === 0 || epoch === epochs - 1) {
                    log('train-log',
                        `Epoch ${String(epoch+1).padStart(4)} — loss: ${logs.loss.toFixed(5)}  acc: ${(logs.acc ?? logs.accuracy ?? 0).toFixed(3)}`,
                        'ok');
                }
            },
        },
    });

    setPipeStep(0, 'done');
    log('train-log', 'Training complete ✓', 'ok');

    // Draw loss chart
    drawLossChart(lossHistory);

    // Store original predictions
    originalPreds = await runPredictions(originalModel);
    log('train-log', 'Original predictions captured ✓', 'ok');

    // Save to LocalStorage
    setPipeStep(1, 'active');
    log('train-log', `Saving to LocalStorage as "${LS_KEY}"…`, 'info');

    await originalModel.save(`localstorage://${LS_KEY}`);
    setPipeStep(1, 'done');
    log('train-log', 'Model saved to LocalStorage ✓', 'ok');
    setBadge('ls-badge', 'Saved', 'badge-ok');

    setBtn('btn-train', false);
    setBtn('btn-reload', false);
    setBtn('btn-export', false);
}

// ─────────────────────────────────────────────────────────────────────────────
//  LAB 2 — RELOAD FROM LOCALSTORAGE + VERIFY
// ─────────────────────────────────────────────────────────────────────────────

async function reloadAndVerify() {
    clearLog('reload-log');
    setBtn('btn-reload', true);
    document.getElementById('compare-table-wrap').style.display = 'none';

    if (originalPreds.length === 0) {
        log('reload-log', 'No original model found. Train first.', 'err');
        setBtn('btn-reload', false);
        return;
    }

    setPipeStep(2, 'active');
    log('reload-log', `Loading model from LocalStorage ("${LS_KEY}")…`, 'info');

    let reloadedModel;
    try {
        reloadedModel = await tf.loadLayersModel(`localstorage://${LS_KEY}`);
        setPipeStep(2, 'done');
        log('reload-log', 'Model loaded ✓', 'ok');
    } catch (e) {
        log('reload-log', `Load failed: ${e.message}`, 'err');
        setBtn('btn-reload', false);
        return;
    }

    reloadedModel.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    setPipeStep(3, 'active');
    const reloadedPreds = await runPredictions(reloadedModel);
    setPipeStep(3, 'done');
    log('reload-log', 'Predictions from reloaded model computed ✓', 'ok');

    // Build comparison table
    const tbody = document.getElementById('compare-tbody');
    tbody.innerHTML = '';
    let allMatch = true;

    XOR_INPUTS.forEach((input, i) => {
        const orig    = originalPreds[i];
        const rel     = reloadedPreds[i];
        const delta   = Math.abs(orig - rel);
        const match   = delta < 1e-4;
        if (!match) allMatch = false;

        tbody.innerHTML += `
            <tr>
                <td>[${input}]</td>
                <td>${orig.toFixed(6)}</td>
                <td>${rel.toFixed(6)}</td>
                <td class="${match ? 'match-yes' : 'match-no'}">${match ? '✓ Yes' : '✗ No'}</td>
            </tr>`;
    });

    document.getElementById('compare-table-wrap').style.display = 'block';
    log('reload-log', allMatch ? 'All predictions match ✓' : 'Some predictions differ!', allMatch ? 'ok' : 'warn');

    setBtn('btn-reload', false);
}

// ─────────────────────────────────────────────────────────────────────────────
//  LAB 3a — EXPORT MODEL TO FILES
// ─────────────────────────────────────────────────────────────────────────────

async function exportModel() {
    clearLog('export-log');

    if (!originalModel) {
        log('export-log', 'No trained model found. Train first.', 'err');
        return;
    }

    log('export-log', 'Exporting model to downloadable files…', 'info');
    setPipeStep(4, 'active');

    try {
        await originalModel.save('downloads://xor-model');
        setPipeStep(4, 'done');
        log('export-log', 'Files downloaded: xor-model.json + xor-model.weights.bin ✓', 'ok');
        setBadge('exp-badge', 'Exported', 'badge-ok');
    } catch (e) {
        log('export-log', `Export failed: ${e.message}`, 'err');
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  LAB 3b — IMPORT MODEL FROM FILES
// ─────────────────────────────────────────────────────────────────────────────

async function importModel(event) {
    const files = Array.from(event.target.files);
    if (files.length < 2) {
        log('export-log', 'Please select both the .json and .bin files.', 'warn');
        document.getElementById('export-log').style.display = 'block';
        return;
    }

    clearLog('export-log');
    document.getElementById('import-table-wrap').style.display = 'none';

    log('export-log', `Importing ${files.map(f => f.name).join(', ')}…`, 'info');
    setPipeStep(5, 'active');

    // Sort: json first, then bin
    const jsonFile = files.find(f => f.name.endsWith('.json'));
    const binFiles = files.filter(f => f.name.endsWith('.bin'));

    if (!jsonFile || binFiles.length === 0) {
        log('export-log', 'Need exactly one .json and at least one .bin file.', 'err');
        return;
    }

    try {
        const importedModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, ...binFiles]));
        importedModel.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

        setPipeStep(5, 'done');
        log('export-log', 'Model imported from files ✓', 'ok');
        setBadge('imp-badge', 'Loaded', 'badge-ok');

        // Run predictions on imported model
        const importedPreds = await runPredictions(importedModel);
        log('export-log', 'Predictions from imported model computed ✓', 'ok');

        // Build result table
        const tbody = document.getElementById('import-tbody');
        tbody.innerHTML = '';

        XOR_INPUTS.forEach((input, i) => {
            const expected = XOR_LABELS[i];
            const pred     = importedPreds[i];
            const err      = Math.abs(pred - expected).toFixed(5);
            const good     = parseFloat(err) < 0.1;

            tbody.innerHTML += `
                <tr>
                    <td>[${input}]</td>
                    <td>${expected}</td>
                    <td>${pred.toFixed(6)}</td>
                    <td class="${good ? 'match-yes' : 'match-no'}">${err}</td>
                </tr>`;
        });

        document.getElementById('import-table-wrap').style.display = 'block';
        log('export-log', 'Import + prediction test complete ✓', 'ok');

    } catch (e) {
        log('export-log', `Import failed: ${e.message}`, 'err');
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────────────────────────────────────

// Run all 4 XOR inputs through a model and return array of floats
async function runPredictions(model) {
    const predTensor = model.predict(xs);
    const data       = await predTensor.data();
    predTensor.dispose();
    return Array.from(data);
}

// Render model architecture in sidebar
function renderModelSummary(model) {
    const container = document.getElementById('model-summary');
    container.innerHTML = '';

    let totalParams = 0;

    model.layers.forEach((layer, i) => {
        const paramCount = layer.countParams();
        totalParams += paramCount;

        const type = layer.getClassName();
        const cfg  = layer.getConfig();
        const info = cfg.units
            ? `${cfg.units} units · ${cfg.activation}`
            : `input: [${layer.batchInputShape?.slice(1).join(',')}]`;

        container.innerHTML += `
            <div class="summary-layer">
                <span class="layer-type">${type}</span>
                <span class="layer-info">${info}</span>
                <span class="layer-params">${paramCount}p</span>
            </div>`;
    });

    container.innerHTML += `
        <div class="summary-total">
            <span>Total Parameters</span>
            <span>${totalParams.toLocaleString()}</span>
        </div>`;
}

// Draw mini loss curve onto canvas
function drawLossChart(history) {
    const canvas = document.getElementById('loss-chart');
    const ctx    = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    document.getElementById('chart-empty').style.display = 'none';
    ctx.clearRect(0, 0, W, H);

    const pad   = { top: 16, right: 16, bottom: 28, left: 42 };
    const cw    = W - pad.left - pad.right;
    const ch    = H - pad.top  - pad.bottom;

    const minL  = Math.min(...history);
    const maxL  = Math.max(...history);
    const range = maxL - minL || 1;

    // Axes
    ctx.strokeStyle = '#252a38';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + ch);
    ctx.lineTo(pad.left + cw, pad.top + ch);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle    = '#5a6380';
    ctx.font         = '9px "Space Mono", monospace';
    ctx.textAlign    = 'right';
    ctx.textBaseline = 'middle';
    [0, 0.5, 1].forEach(t => {
        const val = minL + t * range;
        const y   = pad.top + ch - t * ch;
        ctx.fillText(val.toFixed(3), pad.left - 4, y);
        ctx.strokeStyle = '#1a1e2a';
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + cw, y);
        ctx.stroke();
    });

    // X-axis label
    ctx.fillStyle = '#5a6380';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('epochs', pad.left + cw / 2, H - 2);

    // Loss curve with glow
    ctx.strokeStyle = '#4f8ef7';
    ctx.lineWidth   = 2;
    ctx.shadowColor = 'rgba(79,142,247,0.5)';
    ctx.shadowBlur  = 6;
    ctx.beginPath();

    history.forEach((loss, i) => {
        const x = pad.left + (i / (history.length - 1)) * cw;
        const y = pad.top  + ch - ((loss - minL) / range) * ch;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });

    ctx.stroke();
    ctx.shadowBlur = 0;

    // Final loss dot
    const lastX = pad.left + cw;
    const lastY = pad.top + ch - ((history[history.length-1] - minL) / range) * ch;
    ctx.fillStyle = '#22d3a5';
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4, 0, 2*Math.PI);
    ctx.fill();
}

// ─────────────────────────────────────────────────────────────────────────────
//  INIT — disable reload/export until training is done
// ─────────────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
    setBtn('btn-reload', true);
    setBtn('btn-export', true);
});
