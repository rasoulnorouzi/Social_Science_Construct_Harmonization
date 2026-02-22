import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

env.allowLocalModels = false;

// ─── Large-dataset thresholds ─────────────────────────────────────────────────
const LABEL_THRESHOLD  = 80;   // hide text labels above this many concepts
const MATRIX_THRESHOLD = 300;  // skip matrix render above this
const WEBGL_THRESHOLD  = 200;  // switch to scattergl above this

// ─── Model registry (mirrors config.py MODELS) ───────────────────────────────
const MODEL_MAP = {
    'Xenova/all-mpnet-base-v2': {
        displayName: 'All-MPNet-Base-v2',
        hint: 'General-purpose sentence embeddings · config: all-mpnet-base-v2'
    },
    'Xenova/mpnet-base': {
        displayName: 'MPNet-Personality',
        hint: 'MPNet architecture (personality constructs) · config: dwulff/mpnet-personality'
    },
    'Xenova/bert-base-uncased': {
        displayName: 'BERT Base Uncased',
        hint: 'Token embeddings, mean pooling · config: bert-base-uncased'
    }
};

// ─── UI refs ─────────────────────────────────────────────────────────────────
const modelSelect   = document.getElementById('model-select');
const modelHint     = document.getElementById('model-hint');
const methodBtns    = document.querySelectorAll('.method-btn');
const pParamsEl     = document.getElementById('params-pairwise');
const hParamsEl     = document.getElementById('params-hdbscan');
const eParamsEl     = document.getElementById('params-elsst');
const conceptsInput = document.getElementById('concepts-input');
const conceptsSect  = document.getElementById('concepts-section');
const conceptCount  = document.getElementById('concept-count');
const processBtn    = document.getElementById('process-btn');
const elsstBtn      = document.getElementById('elsst-btn');
const progressWrap  = document.getElementById('progress-wrap');
const progressFill  = document.getElementById('progress-fill');
const statusEl      = document.getElementById('status');
const statusIdle    = document.getElementById('status-idle');
const exportBtn     = document.getElementById('export-btn');
const tabBtns       = document.querySelectorAll('.tab-btn[data-target]');
const tabContents   = document.querySelectorAll('.tab-content');
const matrixWarn    = document.getElementById('matrix-warn');
const elsstQuery    = document.getElementById('elsst-query');
const elsstCacheSt  = document.getElementById('elsst-cache-status');
const elsstResults  = document.getElementById('elsst-results');
const elsstModelSel = document.getElementById('elsst-model-select');
const l2Toggle      = document.getElementById('l2-toggle');

// ─── Slider sync helper ───────────────────────────────────────────────────────
function syncSlider(inputId, valueId, transform = v => v) {
    const input = document.getElementById(inputId);
    const span  = document.getElementById(valueId);
    if (!input || !span) return;
    span.textContent = transform(input.value);
    input.addEventListener('input', () => { span.textContent = transform(input.value); });
}
syncSlider('threshold-input', 'v-threshold', v => (+v).toFixed(2));
syncSlider('umap-nn',         'v-umap-nn',   v => v);
syncSlider('umap-md',         'v-umap-md',   v => (+v).toFixed(2));
syncSlider('eps-input',       'v-eps',       v => (+v).toFixed(2));
syncSlider('mcs-input',       'v-mcs',       v => v);
syncSlider('ms-input',        'v-ms',        v => v);
syncSlider('topk-input',      'v-topk',      v => v);

document.getElementById('umap-nc').addEventListener('change', e => {
    document.getElementById('v-umap-nc').textContent = (+e.target.value === 0) ? 'None' : e.target.value;
});

// Model hint
modelSelect.addEventListener('change', () => { modelHint.textContent = MODEL_MAP[modelSelect.value]?.hint ?? ''; });
modelHint.textContent = MODEL_MAP[modelSelect.value]?.hint ?? '';

// Concept counter
function updateCount() {
    const n = conceptsInput.value.split('\n').map(l => l.trim()).filter(Boolean).length;
    conceptCount.textContent = n;
}
conceptsInput.addEventListener('input', updateCount);
updateCount();

// Method toggle
let currentMethod = 'pairwise';
methodBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        methodBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentMethod = btn.dataset.method;
        pParamsEl.style.display = currentMethod === 'pairwise' ? '' : 'none';
        hParamsEl.style.display = currentMethod === 'hdbscan'  ? '' : 'none';
        eParamsEl.style.display = currentMethod === 'elsst'    ? '' : 'none';
        // Show/hide concepts textarea vs ELSST query input
        conceptsSect.style.display = currentMethod === 'elsst' ? 'none' : '';
        processBtn.style.display   = currentMethod === 'elsst' ? 'none' : '';
        elsstBtn.style.display     = currentMethod === 'elsst' ? ''     : 'none';
        // Auto-select model for ELSST mode
        if (currentMethod === 'elsst') {
            loadElsstCache();  // preload cache for selected model
        }
    });
});

// Tab switching
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.target).classList.add('active');
    });
});

// ─── Progress helpers ─────────────────────────────────────────────────────────
function setProgress(pct, msg) {
    progressFill.style.width = `${pct}%`;
    statusEl.textContent = msg;
}
function showProgress(msg = '') {
    statusIdle.style.display = 'none';
    progressWrap.style.display = '';
    setProgress(0, msg);
}
function hideProgress(msg = 'Done.') {
    setProgress(100, msg);
    setTimeout(() => {
        progressWrap.style.display = 'none';
        statusIdle.style.display   = '';
        statusIdle.textContent     = msg;
    }, 600);
}

// ─── Math helpers ─────────────────────────────────────────────────────────────
function l2Normalize(vec) {
    let norm = 0;
    for (const v of vec) norm += v * v;
    norm = Math.sqrt(norm);
    return norm === 0 ? vec.slice() : vec.map(v => v / norm);
}

function dotProduct(a, b) {
    let d = 0;
    for (let i = 0; i < a.length; i++) d += a[i] * b[i];
    return d;
}

// Build similarity matrix (Float32Array rows for memory efficiency)
function buildSimMatrix(embeddings, onProgress) {
    const n = embeddings.length;
    const sim = Array.from({ length: n }, () => new Float32Array(n));
    for (let i = 0; i < n; i++) {
        sim[i][i] = 1;
        for (let j = i + 1; j < n; j++) {
            const s = Math.max(-1, Math.min(1, dotProduct(embeddings[i], embeddings[j])));
            sim[i][j] = s;
            sim[j][i] = s;
        }
        if (onProgress && i % 50 === 0) onProgress(i / n);
    }
    return sim;
}

// ─── Seeded PRNG (mulberry32) — matches project SEED = 42 ────────────────────
// UMAP is stochastic by default; without a fixed seed every run gives different
// layouts and therefore different HDBSCAN distances → non-reproducible clusters.
function seededRandom(seed = 42) {
    let s = seed >>> 0;
    return function () {
        s += 0x6D2B79F5;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// ─── UMAP wrapper ─────────────────────────────────────────────────────────────
function runUMAPSync(embeddings, nComponents, nNeighbors, minDist) {
    if (embeddings.length <= 2) {
        return embeddings.map((_, i) => [Math.cos(i * Math.PI), Math.sin(i * Math.PI)]);
    }
    const UMAPClass = typeof UMAP !== 'undefined' ? UMAP : null;
    if (!UMAPClass) throw new Error('UMAP library not loaded');
    const umap = new UMAPClass({
        nComponents,
        nNeighbors: Math.min(nNeighbors, embeddings.length - 1),
        minDist,
        random: seededRandom(42)   // fixed seed → reproducible layout every run
    });
    return umap.fit(embeddings);
}

// ─── Clustering: Pairwise connected-components ────────────────────────────────
function clusterPairwise(concepts, simMatrix, threshold) {
    const n = concepts.length;
    const pairs = [];
    const adj   = Array.from({ length: n }, () => []);

    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const s = simMatrix[i][j];
            if (s >= threshold) {
                pairs.push({ concept1: concepts[i], concept2: concepts[j], similarity: s });
                adj[i].push(j); adj[j].push(i);
            }
        }
    }
    pairs.sort((a, b) => b.similarity - a.similarity);

    const visited = new Array(n).fill(false);
    const clusters = [];
    for (let i = 0; i < n; i++) {
        if (visited[i]) continue;
        const cluster = [], queue = [i];
        visited[i] = true;
        while (queue.length) {
            const c = queue.shift();
            cluster.push(concepts[c]);
            for (const nb of adj[c]) { if (!visited[nb]) { visited[nb] = true; queue.push(nb); } }
        }
        clusters.push(cluster);
    }
    clusters.sort((a, b) => b.length - a.length);
    return { clusters, pairs };
}

// ─── True HDBSCAN ────────────────────────────────────────────────────────────
// Algorithm: mutual reachability distances → Prim MST → single-linkage
//            → condensed tree (minClusterSize) → EOM stability extraction
// Fully independent of pairwise parameters.
// Parameters:
//   minSamples          = core distance neighbour count (sklearn convention)
//   minClusterSize      = smallest acceptable cluster in condensed tree
//   clusterSelectionEps = optional density floor for cluster selection (0 = pure EOM)
//   preEmbeddings       = optional UMAP-reduced coords (Euclidean dist); else cosine dist
function clusterHDBSCAN(concepts, simMatrix, minClusterSize, minSamples, clusterSelectionEps, preEmbeddings) {
    const n = concepts.length;
    if (n < 2) return { clusters: concepts.map(c => [c]), pairs: [] };

    // ── Build pairwise distance matrix ────────────────────────────────────
    const dist = Array.from({ length: n }, (_, i) => {
        const row = new Float32Array(n);
        for (let j = 0; j < n; j++) {
            if (i === j) { row[j] = 0; continue; }
            if (preEmbeddings) {
                let d = 0;
                for (let k = 0; k < preEmbeddings[i].length; k++) {
                    const diff = preEmbeddings[i][k] - preEmbeddings[j][k];
                    d += diff * diff;
                }
                row[j] = Math.sqrt(d);
            } else {
                row[j] = Math.max(0, 1 - simMatrix[i][j]); // cosine distance [0,2]
            }
        }
        return row;
    });

    // ── 1. Core distances (sklearn convention: kth neighbor including self) ─
    // coreDist[i] = distance to the (minSamples-1)-th nearest non-self neighbor
    // i.e. sorted dist row (self=0 at index 0), coreDist = sorted[minSamples-1]
    const coreDist = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const sorted = Array.from(dist[i]).sort((a, b) => a - b);
        coreDist[i] = sorted[Math.min(minSamples - 1, n - 1)];
    }

    // ── 2. Minimum spanning tree on mutual reachability (Prim's O(n²)) ────
    // mrd(u,v) = max(coreDist[u], coreDist[v], dist[u][v])
    const inMST = new Uint8Array(n);
    const key   = new Float64Array(n).fill(Infinity);
    const par   = new Int32Array(n).fill(-1);
    key[0] = 0;
    const mstEdges = [];
    for (let iter = 0; iter < n; iter++) {
        let u = -1;
        for (let i = 0; i < n; i++) if (!inMST[i] && (u < 0 || key[i] < key[u])) u = i;
        inMST[u] = 1;
        if (par[u] >= 0) mstEdges.push({ u: par[u], v: u, w: key[u] });
        for (let v = 0; v < n; v++) {
            if (inMST[v]) continue;
            const mrd = Math.max(coreDist[u], coreDist[v], dist[u][v]);
            if (mrd < key[v]) { key[v] = mrd; par[v] = u; }
        }
    }
    mstEdges.sort((a, b) => a.w - b.w); // ascending weight = increasing merge distance

    // ── 3. Single-linkage dendrogram via union-find ────────────────────────
    const uf = Array.from({ length: 2 * n }, (_, i) => i);
    const sz = new Int32Array(2 * n).fill(1);
    function find(x) { while (uf[x] !== x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; }
    let nextNode = n;
    const dendCh = new Map(); // dendNode → { left, right, lambda, szL, szR }
    for (const { u, v, w } of mstEdges) {
        const ru = find(u), rv = find(v);
        if (ru === rv) continue;
        const szL = sz[ru], szR = sz[rv];
        const nd  = nextNode++;
        sz[nd] = szL + szR;
        uf[ru] = nd; uf[rv] = nd;
        dendCh.set(nd, { left: ru, right: rv, lambda: w > 0 ? 1.0 / w : 1e9, szL, szR });
    }
    const dendRoot = nextNode - 1;
    if (dendRoot < n) return { clusters: concepts.map(c => [c]), pairs: [] };

    // ── 4. Condensed tree traversal ───────────────────────────────────────
    // KEY FIX: when only ONE side ≥ minClusterSize → same condensed cluster
    // CONTINUES (birth unchanged); do NOT create a new condensed node.
    // Only a REAL SPLIT (both ≥ minClusterSize) births two new clusters.
    // nodeCluster[i] for i < n tracks which condensed cluster leaf i belongs to.

    const nodeCluster = new Int32Array(n).fill(-1);  // leaf → condensed cluster id

    // Per-condensed-cluster tables
    const clStability = [];   // accumulated EOM stability
    const clBirth     = [];   // birth lambda of this condensed cluster
    const clParentCl  = [];   // parent condensed cluster index (-1 = none)
    const clChildCl   = [];   // child condensed cluster indices (only real splits)

    function newCluster(birth, parent) {
        const ci = clStability.length;
        clStability.push(0); clBirth.push(birth);
        clParentCl.push(parent); clChildCl.push([]);
        return ci;
    }

    // Iterative leaf collector (avoids call-stack overflow for large n)
    function getLeaves(dendNode) {
        if (dendNode < n) return [dendNode];
        const stack = [dendNode], result = [];
        while (stack.length) {
            const cur = stack.pop();
            if (cur < n) { result.push(cur); continue; }
            const m = dendCh.get(cur);
            if (m) { stack.push(m.left, m.right); }
        }
        return result;
    }

    // condenseFrom: dendNode ∈ single-linkage tree, ci = current condensed cluster,
    // clusterBirth = lambda at which ci was born (used for stability accumulation).
    function condenseFrom(dendNode, ci, clusterBirth) {
        if (dendNode < n) {
            // Raw leaf — permanently belongs to condensed cluster ci
            nodeCluster[dendNode] = ci;
            return;
        }
        const m = dendCh.get(dendNode);
        if (!m) { nodeCluster[dendNode] = ci; return; }
        const { left, right, lambda, szL, szR } = m;
        const leftBig  = szL >= minClusterSize;
        const rightBig = szR >= minClusterSize;

        if (leftBig && rightBig) {
            // ── Real split → two new condensed clusters born at this lambda ──
            const cL = newCluster(lambda, ci);
            const cR = newCluster(lambda, ci);
            clChildCl[ci].push(cL, cR);
            condenseFrom(left,  cL, lambda);
            condenseFrom(right, cR, lambda);

        } else if (leftBig) {
            // Right too small → right points fall OUT of ci at this lambda
            const fallen = getLeaves(right);
            clStability[ci] += fallen.length * (lambda - clusterBirth);
            // Left continues as THE SAME cluster ci (birth unchanged)
            condenseFrom(left, ci, clusterBirth);

        } else if (rightBig) {
            // Left too small → left points fall out
            const fallen = getLeaves(left);
            clStability[ci] += fallen.length * (lambda - clusterBirth);
            condenseFrom(right, ci, clusterBirth);

        } else {
            // Both too small → cluster ci ends here; all remaining points fall out
            const allLeaves = getLeaves(dendNode);
            clStability[ci] += allLeaves.length * (lambda - clusterBirth);
            for (const leaf of allLeaves) nodeCluster[leaf] = ci;
        }
    }

    const rootCl = newCluster(0, -1);
    condenseFrom(dendRoot, rootCl, 0);

    // ── 5. Excess-of-Mass cluster selection ──────────────────────────────
    const selectedCl = new Set();

    function selectEOM(ci) {
        if (clChildCl[ci].length === 0) {
            // Leaf condensed cluster (no further splits) → always candidate
            selectedCl.add(ci);
            return clStability[ci];
        }
        const childSum = clChildCl[ci].reduce((s, c) => s + selectEOM(c), 0);
        if (clStability[ci] >= childSum) {
            // This cluster beats its children → absorb them
            function deselAll(c) { selectedCl.delete(c); clChildCl[c].forEach(deselAll); }
            clChildCl[ci].forEach(deselAll);
            selectedCl.add(ci);
            return clStability[ci];
        }
        return childSum; // children win
    }
    selectEOM(rootCl);

    // Optional cluster_selection_epsilon: drop clusters born at too-low density
    if (clusterSelectionEps > 0) {
        const minLambda = 1.0 / clusterSelectionEps;
        for (const ci of [...selectedCl]) {
            if (clBirth[ci] < minLambda) selectedCl.delete(ci);
        }
    }

    // ── 6. Label leaf points ──────────────────────────────────────────────
    // Walk each leaf's condensed cluster ancestry to find the selected ancestor.
    // This correctly handles cases where a leaf's condensed cluster was not
    // selected but an ancestor was (e.g. root absorbs its children via EOM).
    function findSelected(ci) {
        let c = ci;
        while (c >= 0) { if (selectedCl.has(c)) return c; c = clParentCl[c]; }
        return -1;
    }

    const labels   = new Int32Array(n).fill(-1);
    const clIdxMap = new Map();   // selected condensed cluster → output integer label
    let   nextLabel = 0;

    for (let i = 0; i < n; i++) {
        const ci = nodeCluster[i];
        if (ci < 0) continue;
        const selCi = findSelected(ci);
        if (selCi < 0) continue;
        if (!clIdxMap.has(selCi)) clIdxMap.set(selCi, nextLabel++);
        labels[i] = clIdxMap.get(selCi);
    }

    // Enforce minClusterSize on final counts (safety net)
    const labelCounts = new Map();
    for (let i = 0; i < n; i++) if (labels[i] >= 0) labelCounts.set(labels[i], (labelCounts.get(labels[i]) || 0) + 1);
    for (let i = 0; i < n; i++) {
        if (labels[i] >= 0 && (labelCounts.get(labels[i]) || 0) < minClusterSize) labels[i] = -1;
    }

    // ── 7. Build output ───────────────────────────────────────────────────
    const clusterMap = new Map();
    for (let i = 0; i < n; i++) {
        const lbl = labels[i] >= 0 ? labels[i] : 'noise';
        if (!clusterMap.has(lbl)) clusterMap.set(lbl, []);
        clusterMap.get(lbl).push(concepts[i]);
    }
    const clusters = [];
    for (const [lbl, members] of clusterMap) { if (lbl !== 'noise') clusters.push(members); }
    const noise = clusterMap.get('noise') || [];
    for (const c of noise) clusters.push([c]);
    clusters.sort((a, b) => b.length - a.length);

    // Pairs: same-cluster concept pairs sorted by cosine similarity
    const pairs = [];
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            if (labels[i] >= 0 && labels[i] === labels[j]) {
                pairs.push({ concept1: concepts[i], concept2: concepts[j], similarity: simMatrix[i][j] });
            }
        }
    }
    pairs.sort((a, b) => b.similarity - a.similarity);
    return { clusters, pairs };
}

// ─── Visualization (WebGL + scalable) ────────────────────────────────────────
function renderVisualization(concepts, xy, clusters, method, params) {
    const n          = concepts.length;
    const showLabels = n <= LABEL_THRESHOLD;
    const useWebGL   = n > WEBGL_THRESHOLD;
    const traceType  = useWebGL ? 'scattergl' : 'scatter';

    // cluster id per concept (-1 = noise)
    const conceptCluster = new Map();
    let realCnt = 0;
    clusters.forEach(cl => {
        const isNoise = cl.length === 1;
        cl.forEach(c => conceptCluster.set(c, isNoise ? -1 : realCnt));
        if (!isNoise) realCnt++;
    });

    // Large datasets → single colorscale trace (much faster Plotly render)
    if (useWebGL || realCnt > 30) {
        const x = [], y = [], text = [], color = [], customdata = [];
        concepts.forEach((c, i) => {
            x.push(xy[i][0]); y.push(xy[i][1]);
            text.push(showLabels ? c : '');
            customdata.push(c);
            color.push(conceptCluster.get(c));
        });
        Plotly.newPlot('plot-container', [{
            x, y, text, customdata,
            mode: showLabels ? 'markers+text' : 'markers',
            type: traceType,
            textposition: 'top center',
            textfont: { size: 10 },
            hovertemplate: '<b>%{customdata}</b><br>cluster id: %{marker.color}<extra></extra>',
            marker: {
                size: n > 500 ? 5 : n > 200 ? 7 : 9,
                color,
                colorscale: 'Rainbow',
                showscale: true,
                colorbar: { title: 'Cluster ID', thickness: 14, len: 0.7 },
                opacity: 0.85,
                line: { width: 0 }
            }
        }], buildLayout(method, params, n), { responsive: true });
    } else {
        // Per-cluster named traces (rich legend for small-medium datasets)
        const traces = [];
        let clNum = 0;
        clusters.forEach(cl => {
            const isNoise = cl.length === 1;
            const x = [], y = [], text = [], cd = [];
            cl.forEach(c => {
                const i = concepts.indexOf(c);
                x.push(xy[i][0]); y.push(xy[i][1]);
                text.push(c); cd.push(c);
            });
            traces.push({
                x, y, text, customdata: cd,
                mode: showLabels ? 'markers+text' : 'markers',
                type: traceType,
                name: isNoise ? `Noise (${cl.length})` : `Cluster ${++clNum} · ${cl.length}`,
                textposition: 'top center',
                textfont: { size: 10 },
                hovertemplate: '<b>%{customdata}</b><extra>%{fullData.name}</extra>',
                marker: {
                    size: isNoise ? 7 : 11,
                    opacity: isNoise ? 0.4 : 0.9,
                    line: { width: isNoise ? 0 : 1, color: 'white' }
                }
            });
        });
        Plotly.newPlot('plot-container', traces, buildLayout(method, params, n), { responsive: true });
    }
}

function buildLayout(method, params, n) {
    const methodLabel = method === 'hdbscan' ? 'HDBSCAN' : 'Pairwise Threshold';
    const paramStr    = method === 'hdbscan'
        ? `ε=${params.eps}  min_cluster=${params.mcs}  min_samples=${params.ms}  umap_nn=${params.nn}  min_dist=${params.md}`
        : `threshold=${params.threshold}`;
    return {
        title: {
            text: `UMAP 2D · ${methodLabel} · ${n} concepts<br><sub>${paramStr}</sub>`,
            font: { size: 14 }
        },
        hovermode: 'closest',
        showlegend: n <= 200,
        legend: { orientation: 'h', y: -0.15, font: { size: 11 } },
        xaxis: { title: 'UMAP 1', zeroline: false, showgrid: false },
        yaxis: { title: 'UMAP 2', zeroline: false, showgrid: false },
        margin: { t: 75, l: 45, r: 20, b: n > 30 ? 90 : 40 },
        paper_bgcolor: '#f8fafc',
        plot_bgcolor:  '#f8fafc'
    };
}

// ─── Similarity Matrix ────────────────────────────────────────────────────────
function renderMatrix(concepts, simMatrix) {
    const n = concepts.length;
    matrixWarn.style.display = 'none';
    document.getElementById('matrix-container').innerHTML = '';

    if (n > MATRIX_THRESHOLD) {
        matrixWarn.style.display = '';
        matrixWarn.innerHTML = `<p>&#9888; Matrix hidden for <strong>${n}</strong> concepts (limit: ${MATRIX_THRESHOLD}). Too large to render interactively — export the JSON report instead.</p>`;
        return;
    }

    const z = Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (__, j) => +simMatrix[i][j].toFixed(3))
    );
    const tickfont = { size: n > 80 ? 7 : n > 40 ? 9 : 11 };
    const lMargin  = Math.min(220, 8 * Math.max(...concepts.map(c => c.length)));

    Plotly.newPlot('matrix-container', [{
        z, x: concepts, y: concepts,
        type: 'heatmap',
        colorscale: 'RdYlGn',
        zmin: -1, zmax: 1,
        hovertemplate: '%{y} ↔ %{x}<br>similarity: %{z:.3f}<extra></extra>'
    }], {
        title: `Cosine Similarity Matrix (${n}×${n})`,
        margin: { t: 50, l: lMargin, r: 20, b: 160 },
        xaxis: { tickangle: -50, tickfont },
        yaxis: { autorange: 'reversed', tickfont },
        paper_bgcolor: '#f8fafc'
    }, { responsive: true });
}

// ─── Data Report ──────────────────────────────────────────────────────────────
let _lastReport = null;

function renderReport(clusters, pairs, method, modelName, params) {
    const clustersDiv = document.getElementById('clusters-report');
    const pairsDiv    = document.getElementById('pairs-report');
    const real  = clusters.filter(c => c.length > 1);
    const noise = clusters.filter(c => c.length === 1).map(c => c[0]);

    const paramStr = method === 'hdbscan'
        ? `min_cluster_size = ${params.mcs}  ·  min_samples = ${params.ms}  ·  sel_eps = ${params.eps}  ·  umap_n_neighbors = ${params.nn}  ·  umap_min_dist = ${params.md}  ·  umap_n_components = ${params.nc}`
        : `similarity threshold = ${params.threshold}`;

    let html = `<div class="report-section">
        <h3>Clusters &nbsp;<small>${method === 'hdbscan' ? 'HDBSCAN' : 'Pairwise'} &middot; ${modelName}</small></h3>
        <div class="param-pill">${paramStr}</div>
        <div class="cluster-summary">
            <span class="sum-badge">${real.length} cluster${real.length !== 1 ? 's' : ''}</span>
            <span class="sum-badge">${real.reduce((s, c) => s + c.length, 0)} clustered</span>
            <span class="sum-badge muted">${noise.length} unclustered</span>
        </div>`;

    real.forEach((cluster, i) => {
        html += `<div class="cluster-item">
            <div class="cluster-header">
                <span class="cluster-badge">C${i + 1}</span>
                <strong>${cluster.length} concept${cluster.length !== 1 ? 's' : ''}</strong>
            </div>
            <div class="concept-tags">${cluster.map(c => `<span class="concept-tag">${c}</span>`).join('')}</div>
        </div>`;
    });

    if (noise.length) {
        html += `<details class="noise-details">
            <summary>Unclustered / Noise (${noise.length})</summary>
            <div class="concept-tags">${noise.map(c => `<span class="concept-tag muted">${c}</span>`).join('')}</div>
        </details>`;
    }
    html += '</div>';
    clustersDiv.innerHTML = html;

    // Pairs (virtual list capped at 500 for performance)
    const MAX_SHOWN = 500;
    let pHtml = `<div class="report-section">
        <h3>Pairs Above Threshold &nbsp;<small>${pairs.length} total</small></h3>`;
    if (pairs.length === 0) {
        pHtml += '<p class="hint">No pairs found. Try lowering the threshold / epsilon.</p>';
    } else {
        pHtml += '<div class="pairs-list">';
        pairs.slice(0, MAX_SHOWN).forEach(p => {
            const bar = Math.round(Math.max(0, p.similarity) * 100);
            pHtml += `<div class="pair-item">
                <span class="pair-names">${p.concept1} &harr; ${p.concept2}</span>
                <div class="sim-bar-wrap">
                    <div class="sim-bar" style="width:${bar}%"></div>
                    <span class="similarity-score">${p.similarity.toFixed(3)}</span>
                </div>
            </div>`;
        });
        pHtml += '</div>';
        if (pairs.length > MAX_SHOWN) {
            pHtml += `<p class="hint">Showing top ${MAX_SHOWN} of ${pairs.length}. Export JSON for the full list.</p>`;
        }
    }
    pHtml += '</div>';
    pairsDiv.innerHTML = pHtml;

    _lastReport = {
        method, model: modelName, params,
        clusters: clusters.map((c, i) => ({ id: c.length > 1 ? `C${i + 1}` : 'noise', concepts: c })),
        pairs: pairs.map(p => ({ ...p, similarity: +p.similarity.toFixed(4) }))
    };
    exportBtn.style.display = '';
}

// ─── Export ───────────────────────────────────────────────────────────────────
exportBtn.addEventListener('click', () => {
    if (!_lastReport) return;
    const blob = new Blob([JSON.stringify(_lastReport, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `concept_clustering_${Date.now()}.json`;
    a.click();
});

// ─── Main pipeline ────────────────────────────────────────────────────────────
async function processConcepts() {
    const rawConcepts = conceptsInput.value.split('\n').map(c => c.trim()).filter(Boolean);
    if (rawConcepts.length < 2) { alert('Please enter at least 2 concepts.'); return; }

    const concepts  = [...new Set(rawConcepts)];
    const n         = concepts.length;
    const modelKey  = modelSelect.value;
    const modelInfo = MODEL_MAP[modelKey];
    const method    = currentMethod;

    // Read all hyperparameters
    const threshold = parseFloat(document.getElementById('threshold-input').value);
    const umapNN    = parseInt(document.getElementById('umap-nn').value, 10);
    const umapMD    = parseFloat(document.getElementById('umap-md').value);
    const umapNCraw = parseInt(document.getElementById('umap-nc').value, 10);
    const umapNC    = umapNCraw === 0 ? null : umapNCraw;
    const eps       = parseFloat(document.getElementById('eps-input').value);
    const mcs       = parseInt(document.getElementById('mcs-input').value, 10);
    const ms        = parseInt(document.getElementById('ms-input').value, 10);

    const paramSummary = method === 'hdbscan'
        ? { eps, mcs, ms, nn: umapNN, md: umapMD, nc: umapNC ?? 'None' }
        : { threshold };

    try {
        processBtn.disabled = true;
        showProgress(`Loading ${modelInfo.displayName}…`);

        // 1. Load embedding model
        setProgress(3, `Loading ${modelInfo.displayName}… (first run may take 30–120 s)`);
        const extractor = await pipeline('feature-extraction', modelKey);

        // 2. Embed all concepts with progress reporting
        const embeddings = [];
        for (let i = 0; i < n; i++) {
            setProgress(3 + Math.round((i / n) * 45), `Embedding ${i + 1}/${n}: "${concepts[i]}"`);
            const out = await extractor(concepts[i], { pooling: 'mean', normalize: false });
            embeddings.push(l2Normalize(Array.from(out.data)));
        }

        // 3. Build pairwise similarity matrix
        setProgress(48, `Building similarity matrix (${n}×${n})…`);
        const simMatrix = buildSimMatrix(embeddings, pct =>
            setProgress(48 + Math.round(pct * 12), `Similarity matrix ${Math.round(pct * 100)}%…`)
        );

        // 4. Cluster
        setProgress(62, 'Running clustering…');
        let clusters, pairs, preEmbeddings = null;

        if (method === 'hdbscan') {
            // UMAP pre-reduction before HDBSCAN (matches config.py CLUSTERING_PARAMS approach)
            if (umapNC && umapNC < embeddings[0].length) {
                setProgress(64, `UMAP pre-reduction → ${umapNC}D (nn=${umapNN}, min_dist=${umapMD})…`);
                try {
                    preEmbeddings = runUMAPSync(embeddings, umapNC, umapNN, umapMD);
                } catch (e) {
                    console.warn('UMAP pre-reduction failed, falling back to raw embeddings:', e);
                }
            }
            setProgress(70, `HDBSCAN (min_cluster=${mcs}, min_samples=${ms}, sel_eps=${eps})…`);
            ({ clusters, pairs } = clusterHDBSCAN(concepts, simMatrix, mcs, ms, eps, preEmbeddings));
        } else {
            setProgress(70, `Pairwise threshold (τ=${threshold})…`);
            ({ clusters, pairs } = clusterPairwise(concepts, simMatrix, threshold));
        }

        // 5. UMAP 2D for visualization (always 2 components for display)
        setProgress(80, `UMAP 2D projection (${n} concepts)…`);
        let xy;
        try {
            xy = runUMAPSync(embeddings, 2, Math.min(umapNN, n - 1), umapMD);
        } catch (e) {
            console.warn('UMAP viz failed, using first 2 dims:', e);
            xy = embeddings.map(e => [e[0] ?? 0, e[1] ?? 0]);
        }

        // 6. Render all panels
        setProgress(92, 'Rendering…');
        renderVisualization(concepts, xy, clusters, method, paramSummary);
        renderMatrix(concepts, simMatrix);
        renderReport(clusters, pairs, method, modelInfo.displayName, paramSummary);

        const nReal = clusters.filter(c => c.length > 1).length;
        hideProgress(`Done · ${nReal} cluster${nReal !== 1 ? 's' : ''} · ${pairs.length} pairs · ${n} concepts`);
    } catch (err) {
        console.error(err);
        hideProgress(`Error: ${err.message}`);
    } finally {
        processBtn.disabled = false;
    }
}

// ─── ELSST Hierarchy Lookup ────────────────────────────────────────────────────
// Pre-computed caches per model:
//   elsst_paths.json                      (shared metadata, original-case)
//   elsst_embeddings_<key>.bin            (raw Float32)
//   elsst_embeddings_<key>_l2.bin         (L2-normalized Float32)
//
// ELSST model registry — maps model keys to their Xenova browser models
const ELSST_MODELS = {
    'allmpnet': { browser: 'Xenova/all-mpnet-base-v2', display: 'All-MPNet-Base-v2' }
};

// Cache store: { '<key>_raw': ..., '<key>_l2': ..., paths: [...] }
const elsstCacheStore = { paths: null };
let elsstPathsLoaded = false;

async function loadElsstCache() {
    const modelKey = elsstModelSel.value;
    const useL2    = l2Toggle.checked;
    const suffix   = useL2 ? '_l2' : '';
    const cacheKey = `${modelKey}${suffix}`;

    // Already loaded?
    if (elsstCacheStore[cacheKey] && elsstPathsLoaded) {
        const c = elsstCacheStore[cacheKey];
        elsstCacheSt.textContent = `Cache ready: ${c.n} paths · ${ELSST_MODELS[modelKey].display} · ${useL2 ? 'L2' : 'raw'}`;
        return;
    }

    elsstCacheSt.textContent = 'Loading ELSST cache…';
    try {
        // Load shared paths JSON once
        if (!elsstPathsLoaded) {
            const jsonResp = await fetch('elsst_paths.json');
            if (!jsonResp.ok) throw new Error('elsst_paths.json not found. Run build_elsst_cache.py first.');
            elsstCacheStore.paths = await jsonResp.json();
            elsstPathsLoaded = true;
        }

        // Load model-specific embeddings
        const binFile = `elsst_embeddings_${modelKey}${suffix}.bin`;
        const binResp = await fetch(binFile);
        if (!binResp.ok) throw new Error(`${binFile} not found. Run build_elsst_cache.py first.`);

        const buf    = await binResp.arrayBuffer();
        const header = new Uint32Array(buf, 0, 2);
        const n   = header[0];
        const dim = header[1];
        const raw = new Float32Array(buf, 8);

        const embeddings = [];
        for (let i = 0; i < n; i++) {
            embeddings.push(raw.subarray(i * dim, (i + 1) * dim));
        }

        elsstCacheStore[cacheKey] = { embeddings, dim, n };
        elsstCacheSt.textContent = `Cache ready: ${n} paths · ${ELSST_MODELS[modelKey].display} · ${useL2 ? 'L2' : 'raw'}`;
    } catch (err) {
        elsstCacheSt.textContent = `Error: ${err.message}`;
        console.error('ELSST cache load error:', err);
    }
}

// Reload cache when model or L2 toggle changes
elsstModelSel.addEventListener('change', () => loadElsstCache());
l2Toggle.addEventListener('change', () => loadElsstCache());

function elsstDotProduct(a, b) {
    let d = 0;
    for (let i = 0; i < a.length; i++) d += a[i] * b[i];
    return d;
}

function elsstCosineSim(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot   += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
}

async function searchELSST() {
    const rawQuery = elsstQuery.value.trim();
    if (!rawQuery) { alert('Please enter a query term.'); return; }
    const query  = rawQuery.toLowerCase();  // always lowercase for matching
    const topK   = parseInt(document.getElementById('topk-input').value, 10);
    const useL2  = l2Toggle.checked;
    const modelKey = elsstModelSel.value;
    const suffix = useL2 ? '_l2' : '';
    const cacheKey = `${modelKey}${suffix}`;

    try {
        elsstBtn.disabled = true;
        showProgress('Loading ELSST cache…');
        await loadElsstCache();
        const cache = elsstCacheStore[cacheKey];
        if (!cache) throw new Error('ELSST cache not available.');
        const paths = elsstCacheStore.paths;

        // Embed query using the same browser model as the cache expects
        const browserModel = ELSST_MODELS[modelKey].browser;
        setProgress(20, `Loading ${ELSST_MODELS[modelKey].display}…`);
        const extractor = await pipeline('feature-extraction', browserModel);

        setProgress(40, `Embedding query: "${query}"…`);
        const out = await extractor(query, { pooling: 'mean', normalize: false });
        let queryEmb = Array.from(out.data);

        // Apply L2 normalization to query if toggle is on
        if (useL2) {
            queryEmb = l2Normalize(queryEmb);
        }

        // Compute similarity against all cached path embeddings
        setProgress(60, `Comparing against ${cache.n} paths…`);
        const simFn = useL2 ? elsstDotProduct : elsstCosineSim;
        const scores = [];
        for (let i = 0; i < cache.n; i++) {
            scores.push({
                idx: i,
                similarity: simFn(queryEmb, cache.embeddings[i])
            });
        }

        // Sort descending
        scores.sort((a, b) => b.similarity - a.similarity);

        // ── Deduplicate by leaf: keep only the best-scoring path per leaf ──
        const seenLeaves = new Set();
        const deduped = [];
        for (const s of scores) {
            const leaf = paths[s.idx].leaf;
            if (!seenLeaves.has(leaf)) {
                seenLeaves.add(leaf);
                deduped.push(s);
            }
            if (deduped.length >= topK) break;
        }

        setProgress(85, 'Rendering results…');
        renderElsstResults(rawQuery, deduped, topK, modelKey, useL2, paths, cache.n);

        // Switch to ELSST tab
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        const elsstTabBtn = document.querySelector('.tab-btn[data-target="tab-elsst"]');
        if (elsstTabBtn) elsstTabBtn.classList.add('active');
        document.getElementById('tab-elsst').classList.add('active');

        hideProgress(`Done · Top ${deduped.length} matches for "${rawQuery}"`);
    } catch (err) {
        console.error(err);
        hideProgress(`Error: ${err.message}`);
    } finally {
        elsstBtn.disabled = false;
    }
}

function renderElsstResults(query, topResults, topK, modelKey, useL2, paths, nPaths) {
    const modelName = ELSST_MODELS[modelKey].display;
    const normLabel = useL2 ? 'L2-normalized (cosine)' : 'raw (dot product)';

    let html = `<div class="report-section">
        <h3>ELSST Hierarchy Lookup &nbsp;<small>Query: "${query}" · Top ${topK}</small></h3>
        <div class="param-pill">Model: ${modelName} · ${nPaths} root→leaf paths · ${normLabel} · duplicates collapsed</div>
        <div class="elsst-results-list">`;

    topResults.forEach((r, rank) => {
        const entry = paths[r.idx];
        const pathParts = entry.path.split(' > ');
        const leaf = pathParts[pathParts.length - 1];
        const parents = pathParts.slice(0, -1);
        const simPct = Math.round(Math.max(0, r.similarity) * 100);

        html += `<div class="elsst-result-card">
            <div class="elsst-rank">#${rank + 1}</div>
            <div class="elsst-result-body">
                <div class="elsst-leaf-name">${leaf}</div>
                <div class="elsst-path-breadcrumb">
                    ${parents.map(p => `<span class="elsst-bc-parent">${p}</span>`).join('<span class="elsst-bc-sep">›</span>')}
                    ${parents.length ? '<span class="elsst-bc-sep">›</span>' : ''}
                    <span class="elsst-bc-leaf">${leaf}</span>
                </div>
                <div class="sim-bar-wrap">
                    <div class="sim-bar" style="width:${simPct}%"></div>
                    <span class="similarity-score">${r.similarity.toFixed(4)}</span>
                </div>
            </div>
        </div>`;
    });

    html += '</div></div>';
    elsstResults.innerHTML = html;
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
processBtn.addEventListener('click', processConcepts);
elsstBtn.addEventListener('click', searchELSST);

// Allow Enter key in ELSST query input
elsstQuery.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); searchELSST(); }
});
