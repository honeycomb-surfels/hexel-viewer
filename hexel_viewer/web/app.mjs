import {parseBundle, navigationRadius} from "./core.mjs";
import {Controls} from "./controls.mjs";
import {HexelRenderer} from "./renderer.mjs";
import {FrameRate, modelDetails} from "./telemetry.mjs";
import {LoadingNotice, nextPaint} from "./loading.mjs";

const element = name => document.getElementById(name);
const canvas = element("viewport");
const parameters = new URLSearchParams(location.search);
const remote = parameters.get("backend") === "cuda";
const token = new URLSearchParams(location.hash.slice(1)).get("token") || "";
const modes = ["rgb", "primitives", "wireframe", "depth", "normals", "opacity", "overdraw"];
const notes = {
    rgb: "Soft boundary, spatial color and trained view dependence.",
    primitives: "Diagnostic IDs, not RGB. Browser: hard hexagons. CUDA: strongest contributing primitive.",
    wireframe: "Colored SDF-zero boundaries, not a fused or watertight mesh. CUDA outlines the strongest contributor.",
    depth: "CUDA: alpha-normalized expected depth. Browser: alpha-weighted depth preview.",
    normals: "Camera-facing normals. CUDA normalizes the accumulated vector; browser blends primitive colors.",
    opacity: "Accumulated coverage, not per-primitive opacity.",
    overdraw: "CUDA accepted base-hexel hits before color early termination; log scale, white at 16 hits.",
};
let dirty = true, sortDirty = true, renderer, context, metadata, worker;
let generation = 0, sorting = false, sortMs = 0, frameMs = 16, lastTime = performance.now();
let scaleFactor = 1, busy = false, lastRemote = 0, lastStats = 0, lastAdaptation = 0, remoteStats = {};
let remoteBitmap = null, remoteComparison = false;
const displayedFrames = new FrameRate();
const controls = new Controls(canvas, navigation => {
    dirty = true; sortDirty = true;
    if (navigation) element("navigation").value = navigation;
});
const status = message => { element("status").textContent = message; };
const fail = error => { console.error(error); status(error.message || String(error)); };
const loading = new LoadingNotice(element("loading-notice"), status);

function scene(info) {
    metadata = info;
    controls.reset(info);
    element("scene-name").textContent = info.name || `Hexels · ${info.count.toLocaleString()} primitives`;
    element("intro").textContent = remote ? "CUDA rendering · read-only viewer" : `${info.view_model} appearance · ${info.count.toLocaleString()} hexels · no rendering server`;
    element("model-details").textContent = modelDetails(info);
    element("welcome").hidden = true;
    element("depth-scale").value = Math.max(1, navigationRadius(info) * 4).toFixed(2);
    element("camera").replaceChildren(new Option("Free camera", "-1"));
    info.cameras?.forEach((camera, index) => element("camera").add(new Option(camera.name, String(index))));
    if (info.cameras?.length) element("camera").value = "0";
    dirty = true; sortDirty = true;
}

async function load(buffer) {
    loading.stage(`Reading model attributes · ${(buffer.byteLength / 1e6).toFixed(1)} MB`);
    await nextPaint();
    const {metadata: info, rows} = parseBundle(buffer);
    info.bundle_bytes = buffer.byteLength;
    displayedFrames.times = [];
    if (!renderer) renderer = new HexelRenderer(canvas);
    if (!renderer.floatTarget) {
        element("quality").value = "adaptive";
        element("quality").options[0].disabled = true;
    }
    loading.stage(`Uploading ${info.count.toLocaleString()} primitives to the device GPU…`);
    await nextPaint();
    renderer.load(info, rows);
    const positions = new Float32Array(info.count * 3);
    for (let index = 0; index < info.count; index++) positions.set(rows.subarray(index * info.texels * 4, index * info.texels * 4 + 3), index * 3);
    worker?.terminate();
    worker = new Worker(new URL("./sort-worker.mjs", import.meta.url), {type: "module"});
    worker.postMessage({positions}, [positions.buffer]);
    generation++; sorting = false;
    worker.onmessage = ({data}) => {
        if (data.generation !== generation) return;
        renderer.setOrder(data.order); sortMs = data.milliseconds; sorting = false; dirty = true;
    };
    worker.onerror = event => { sorting = false; fail(new Error(event.message)); };
    scene(info);
    status(renderer.floatTarget ? `Loaded · float${renderer.float32Blend ? 32 : 16} accumulation; not CUDA-bit-exact` : "Preview only: float framebuffer unavailable, RGB clips per layer");
}

async function openModel(read, message) {
    if (loading.active) return;
    loading.start(message);
    element("welcome").hidden = true;
    element("file").disabled = true;
    element("load-url").disabled = true;
    canvas.setAttribute("aria-busy", "true");
    try {
        await nextPaint();
        await load(await read());
    } finally {
        loading.stop();
        element("file").disabled = false;
        element("file").value = "";
        element("load-url").disabled = false;
        canvas.setAttribute("aria-busy", "false");
        element("welcome").hidden = Boolean(metadata);
    }
}

const loadFile = file => openModel(() => file.arrayBuffer(), `Reading ${file.name} · ${(file.size / 1e6).toFixed(1)} MB`);

async function loadUrl(url) {
    if (!url) return;
    const parsed = new URL(url, location.href);
    if (!["http:", "https:"].includes(parsed.protocol)) throw new Error("Use an HTTP(S) bundle URL");
    await openModel(async () => {
        const response = await fetch(parsed);
        if (!response.ok) throw new Error(`Download failed (${response.status}); check the URL and CORS settings`);
        return response.arrayBuffer();
    }, "Downloading model…");
}

function download(blob, name) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob); link.download = name; link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function resolution() {
    const adaptive = element("quality").value === "adaptive";
    const density = adaptive ? Math.min(devicePixelRatio, 1.5) : devicePixelRatio;
    let width = Math.round(innerWidth * density * Number(element("resolution").value) * (adaptive ? scaleFactor : 1));
    let height = Math.round(innerHeight * density * Number(element("resolution").value) * (adaptive ? scaleFactor : 1));
    const cap = remote ? 1920 * 1080 / (element("compare-enabled").checked ? 2 : 1) : adaptive ? 1440 * 1080 : 4096 * 2160;
    const reduction = Math.min(1, Math.sqrt(cap / (width * height)), 4096 / Math.max(width, height));
    return [Math.max(16, Math.floor(width * reduction)), Math.max(16, Math.floor(height * reduction))];
}

async function remoteFrame(width, height) {
    busy = true; dirty = false;
    const camera = controls.camera();
    const compareMode = element("compare-enabled").checked ? element("compare-mode").value : null;
    try {
        const response = await fetch("api/render", {
            method: "POST", headers: {"Authorization": `Bearer ${token}`, "Content-Type": "application/json"},
            body: JSON.stringify({...camera, view: undefined, width, height, mode: element("mode").value, compare_mode: compareMode, depth_scale: Number(element("depth-scale").value)}),
            signal: AbortSignal.timeout(35000),
        });
        if (!response.ok) throw new Error((await response.json()).error);
        const bitmap = await createImageBitmap(await response.blob());
        if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
        remoteBitmap?.close(); remoteBitmap = bitmap; remoteComparison = compareMode !== null;
        drawRemote();
        displayedFrames.record(performance.now());
        remoteStats = JSON.parse(response.headers.get("X-Hexels-Stats") || "{}");
        status("CUDA connected · read-only");
    } catch (error) { fail(error); dirty = true; }
    finally { busy = false; lastRemote = performance.now(); }
}

function drawRemote() {
    if (!remoteBitmap) return;
    const width = canvas.width, height = canvas.height;
    context.drawImage(remoteBitmap, 0, 0, width, height, 0, 0, width, height);
    if (remoteComparison) {
        const split = Math.round(width * Number(element("comparison-position").value) / 100);
        if (split < width) context.drawImage(remoteBitmap, width + split, 0, width - split, height, split, 0, width - split, height);
    }
}

function updateComparison() {
    const enabled = element("compare-enabled").checked;
    element("comparison-layer").hidden = !enabled;
    element("compare-controls").hidden = !enabled;
    element("comparison-left").textContent = element("mode").selectedOptions[0].text;
    element("comparison-right").textContent = element("compare-mode").selectedOptions[0].text;
    dirty = true;
}

function divider(value) {
    const position = Math.max(0, Math.min(100, value));
    element("comparison-position").value = String(position);
    element("comparison-handle").style.left = `${position}%`;
    element("comparison-handle").setAttribute("aria-valuenow", String(Math.round(position)));
    if (remote) drawRemote(); else dirty = true;
}

function frame(now) {
    requestAnimationFrame(frame);
    const elapsed = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;
    if (document.hidden || !metadata || loading.active) return;
    controls.tick(elapsed);
    const adaptive = element("quality").value === "adaptive";
    const budget = adaptive ? Number(element("budget").value) || metadata.count : metadata.count;
    if (!remote && sortDirty && !sorting) {
        sorting = true; sortDirty = false;
        worker.postMessage({forward: controls.forward, position: controls.position, budget, generation});
    }
    const [width, height] = resolution();
    if (remote) {
        if (!busy && (dirty || metadata.live) && now - lastRemote > (metadata.live ? 500 : 30)) remoteFrame(width, height);
    } else if (dirty || canvas.width !== width || canvas.height !== height) {
        renderer.resize(width, height);
        renderer.render(controls.camera(), modes.indexOf(element("mode").value), Number(element("depth-scale").value));
        if (element("compare-enabled").checked) {
            const split = Math.round(width * Number(element("comparison-position").value) / 100);
            renderer.render(controls.camera(), modes.indexOf(element("compare-mode").value), Number(element("depth-scale").value), [split, width - split]);
        }
        dirty = false;
        displayedFrames.record(performance.now());
        frameMs = frameMs * 0.9 + elapsed * 1000 * 0.1;
        if (adaptive && controls.keys.size && now - lastAdaptation > 1000) {
            lastAdaptation = now;
            if (frameMs > 40) scaleFactor = Math.max(0.5, scaleFactor * 0.9);
            if (frameMs < 22) scaleFactor = Math.min(1, scaleFactor * 1.05);
        }
    }
    if (now - lastStats > 500) {
        lastStats = now;
        const detail = adaptive ? "PREVIEW" : "FULL";
        const fps = displayedFrames.value(performance.now()).toFixed(1);
        if (remote) element("model-details").textContent = modelDetails({...metadata, count: remoteStats.primitive_count ?? metadata.count});
        element("stats").textContent = remote
            ? `${fps} delivered fps · iter ${remoteStats.iteration ?? metadata.iteration} · ${(remoteStats.primitive_count ?? metadata.count).toLocaleString()} hexels · ${(remoteStats.render_transfer_ms || 0).toFixed(1)} ms render+transfer · ${(remoteStats.encode_ms || 0).toFixed(1)} ms PNG`
            : `${detail} · ${renderer.visible.toLocaleString()} drawn · ${fps} displayed fps · ${sortMs.toFixed(1)} ms sort · ${width}×${height}`;
    }
}

element("file").onchange = event => { const file = event.target.files[0]; if (file) loadFile(file).catch(fail); };
element("load-url").onclick = () => loadUrl(element("model-url").value).catch(fail);
window.addEventListener("dragover", event => event.preventDefault());
window.addEventListener("drop", event => { event.preventDefault(); const file = event.dataTransfer.files[0]; if (!remote && file) loadFile(file).catch(fail); });
element("mode").onchange = () => { updateComparison(); element("mode-note").textContent = notes[element("mode").value]; };
element("compare-enabled").onchange = updateComparison;
element("compare-mode").onchange = updateComparison;
element("comparison-position").oninput = event => divider(Number(event.target.value));
element("comparison-handle").onpointerdown = event => { event.preventDefault(); event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId); };
element("comparison-handle").onpointermove = event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) divider(event.clientX / innerWidth * 100); };
element("comparison-handle").onkeydown = event => {
    const value = Number(element("comparison-position").value);
    const positions = {ArrowLeft: value - 2, ArrowRight: value + 2, Home: 0, End: 100};
    if (event.key in positions) { event.preventDefault(); divider(positions[event.key]); }
};
element("help-toggle").onclick = () => element("help-dialog").showModal();
window.addEventListener("keydown", event => {
    if (event.key === "?" && !/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) element("help-dialog").showModal();
});
element("navigation").onchange = () => { controls.navigation = element("navigation").value; };
element("camera").onchange = () => { const selected = Number(element("camera").value); if (selected >= 0) controls.setView(metadata.cameras[selected]); };
element("reset").onclick = () => metadata && controls.reset(metadata);
element("fullscreen").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen?.().catch(fail); };
element("toggle-panel").onclick = () => {
    element("panel").hidden = !element("panel").hidden;
    element("touch-pad").hidden = !element("panel").hidden;
    element("toggle-panel").setAttribute("aria-expanded", String(!element("panel").hidden));
};
for (const name of ["quality", "resolution", "budget", "depth-scale"]) element(name).oninput = () => { dirty = true; sortDirty = true; element("scale-value").textContent = `${Math.round(Number(element("resolution").value) * 100)}%`; };
element("save-camera").onclick = () => download(new Blob([JSON.stringify(controls.camera(), null, 2)], {type: "application/json"}), "hexels-camera.json");
element("camera-file").onchange = async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
        const camera = JSON.parse(await file.text());
        if (!["position", "forward", "up"].every(name => Array.isArray(camera[name]) && camera[name].length === 3 && camera[name].every(Number.isFinite)) || !(camera.fov >= 10 && camera.fov <= 140)) throw new Error("Invalid camera JSON");
        if (Math.hypot(...camera.forward) < 0.001 || Math.hypot(...camera.up) < 0.001) throw new Error("Camera axes cannot be zero");
        controls.setView(camera);
        element("camera").value = "-1";
    } catch (error) { fail(error); }
};
element("screenshot").onclick = () => canvas.toBlob(blob => blob && download(blob, "hexels-view.png"));
window.addEventListener("resize", () => { dirty = true; });
canvas.addEventListener("webglcontextlost", event => { event.preventDefault(); status("GPU context lost. Reload the page and use a smaller preview budget."); metadata = null; });
if (matchMedia("(pointer: coarse)").matches) { element("quality").value = "adaptive"; element("panel").hidden = true; element("toggle-panel").setAttribute("aria-expanded", "false"); }
if (parameters.has("scale")) {
    const requested = Number(parameters.get("scale"));
    if (Number.isFinite(requested)) element("resolution").value = String(Math.max(0.35, Math.min(1, requested)));
    element("scale-value").textContent = `${Math.round(Number(element("resolution").value) * 100)}%`;
}

if (remote) {
    context = canvas.getContext("2d");
    element("backend").textContent = "CUDA STREAM";
    element("file-label").hidden = true; element("url-controls").hidden = true; element("budget-label").hidden = true;
    element("mode").add(new Option("Accepted overdraw hits", "overdraw"));
    status("Connecting to CUDA viewer…");
    fetch("api/info", {headers: {"Authorization": `Bearer ${token}`}}).then(async response => {
        const info = await response.json(); if (!response.ok) throw new Error(info.error); scene(info);
    }).catch(fail);
} else if (parameters.get("model")) loadUrl(parameters.get("model")).catch(fail);
requestAnimationFrame(frame);
