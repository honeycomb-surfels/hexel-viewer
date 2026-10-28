import assert from "node:assert/strict";
import {test} from "node:test";
import {parseBundle, sortDepths, viewMatrix, navigationRadius} from "../hexel_viewer/web/core.mjs";
import {FrameRate, modelDetails} from "../hexel_viewer/web/telemetry.mjs";
import {LoadingNotice, patienceMessages} from "../hexel_viewer/web/loading.mjs";

test("loading phrases rotate only while active and reset after failure or success", context => {
    context.mock.timers.enable({apis: ["setTimeout", "setInterval"]});
    const classes = new Set();
    const message = {textContent: "", classList: {add: value => classes.add(value), remove: value => classes.delete(value)}};
    const stage = {textContent: ""};
    const root = {hidden: true, querySelector: selector => selector === ".loading-message" ? message : stage};
    let status;
    const notice = new LoadingNotice(root, value => { status = value; });
    notice.start("Reading file");
    assert.equal(notice.active, true);
    assert.equal(status, "Reading file");
    assert.equal(message.textContent, patienceMessages[0]);
    context.mock.timers.tick(5000);
    assert.equal(classes.has("changing"), true);
    context.mock.timers.tick(220);
    assert.equal(message.textContent, patienceMessages[1]);
    notice.stop();
    context.mock.timers.tick(10000);
    assert.equal(notice.active, false);
    assert.equal(message.textContent, patienceMessages[1]);
    notice.start("Retry");
    assert.equal(message.textContent, patienceMessages[0]);
    context.mock.timers.tick(5000);
    notice.stop();
    context.mock.timers.tick(220);
    assert.equal(message.textContent, patienceMessages[0]);
    assert.equal(classes.has("changing"), false);
});

test("FPS counts real frames without clamping slow frame intervals", () => {
    const frames = new FrameRate();
    [0, 400, 800, 1200].forEach(now => frames.record(now));
    assert.equal(frames.value(1200), 2.5);
    assert.equal(frames.value(3000), 0);
    const slow = new FrameRate();
    slow.record(0); slow.record(2000);
    assert.equal(slow.value(2000), 0.5);
});

test("model storage labels distinguish deployment and browser representations", () => {
    const text = modelDetails({count: 20, bundle_bytes: 2e6, deployment_bytes: 1e6, active_state_bytes: 1600});
    assert.match(text, /20 primitives/);
    assert.match(text, /2.00 MB browser bundle/);
    assert.match(text, /1.00 MB saved model/);
});

test("WASD scale follows the capture rather than distant background primitives", () => {
    assert.equal(navigationRadius({radius: 30, cameras: [{position: [-2, 0, 0]}, {position: [2, 0, 0]}]}), 2);
    assert.equal(navigationRadius({radius: 1, cameras: []}), 1);
    assert.equal(navigationRadius({radius: 1, cameras: [{position: [0, 0, 0]}, {position: [0, 0, 0]}]}), 1);
});

test("camera matrices use +Z-forward and column-major storage", () => {
    assert.deepEqual(Array.from(viewMatrix([1, 2, 3], [0, 0, 1], [0, -1, 0])).map(value => value || 0), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, -2, -3, 1]);
});

test("radix sorting is front-to-back and stable at equal depth", () => {
    const positions = new Float32Array([0, 0, 5, 0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 3]);
    assert.deepEqual(Array.from(sortDepths(positions, [0, 0, 1], [0, 0, 0])), [1, 3, 4, 0]);
    assert.deepEqual(Array.from(sortDepths(positions, [0, 0, 1], [0, 0, 0], 2)), [0]);
});

test("radix sorting matches stable float32 reference", () => {
    const positions = Float32Array.from({length: 3000}, (_, index) => Math.sin(index * 1.71) * 10);
    const expected = Array.from({length: 1000}, (_, index) => index).filter(index => positions[index * 3 + 2] > 0.2).sort((left, right) => positions[left * 3 + 2] - positions[right * 3 + 2]);
    assert.deepEqual(Array.from(sortDepths(positions, [0, 0, 1], [0, 0, 0])), expected);
});

test("invalid/truncated bundles fail explicitly", () => {
    assert.throws(() => parseBundle(new ArrayBuffer(4)), /bundle/);
    const header = new TextEncoder().encode(JSON.stringify({format: "hexels.viewer", version: 1, count: 1, texels: 6, basis_count: 0}));
    const aligned = Math.ceil(header.length / 4) * 4;
    const buffer = new ArrayBuffer(12 + aligned + 96);
    const bytes = new Uint8Array(buffer);
    bytes.set(new TextEncoder().encode("HEXVIEW1"));
    new DataView(buffer).setUint32(8, aligned, true);
    bytes.fill(32, 12, 12 + aligned); bytes.set(header, 12);
    assert.equal(parseBundle(buffer).rows.length, 24);
    assert.throws(() => parseBundle(buffer.slice(0, -4)), /Truncated/);
});
