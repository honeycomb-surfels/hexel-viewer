export const add = (left, right) => left.map((value, axis) => value + right[axis]);
export const scale = (vector, amount) => vector.map(value => value * amount);
export const dot = (left, right) => left.reduce((sum, value, axis) => sum + value * right[axis], 0);
export const cross = (left, right) => [left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2], left[0] * right[1] - left[1] * right[0]];
export const normalize = vector => scale(vector, 1 / Math.max(Math.hypot(...vector), 1e-12));

export function rotate(vector, axis, angle) {
    return add(add(scale(vector, Math.cos(angle)), scale(cross(axis, vector), Math.sin(angle))), scale(axis, dot(axis, vector) * (1 - Math.cos(angle))));
}

export function viewMatrix(position, forward, up) {
    const right = normalize(cross(forward, up));
    const down = cross(forward, right);
    return new Float32Array([
        right[0], down[0], forward[0], 0,
        right[1], down[1], forward[1], 0,
        right[2], down[2], forward[2], 0,
        -dot(right, position), -dot(down, position), -dot(forward, position), 1,
    ]);
}

export function parseBundle(buffer) {
    if (buffer.byteLength < 12 || new TextDecoder().decode(new Uint8Array(buffer, 0, 8)) !== "HEXVIEW1") throw new Error("Expected a .hexview bundle exported by Hexels");
    const length = new DataView(buffer).getUint32(8, true);
    if (length % 4 || length > 16 * 1024 * 1024 || 12 + length > buffer.byteLength) throw new Error("Invalid bundle header length");
    const metadata = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 12, length)));
    if (metadata.version !== 1 || metadata.format !== "hexels.viewer" || !Number.isSafeInteger(metadata.count) || metadata.count < 1) throw new Error("Unsupported bundle version or count");
    if (![0, 3, 15].includes(metadata.basis_count) || metadata.texels !== 6 + metadata.basis_count) throw new Error("Unsupported appearance layout");
    const expected = metadata.count * metadata.texels * 16;
    if (buffer.byteLength !== 12 + length + expected) throw new Error("Truncated bundle or inconsistent primitive count");
    const rows = new Float32Array(buffer, 12 + length);
    for (const value of rows) if (!Number.isFinite(value)) throw new Error("Non-finite primitive attributes");
    return {metadata, rows};
}

export function sortDepths(positions, forward, position, budget = Infinity) {
    const count = positions.length / 3;
    const selected = Math.min(count, Math.max(1, Math.floor(budget)));
    let indices = new Uint32Array(selected);
    const ids = new Uint32Array(selected);
    const depth = new Float32Array(selected);
    const keys = new Uint32Array(depth.buffer);
    let visible = 0;
    for (let index = 0; index < selected; index++) {
        const primitive = Math.floor(index * count / selected);
        const offset = primitive * 3;
        const distance = (positions[offset] - position[0]) * forward[0] + (positions[offset + 1] - position[1]) * forward[1] + (positions[offset + 2] - position[2]) * forward[2];
        if (distance <= 0.2 || !Number.isFinite(distance)) continue;
        indices[visible] = visible;
        ids[visible] = primitive;
        depth[visible++] = distance;
    }
    let temporary = new Uint32Array(selected);
    const histogram = new Uint32Array(256);
    for (let shift = 0; shift < 32; shift += 8) {
        histogram.fill(0);
        for (let index = 0; index < visible; index++) histogram[(keys[indices[index]] >>> shift) & 255]++;
        let offset = 0;
        for (let bucket = 0; bucket < 256; bucket++) {
            const size = histogram[bucket]; histogram[bucket] = offset; offset += size;
        }
        for (let index = 0; index < visible; index++) {
            const value = indices[index];
            temporary[histogram[(keys[value] >>> shift) & 255]++] = value;
        }
        [indices, temporary] = [temporary, indices];
    }
    return Uint32Array.from(indices.subarray(0, visible), index => ids[index]);
}

export function navigationRadius(metadata) {
    const cameras = metadata.cameras || [];
    if (cameras.length < 2) return metadata.radius;
    const center = [0, 1, 2].map(axis => cameras.reduce((total, camera) => total + camera.position[axis], 0) / cameras.length);
    const distances = cameras.map(camera => Math.hypot(...camera.position.map((value, axis) => value - center[axis]))).sort((left, right) => left - right);
    const radius = distances[Math.floor(distances.length / 2)];
    return radius > 0 ? Math.min(metadata.radius, radius) : metadata.radius;
}
