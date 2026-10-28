export class FrameRate {
    constructor(windowMs = 1500) {
        this.windowMs = windowMs;
        this.times = [];
    }

    record(now) {
        this.times.push(now);
        this.trim(now);
    }

    trim(now) {
        while (this.times.length > 2 && this.times[0] < now - this.windowMs) this.times.shift();
    }

    value(now) {
        this.trim(now);
        if (!this.times.length || now - this.times.at(-1) > this.windowMs) return 0;
        return this.times.length > 1 ? (this.times.length - 1) * 1000 / Math.max(1, now - this.times[0]) : 0;
    }
}

export function modelDetails(info) {
    const parts = [`${info.count.toLocaleString()} primitives`];
    if (info.bundle_bytes != null) parts.push(`${(info.bundle_bytes / 1e6).toFixed(2)} MB browser bundle`);
    if (info.deployment_bytes != null) parts.push(`${(info.deployment_bytes / 1e6).toFixed(2)} MB saved model`);
    if (info.active_state_bytes != null) parts.push(`${(info.active_state_bytes / 1e6).toFixed(2)} MB learned state`);
    if (info.checkpoint_bytes != null && info.deployment_bytes == null) parts.push(`${(info.checkpoint_bytes / 1e6).toFixed(2)} MB checkpoint`);
    return parts.join(" · ");
}
