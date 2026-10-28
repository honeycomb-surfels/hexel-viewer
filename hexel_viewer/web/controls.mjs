import {add, scale, cross, normalize, rotate, dot, viewMatrix, navigationRadius} from "./core.mjs";

export class Controls {
    constructor(canvas, changed) {
        this.canvas = canvas;
        canvas.tabIndex = 0;
        this.changed = changed;
        this.keys = new Set();
        this.pointers = new Map();
        this.navigation = "fly";
        this.speed = 1;
        this.reset({center: [0, 0, 0], radius: 1, cameras: []});
        window.addEventListener("keydown", event => {
            if (/INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName) || event.target.isContentEditable) return;
            if (["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE", "Space", "ShiftLeft", "ShiftRight"].includes(event.code)) {
                this.keys.add(event.code); event.preventDefault();
            }
            if (event.code === "KeyR") this.reset(this.metadata);
            if (event.code === "KeyF") canvas.requestPointerLock?.();
            if (event.code === "KeyO") {
                this.navigation = this.navigation === "fly" ? "orbit" : "fly";
                this.changed(this.navigation);
            }
        });
        window.addEventListener("keyup", event => this.keys.delete(event.code));
        window.addEventListener("blur", () => { this.keys.clear(); this.pointers.clear(); });
        document.addEventListener("visibilitychange", () => this.keys.clear());
        canvas.addEventListener("blur", () => this.keys.clear());
        canvas.addEventListener("contextmenu", event => event.preventDefault());
        canvas.addEventListener("pointerdown", event => {
            canvas.focus({preventScroll: true});
            canvas.setPointerCapture(event.pointerId);
            this.pointers.set(event.pointerId, [event.clientX, event.clientY]);
        });
        const release = event => this.pointers.delete(event.pointerId);
        canvas.addEventListener("pointerup", release);
        canvas.addEventListener("pointercancel", release);
        canvas.addEventListener("pointermove", event => this.pointer(event));
        canvas.addEventListener("wheel", event => {
            event.preventDefault();
            if (this.navigation === "orbit") this.position = add(this.position, scale(this.forward, event.deltaY * -0.002 * this.radius));
            else this.speed = Math.min(100, Math.max(0.01, this.speed * Math.exp(-event.deltaY * 0.001)));
            this.changed();
        }, {passive: false});
        document.querySelectorAll("[data-key]").forEach(button => {
            button.addEventListener("pointerdown", event => { event.preventDefault(); button.setPointerCapture(event.pointerId); this.keys.add(button.dataset.key); });
            for (const event of ["pointerup", "pointercancel"]) button.addEventListener(event, () => this.keys.delete(button.dataset.key));
        });
    }

    reset(metadata) {
        this.metadata = metadata;
        this.radius = metadata.radius;
        this.movementRadius = navigationRadius(metadata);
        this.target = [...metadata.center];
        this.position = add(this.target, [0, 0, this.radius * 2.5]);
        this.forward = [0, 0, -1]; this.up = [0, 1, 0]; this.fov = 60;
        if (metadata.cameras?.length) this.setView(metadata.cameras[0]);
        this.changed();
    }

    setView(camera) {
        this.position = [...camera.position];
        this.forward = normalize(camera.forward);
        this.up = normalize(camera.up);
        if (Math.abs(dot(this.forward, this.up)) > 0.999) throw new Error("Camera forward and up cannot be parallel");
        this.fov = camera.fov || 60;
        this.changed();
    }

    look(horizontal, vertical, orbit = false) {
        if (orbit) this.forward = normalize(add(this.target, scale(this.position, -1)));
        const distance = Math.max(0.2, Math.hypot(...add(this.position, scale(this.target, -1))));
        this.forward = normalize(rotate(this.forward, this.up, -horizontal * 0.003));
        const right = normalize(cross(this.forward, this.up));
        const candidate = normalize(rotate(this.forward, right, -vertical * 0.003));
        if (Math.abs(dot(candidate, this.up)) < 0.995) this.forward = candidate;
        if (orbit) this.position = add(this.target, scale(this.forward, -distance));
        this.changed();
    }

    pointer(event) {
        if (document.pointerLockElement === this.canvas) { this.look(event.movementX, event.movementY); return; }
        const previous = this.pointers.get(event.pointerId);
        if (!previous) return;
        const current = [event.clientX, event.clientY];
        const horizontal = current[0] - previous[0], vertical = current[1] - previous[1];
        if (this.pointers.size === 2) {
            const other = [...this.pointers.entries()].find(([id]) => id !== event.pointerId)[1];
            const before = Math.hypot(previous[0] - other[0], previous[1] - other[1]);
            const after = Math.hypot(current[0] - other[0], current[1] - other[1]);
            const right = normalize(cross(this.forward, this.up));
            this.position = add(this.position, add(scale(this.forward, (after - before) * 0.005 * this.radius), add(scale(right, -horizontal * this.radius * 0.001), scale(this.up, vertical * this.radius * 0.001))));
            this.changed();
        } else this.look(horizontal, vertical, this.navigation === "orbit");
        this.pointers.set(event.pointerId, current);
    }

    tick(seconds) {
        const forward = Number(this.keys.has("KeyW")) - Number(this.keys.has("KeyS"));
        const sideways = Number(this.keys.has("KeyD")) - Number(this.keys.has("KeyA"));
        const vertical = Number(this.keys.has("KeyE") || this.keys.has("Space")) - Number(this.keys.has("KeyQ"));
        if (!(forward || sideways || vertical)) return;
        const right = normalize(cross(this.forward, this.up));
        const movement = normalize(add(add(scale(this.forward, forward), scale(right, sideways)), scale(this.up, vertical)));
        const boost = this.keys.has("ShiftLeft") || this.keys.has("ShiftRight") ? 4 : 1;
        const delta = scale(movement, seconds * this.speed * this.movementRadius * 0.5 * boost);
        this.position = add(this.position, delta);
        this.target = add(this.target, delta);
        this.changed();
    }

    camera() {
        return {position: [...this.position], forward: [...this.forward], up: [...this.up], fov: this.fov, view: viewMatrix(this.position, this.forward, this.up)};
    }
}
