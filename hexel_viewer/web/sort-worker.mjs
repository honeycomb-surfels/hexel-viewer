import {sortDepths} from "./core.mjs";

let positions;
self.onmessage = ({data}) => {
    if (data.positions) { positions = data.positions; return; }
    const started = performance.now();
    const order = sortDepths(positions, data.forward, data.position, data.budget);
    self.postMessage({order, milliseconds: performance.now() - started, generation: data.generation}, [order.buffer]);
};
