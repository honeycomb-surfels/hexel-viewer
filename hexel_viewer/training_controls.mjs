const token = new URLSearchParams(location.hash.slice(1)).get("token") || "";
const headers = {Authorization: `Bearer ${token}`, "Content-Type": "application/json"};
const panel = document.createElement("section");
panel.innerHTML = `<div class="section-label">TRAINING</div>
  <div class="button-row"><button id="training-start">Start training</button><button id="training-pause">Pause</button></div>
  <p id="training-state" role="status">Connecting to training controls...</p>
  <p class="note">Recording starts automatically with training. Your camera does not affect the movie. Pause takes effect between optimizer iterations. This is not a timing benchmark.</p>`;
document.getElementById("intro").after(panel);
const start = document.getElementById("training-start");
const pause = document.getElementById("training-pause");
const status = document.getElementById("training-state");
let currentState = "waiting";

async function update(action) {
    try {
        const response = await fetch("api/training", action
            ? {method: "POST", headers, body: JSON.stringify({action})}
            : {headers});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error);
        currentState = result.state;
        start.disabled = currentState !== "waiting";
        pause.disabled = !["running", "paused"].includes(currentState);
        pause.textContent = currentState === "paused" ? "Resume" : "Pause";
        status.textContent = `${currentState} · step ${result.iteration.toLocaleString()} · ${result.primitive_count.toLocaleString()} primitives`;
    } catch (error) {
        status.textContent = error.message;
    }
}

start.onclick = () => update("start");
pause.onclick = () => update(currentState === "paused" ? "resume" : "pause");
update();
setInterval(() => update(), 1000);
