export const patienceMessages = [
    "Good things come to those who wait.",
    "Rome wasn't built in a day.",
    "A little patience goes a long way.",
    "All in good time.",
];

export const nextPaint = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

export class LoadingNotice {
    constructor(root, setStatus) {
        this.root = root;
        this.message = root.querySelector(".loading-message");
        this.stageLabel = root.querySelector(".loading-stage");
        this.setStatus = setStatus;
    }

    get active() { return !this.root.hidden; }

    start(stage) {
        this.stop();
        this.root.hidden = false;
        this.index = 0;
        this.message.textContent = patienceMessages[0];
        this.stage(stage);
        this.interval = setInterval(() => {
            this.message.classList.add("changing");
            this.transition = setTimeout(() => {
                this.index = (this.index + 1) % patienceMessages.length;
                this.message.textContent = patienceMessages[this.index];
                this.message.classList.remove("changing");
            }, 220);
        }, 5000);
    }

    stage(message) {
        this.stageLabel.textContent = message;
        this.setStatus(message);
    }

    stop() {
        clearInterval(this.interval);
        clearTimeout(this.transition);
        this.message.classList.remove("changing");
        this.root.hidden = true;
    }
}
