export class FPSController {
  constructor(targetFPS = 30) {
    this.targetFPS = targetFPS;
    this.frameInterval = 1000 / targetFPS;
    this.lastFrameTime = 0;
    this.isRunning = false;
    this.animationId = null;
  }

  start(callback) {
    this.isRunning = true;
    this.lastFrameTime = performance.now();

    const loop = (currentTime) => {
      if (!this.isRunning) return;

      const deltaTime = currentTime - this.lastFrameTime;

      if (deltaTime >= this.frameInterval) {
        // Call the callback function (your frame capture logic)
        callback();

        // Update last frame time, accounting for any drift
        this.lastFrameTime = currentTime - (deltaTime % this.frameInterval);
      }

      this.animationId = requestAnimationFrame(loop);
    };

    this.animationId = requestAnimationFrame(loop);
  }

  stop() {
    this.isRunning = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }
}
