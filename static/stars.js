// Draws the scattered white "stars" once (no per-frame work, so it never slows the camera).
(function () {
  // small seeded random generator so the sky looks the same after every resize
  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function paint(canvas, count, seed, minR, maxR, minA, maxA, glow) {
    const w = (canvas.width = window.innerWidth);
    const h = (canvas.height = window.innerHeight);
    const ctx = canvas.getContext("2d");
    const rand = rng(seed);
    ctx.clearRect(0, 0, w, h);
    for (let i = 0; i < count; i++) {
      const x = rand() * w, y = rand() * h;
      const r = minR + rand() * (maxR - minR);
      const a = minA + rand() * (maxA - minA);
      ctx.fillStyle = `rgba(255, 255, 255, ${a})`;
      if (glow) { ctx.shadowColor = "rgba(200, 215, 255, 0.9)"; ctx.shadowBlur = 6; }
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function render() {
    const area = window.innerWidth * window.innerHeight;
    const dim = Math.round(area / 9000);      // lots of faint stars
    const bright = Math.round(area / 45000);  // a few bright, twinkling ones
    paint(document.getElementById("stars-a"), dim, 11, 0.4, 1.1, 0.25, 0.75, false);
    paint(document.getElementById("stars-b"), bright, 29, 0.9, 1.7, 0.6, 1, true);
  }

  render();
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(render, 150); });
})();