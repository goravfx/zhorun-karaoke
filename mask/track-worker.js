// Поиск лица в отдельном потоке: основной поток рисует кадр и пишет видео, а здесь работает MediaPipe.
/* global importScripts */
let fl = null, delegate = "CPU", make = null, lastTs = 0;
self.exports = {};
self.module = { exports: self.exports };

async function init(m){
  importScripts(m.bundle);
  const lib = self.module.exports.FaceLandmarker ? self.module.exports : self.exports;
  const files = { wasmLoaderPath: m.loader, wasmBinaryPath: m.wasm };
  const model = new Uint8Array(m.model);
  make = d => lib.FaceLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetBuffer: model, delegate: d },
    runningMode: "VIDEO", numFaces: 1,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false,
  });
  delegate = m.delegate;
  try { fl = await make(delegate); }
  catch (e) { if (delegate === "CPU") throw e; delegate = "CPU"; fl = await make("CPU"); }
  // прогрев: первый кадр собирает шейдеры и графы — пусть это случится до съёмки
  const warm = () => { const c = new OffscreenCanvas(64, 64); c.getContext("2d").fillRect(0, 0, 64, 64); fl.detectForVideo(c, ++lastTs); };
  try { warm(); }
  catch (e) {
    if (delegate !== "GPU") throw e;
    try { fl.close(); } catch (e2) {}
    delegate = "CPU"; fl = await make("CPU"); warm();
  }
}

self.onmessage = async e => {
  const m = e.data;
  if (m.type === "init") {
    try { await init(m); self.postMessage({ type: "ready", delegate }); }
    catch (err) { self.postMessage({ type: "error", error: String(err && err.message || err) }); }
    return;
  }
  if (m.type === "frame") {
    const t0 = performance.now();
    let lm = null;
    try {
      lastTs = Math.max(lastTs + 1, m.ts);
      const r = fl.detectForVideo(m.bmp, lastTs);
      const f = r && r.faceLandmarks && r.faceLandmarks[0];
      if (f) {
        lm = new Float32Array(f.length * 3);
        for (let i = 0; i < f.length; i++) { lm[i * 3] = f[i].x; lm[i * 3 + 1] = f[i].y; lm[i * 3 + 2] = f[i].z; }
      }
    } catch (err) {
      self.postMessage({ type: "result", id: m.id, lm: null, ms: performance.now() - t0, error: String(err && err.message || err) });
      try { m.bmp.close(); } catch (e2) {}
      return;
    }
    try { m.bmp.close(); } catch (e2) {}
    self.postMessage({ type: "result", id: m.id, lm, ms: performance.now() - t0 }, lm ? [lm.buffer] : []);
  }
};
