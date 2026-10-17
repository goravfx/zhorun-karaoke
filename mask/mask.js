// Маска «Жорун»: мокрый пампадур с локоном и усы-подкова с бакенбардами.
// Лицо ищет MediaPipe Face Landmarker, причёска и усы рисуются в WebGL2
// поверх кадра камеры, усы и бакенбарды привязаны к точкам лица и двигаются с губами.
import { CANON_V, CANON_F } from "./face-canon.js";

const BASE = new URL("./", import.meta.url).href;
const DEG = Math.PI / 180;

/* ---------------- маленькая математика */
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = a => Math.hypot(a[0], a[1], a[2]);
const nrm = a => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
function interp(xs, ys, x){
  if (x <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++) if (x <= xs[i]) return lerp(ys[i - 1], ys[i], (x - xs[i - 1]) / (xs[i] - xs[i - 1]));
  return ys[ys.length - 1];
}
function rng(seed){
  let a = seed >>> 0;
  return () => { a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
const cv = i => [CANON_V[i * 3], CANON_V[i * 3 + 1], CANON_V[i * 3 + 2]];

/* ---------------- голова: эллипсоид черепа в координатах канонического лица (см) */
const HC = [0, 3.0, -4.6], HA = [7.9, 10.6, 9.9];
const TF = 15 * DEG;   // ось «зачёса»: от лба назад к затылку
const FP = [0, Math.sin(TF), Math.cos(TF)], E1 = [0, Math.cos(TF), -Math.sin(TF)], E2 = [1, 0, 0];
function dirLM(l, m){ // градусы → направление из центра черепа
  const sm = Math.sin(m * DEG), cm = Math.cos(m * DEG), cl = Math.cos(l * DEG), sl = Math.sin(l * DEG);
  return [FP[0] * cm + sm * (E1[0] * cl + E2[0] * sl), FP[1] * cm + sm * (E1[1] * cl + E2[1] * sl), FP[2] * cm + sm * (E1[2] * cl + E2[2] * sl)];
}
const lmOf = d => [Math.atan2(dot(d, E2), dot(d, E1)) / DEG, Math.acos(clamp(dot(d, FP), -1, 1)) / DEG];
const rE = d => 1 / Math.hypot(d[0] / HA[0], d[1] / HA[1], d[2] / HA[2]);

function faceNormals(){
  const n = new Float32Array(468 * 3);
  for (let f = 0; f < CANON_F.length; f += 3) {
    const ids = [CANON_F[f], CANON_F[f + 1], CANON_F[f + 2]];
    const [A, B, C] = ids.map(cv);
    let fn = cross(sub(B, A), sub(C, A));
    const cen = mul(add(add(A, B), C), 1 / 3);
    if (dot(fn, sub(cen, [0, 0, -3])) < 0) fn = mul(fn, -1);
    for (const i of ids) { n[i * 3] += fn[0]; n[i * 3 + 1] += fn[1]; n[i * 3 + 2] += fn[2]; }
  }
  for (let i = 0; i < 468; i++) n.set(nrm([n[i * 3], n[i * 3 + 1], n[i * 3 + 2]]), i * 3);
  return n;
}

/* ================= ГРУМ: ПАМПАДУР (жёстко сидит на голове) ================= */
function buildPompadour(Q){
  const rnd = rng(7);
  // линия роста волос — по верхнему контуру лица
  const hl = [10, 338, 297, 332, 284, 251, 389, 356].map(i => lmOf(nrm(sub(cv(i), HC)))).map(([l, m]) => [Math.abs(l), m]);
  hl.sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < hl.length; i++) hl[i][1] = Math.max(hl[i][1], hl[i - 1][1] + 0.2);
  hl[0][0] = 0;
  const HL_L = hl.map(p => p[0]), HL_M = hl.map(p => p[1]);
  // граница области волос: |λ| ≤ λmax(μ) — спереди линия роста, сбоку над ухом и вниз к затылку
  const MU_T = [...HL_M, 80, 88, 95, 108, 120, 147.5, 181], LA_T = [...HL_L, 104, 97, 86, 95, 125, 180, 180];
  const lamMax = m => m < MU_T[0] ? -1 : interp(MU_T, LA_T, m);
  const inside = (l, m) => Math.abs(l) <= lamMax(m);
  const muFront = al => interp(HL_L, HL_M, al);
  const jit = l => 1.3 * Math.sin(l * 0.29 + 1.1) + 0.8 * Math.sin(l * 0.83 + 2.3) + 0.45 * Math.sin(l * 1.9 + 0.4);
  const muHair = l => muFront(Math.abs(l)) + jit(l) + 0.6;
  const bnd = [];
  for (let m = MU_T[0]; m <= 147.5; m += 0.5) { const lb = lamMax(m); bnd.push(dirLM(lb, m), dirLM(-lb, m)); }
  const edge = d => { let b = -2; for (const q of bnd) { const c = dot(d, q); if (c > b) b = c; } return Math.acos(Math.min(1, b)) / DEG; };
  // толщина причёски над черепом: гладкие бока + высокий кок над лбом
  const thick = (l, m, e) => {
    const al = Math.abs(l);
    const h0 = 1.05 - 0.3 * smooth(80, 150, m);
    const dF = m - muFront(al);
    const gq = Math.exp(-Math.pow(al / 60, 4));
    const fq = smooth(0, 12, dF) * Math.exp(-Math.pow(Math.max(0, dF - 16) / 36, 2));
    const dH = al < 99 ? m - muHair(l) : 99;
    return (h0 + 4.5 * gq * fq) * smooth(0, 3.0, e) * smooth(-0.5, 3.5, dH);
  };
  const NL = 120, LS = 3, M0 = 12, MS = 2.5, NM = Math.round((180 - M0) / MS) + 1;
  const gr = new Float32Array(NL * NM), ge = new Float32Array(NL * NM);
  for (let j = 0; j < NM; j++) for (let i = 0; i < NL; i++) {
    const l = -180 + i * LS, m = Math.min(179.8, M0 + j * MS);
    const d = dirLM(l, m);
    const e = edge(d) * (inside(l, m) ? 1 : -1);
    ge[j * NL + i] = e;
    gr[j * NL + i] = rE(d) + thick(l, m, e);
  }
  const sampleR = (l, m) => {
    const fi = (l + 180) / LS, fj = clamp((m - M0) / MS, 0, NM - 1.001);
    const i0 = Math.floor(fi), j0 = Math.floor(fj), u = fi - i0, v = fj - j0;
    const ia = ((i0 % NL) + NL) % NL, ib = (ia + 1) % NL;
    const a = gr[j0 * NL + ia], b = gr[j0 * NL + ib], c = gr[(j0 + 1) * NL + ia], d = gr[(j0 + 1) * NL + ib];
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
  };
  const shellP = (l, m, lift = 0) => { const d = dirLM(l, m); return add(HC, mul(d, sampleR(l, m) + lift)); };
  const shellN = (l, m) => {
    let n = nrm(cross(sub(shellP(l + 1, m), shellP(l - 1, m)), sub(shellP(l, m + 1), shellP(l, m - 1))));
    if (dot(n, dirLM(l, m)) < 0) n = mul(n, -1);
    return n;
  };

  /* оболочка под прядями: тёмная масса, чтобы не просвечивали свои волосы */
  const shell = { p: [], n: [], t: [], a: [], idx: [] };
  for (let j = 0; j < NM; j++) for (let i = 0; i < NL; i++) {
    const l = -180 + i * LS, m = Math.min(179.8, M0 + j * MS);
    const e = ge[j * NL + i];
    shell.p.push(...shellP(l, m, -0.12));
    shell.n.push(...shellN(l, m));
    shell.t.push(...nrm(sub(shellP(l, m + 1), shellP(l, m - 1))));
    shell.a.push(smooth(-0.8, 1.6, e) * (Math.abs(l) < 99 ? smooth(0.5, 3.0, m - muHair(l)) : 1));
  }
  for (let j = 0; j < NM - 1; j++) for (let i = 0; i < NL; i++) {
    const a = j * NL + i, b = j * NL + (i + 1) % NL, c = (j + 1) * NL + i, d = (j + 1) * NL + (i + 1) % NL;
    if (Math.max(shell.a[a], shell.a[b], shell.a[c], shell.a[d]) <= 0.001) continue;
    shell.idx.push(a, c, b, b, c, d);
  }

  /* пряди: идут вдоль меридианов от лба к затылку, собираются в мокрые пучки */
  const guides = [];
  while (guides.length < 230) {
    const d = nrm([rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1]);
    const [l, m] = lmOf(d);
    if (!inside(l, m)) continue;
    const big = rnd() < 0.22;
    guides.push({ l, m, d, lift: 0.04 + 0.22 * rnd() + (big ? 0.3 + 0.4 * rnd() : 0), cl: 0.72 + 0.26 * rnd(), ph: rnd() * 6.283, rnd: rnd(), stray: rnd() < 0.035 });
  }
  const nearestGuide = (l, m) => {
    let best = null, bd = 1e9;
    for (const g of guides) {
      const dl = (l - g.l) * Math.sin(Math.max(0.2, m * DEG)), dm = (m - g.m) * 0.35;
      const dd = dl * dl + dm * dm;
      if (dd < bd) { bd = dd; best = g; }
    }
    return best;
  };
  const strands = [];
  const trace = (l0, m0, L, w) => {
    const g = nearestGuide(l0, m0);
    const pts = [], nr = [];
    const lift0 = 0.02 + 0.09 * rnd();
    let s = 0, m = m0;
    const step = 0.7;
    for (let guard = 0; guard < 60; guard++) {
      const u = Math.min(1, s / L);
      const k = g.cl * smooth(0, 0.7, u);
      const l = l0 + (g.l - l0) * k + 0.7 * Math.sin(m * 0.09 + g.ph) * u;
      const lift = lerp(lift0, g.lift, k) + 0.05 * Math.sin(m * 0.23 + g.ph * 2) + (g.stray ? 0.45 * u * u : 0);
      if (pts.length && !inside(l, m)) break;
      const P = shellP(l, m, lift);
      pts.push(P); nr.push(shellN(l, m));
      if (s >= L || m > 178) break;
      let dm = step / (sampleR(l, m) + lift) / DEG;
      const dd = len(sub(shellP(l, m + dm, lift), P));
      dm *= step / Math.max(0.15, dd);
      m += dm; s += step;
    }
    if (pts.length >= 3) strands.push({ pts, nr, w, rnd: lerp(g.rnd, rnd(), 0.3) });
  };
  const nFront = Math.round(300 * Q), nBody = Math.round(3300 * Q);
  for (let i = 0; i < nFront; i++) {
    const l = -96 + 192 * (i + rnd() * 0.8) / nFront;
    trace(l, muHair(l) + 0.6 + rnd() * 1.4, 12 + 4 * rnd(), 0.8 + 0.5 * rnd());
  }
  for (let i = 0; i < nBody; i++) {
    let l, m;
    for (;;) {
      const d = nrm([rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1]);
      [l, m] = lmOf(d);
      if (inside(l, m) && m > muFront(Math.min(99, Math.abs(l))) + 1.2) break;
    }
    trace(l, m, 7 + 5 * rnd(), 0.7 + 0.6 * rnd());
  }

  /* локон, который падает на лоб */
  const curl = (l, m, ctrl, n, r0) => {
    const P0 = shellP(l, m, 0.2);
    const cp = [P0, add(P0, ctrl[0]), ...ctrl.slice(1)];
    const path = catmull(cp, 34);
    for (let k = 0; k < n; k++) {
      const ph = rnd() * 6.283, rr = r0 * Math.sqrt(rnd()), tw = 0.6 + rnd();
      const pts = [], nr = [];
      for (let i = 0; i < path.length; i++) {
        const u = i / (path.length - 1);
        const T = nrm(sub(path[Math.min(path.length - 1, i + 1)], path[Math.max(0, i - 1)]));
        const B1 = nrm(cross(T, [0, 0, 1])), B2 = cross(T, B1);
        const a = ph + u * tw * 2.4, r = rr * (1 - 0.5 * u);
        pts.push(add(path[i], add(mul(B1, Math.cos(a) * r), mul(B2, Math.sin(a) * r))));
        nr.push(nrm(add(mul(B2, -1), [0, 0, 0.6])));
      }
      strands.push({ pts, nr, w: 0.8 + 0.4 * rnd(), rnd: rnd() });
    }
  };
  curl(4, 30, [[0.4, -0.3, 1.2], [1.6, 12.0, 8.0], [2.4, 9.7, 7.6], [2.1, 7.5, 6.95], [1.0, 6.0, 6.55], [0.25, 5.05, 6.4], [0.35, 4.45, 6.3]], Math.round(140 * Q + 12), 0.88);
  curl(-6, 28, [[-0.3, -0.3, 1.0], [-1.0, 11.6, 7.8], [-1.6, 9.4, 7.3], [-1.1, 7.7, 6.8]], Math.round(45 * Q + 6), 0.5);
  curl(14, 30, [[0.3, -0.3, 0.9], [2.9, 12.0, 7.6], [3.5, 9.9, 7.0], [3.2, 8.8, 6.6]], Math.round(18 * Q + 4), 0.25);
  return { shell, strands };
}

function catmull(P, n){
  const out = [];
  const seg = P.length - 1;
  for (let s = 0; s < seg; s++) {
    const p0 = P[Math.max(0, s - 1)], p1 = P[s], p2 = P[s + 1], p3 = P[Math.min(P.length - 1, s + 2)];
    const steps = Math.max(2, Math.round(n / seg));
    for (let i = 0; i < steps; i++) {
      const t = i / steps, t2 = t * t, t3 = t2 * t;
      out.push([0, 1, 2].map(k => 0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)));
    }
  }
  out.push(P[P.length - 1]);
  return out;
}

/* ================= ГРУМ: УСЫ-ПОДКОВА И БАКЕНБАРДЫ (на точках лица) ================= */
const NOSE_X = [0, 0.6, 1.41, 1.9, 2.5, 3.0, 3.45], NOSE_Y = [-2.2, -2.12, -1.86, -1.98, -2.3, -2.75, -3.25];
const ULIP_X = [0, 0.71, 1.43, 1.91, 2.29, 2.46], ULIP_Y = [-3.41, -3.33, -3.5, -3.8, -4.05, -4.34];
const LLIP_X = [0, 0.7, 1.33, 1.84, 2.2, 2.46], LLIP_Y = [-5.37, -5.29, -5.11, -4.83, -4.6, -4.34];
const SB_Y = [-2.89, -1.05, 0.67, 2.36, 4.11, 5.43], SB_X = [7.27, 7.54, 7.66, 7.74, 7.56, 7.09];
function inMouth(ax, y){ return ax < 2.46 && y < interp(ULIP_X, ULIP_Y, ax) + 0.02 && y > interp(LLIP_X, LLIP_Y, ax) - 0.02; }
function inMustache(x, y){
  const ax = Math.abs(x);
  if (inMouth(ax, y)) return false;
  if (ax <= 3.45 && y <= interp(NOSE_X, NOSE_Y, ax) && y >= interp(ULIP_X, ULIP_Y, Math.min(ax, 2.46)) - (ax > 2.46 ? 0.6 : 0.0)) return true;
  if (y <= -3.2 && y >= -8.0) {
    const xc = 3.0 + 0.04 * (y + 4.4);
    const hw = 0.7 * (0.3 + 0.7 * smooth(-8.05, -6.7, y));
    if (Math.abs(ax - xc) <= hw) return true;
  }
  return false;
}
function sideburnIn(x, y, z){ // насколько точка внутри бакенбарда, см (<0 — снаружи)
  if (z > 2.6) return -1;
  const w = lerp(1.25, 1.55, smooth(-1.8, 4.5, y));
  return Math.min(Math.abs(x) - (interp(SB_Y, SB_X, y) - w), y + 1.8, 6.2 - y);
}
const inSideburn = (x, y, z) => sideburnIn(x, y, z) >= 0;

function buildFace(Q){
  const rnd = rng(11);
  const N = faceNormals();
  const tri = [];
  for (let f = 0; f < CANON_F.length; f += 3) {
    const ids = [CANON_F[f], CANON_F[f + 1], CANON_F[f + 2]];
    const [A, B, C] = ids.map(cv);
    tri.push({ ids, A, B, C, area: len(cross(sub(B, A), sub(C, A))) / 2 });
  }
  // мягкая маска области для «подложки» на коже
  const weight = (test) => {
    const w = new Float32Array(468);
    for (let i = 0; i < 468; i++) {
      const p = cv(i); let k = 0;
      for (let s = 0; s < 25; s++) { const a = s * 2.4, r = 0.32 * Math.sqrt(s / 24); if (test(p[0] + Math.cos(a) * r, p[1] + Math.sin(a) * r, p[2])) k++; }
      w[i] = k / 25;
    }
    return w;
  };
  const region = (test, count, grow) => {
    const cand = tri.filter(t => [t.A, t.B, t.C, mul(add(add(t.A, t.B), t.C), 1 / 3)].some(p => test(p[0], p[1], p[2])) );
    const total = cand.reduce((s, t) => s + t.area, 0);
    const out = [];
    let guard = 0;
    while (out.length < count && guard++ < count * 40) {
      let r = rnd() * total, t = cand[0];
      for (const c of cand) { r -= c.area; if (r <= 0) { t = c; break; } }
      let u = rnd(), v = rnd(); if (u + v > 1) { u = 1 - u; v = 1 - v; }
      const p = add(add(mul(t.A, 1 - u - v), mul(t.B, u)), mul(t.C, v));
      if (!test(p[0], p[1], p[2], rnd)) continue;
      const n = nrm(add(add(mul(cv3(N, t.ids[0]), 1 - u - v), mul(cv3(N, t.ids[1]), u)), mul(cv3(N, t.ids[2]), v)));
      const pts = grow(add(p, mul(n, 0.01)), n, p);
      out.push({ tri: t.ids, bary: [u, v], root: pts[0], pts, n, w: 0.75 + 0.5 * rnd(), rnd: rnd() });
    }
    return out;
  };
  const down = [0, -1, 0];
  const growMustache = (root, n, p) => {
    const ax = Math.abs(p[0]), sx = Math.sign(p[0]) || 1;
    const dt = nrm(sub(down, mul(n, dot(down, n))));
    const upper = p[1] > -3.6 && ax < 3.3;
    let d = upper
      ? nrm(add(add(mul(n, 0.5), mul(dt, 0.85)), [sx * 0.42 * Math.min(1, ax / 2.6), 0, 0]))
      : nrm(add(add(mul(n, 0.42), mul(dt, 0.9)), [sx * 0.1, 0, 0]));
    d = nrm(add(d, [(rnd() - 0.5) * 0.3, (rnd() - 0.5) * 0.15, (rnd() - 0.5) * 0.2]));
    const L = upper ? (1.25 + 0.55 * rnd()) * (1 - 0.25 * smooth(2.0, 3.3, ax)) : 0.95 + 0.4 * rnd();
    const pts = [root];
    for (let k = 1; k <= 6; k++) { d = nrm(add(d, mul([0, -1, 0.04], 0.17))); pts.push(add(pts[k - 1], mul(d, L / 6))); }
    return pts;
  };
  const growSideburn = (root, n) => {
    const dt = nrm(sub(down, mul(n, dot(down, n))));
    let d = nrm(add(add(mul(n, 0.28), mul(dt, 0.95)), [(rnd() - 0.5) * 0.2, 0, (rnd() - 0.5) * 0.2]));
    const L = 0.8 + 0.6 * rnd(), pts = [root];
    for (let k = 1; k <= 5; k++) { d = nrm(add(d, mul(dt, 0.12))); pts.push(add(pts[k - 1], mul(d, L / 5))); }
    return pts;
  };
  const growBaby = (root, n, p) => {
    const up = [0, 1, 0.15];
    const dt = nrm(sub(up, mul(n, dot(up, n))));
    let d = nrm(add(add(mul(n, 0.3), dt), [(rnd() - 0.5) * 0.5, 0, 0]));
    const L = 0.3 + 0.45 * rnd(), pts = [root];
    for (let k = 1; k <= 4; k++) { d = nrm(add(d, mul(dt, 0.2))); pts.push(add(pts[k - 1], mul(d, L / 4))); }
    return pts;
  };
  const babies = region((x, y, z) => y > 7.7 + 0.07 * Math.abs(x) && z > -0.5, Math.round(170 * Q), growBaby);
  const mustache = region((x, y) => inMustache(x, y), Math.round(2300 * Q), growMustache);
  const sideburns = region((x, y, z, rr) => { const d = sideburnIn(x, y, z); return d >= 0 && (!rr || rr() < smooth(0, 0.45, d)); }, Math.round(900 * Q), growSideburn);
  const base = (w, off) => {
    const idx = [];
    for (const t of tri) if (Math.max(w[t.ids[0]], w[t.ids[1]], w[t.ids[2]]) > 0.02) idx.push(...t.ids);
    return { w, idx, off };
  };
  return {
    N, mustache, sideburns, babies,
    mBase: base(weight((x, y) => inMustache(x, y)), 0.05),
    sBase: base(weight(inSideburn), 0.04),
  };
}
const cv3 = (arr, i) => [arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]];

/* ---------------- упаковка прядей в ленты для GPU */
function packStrands(list, bound){
  let nv = 0, ni = 0;
  for (const s of list) { nv += s.pts.length * 2; ni += (s.pts.length - 1) * 6; }
  const P = new Float32Array(nv * 3), T = new Float32Array(nv * 3), Nn = new Float32Array(nv * 3), I = new Float32Array(nv * 4);
  const TR = bound ? new Uint16Array(nv * 3) : null, BA = bound ? new Float32Array(nv * 2) : null, RO = bound ? new Float32Array(nv * 3) : null;
  const idx = new Uint32Array(ni);
  let v = 0, ii = 0;
  for (const s of list) {
    const n = s.pts.length;
    for (let k = 0; k < n; k++) {
      const tan = nrm(sub(s.pts[Math.min(n - 1, k + 1)], s.pts[Math.max(0, k - 1)]));
      const nr = s.nr ? s.nr[k] : s.n;
      for (const side of [-1, 1]) {
        P.set(s.pts[k], v * 3); T.set(tan, v * 3); Nn.set(nr, v * 3);
        I.set([k / (n - 1), side, s.rnd, s.w], v * 4);
        if (bound) { TR.set(s.tri, v * 3); BA.set(s.bary, v * 2); RO.set(s.root, v * 3); }
        v++;
      }
      if (k < n - 1) {
        const a = v - 2;
        idx.set([a, a + 1, a + 2, a + 1, a + 3, a + 2], ii); ii += 6;
      }
    }
  }
  return { P, T, N: Nn, I, TR, BA, RO, idx };
}

/* ---------------- шейдеры */
const VHEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D u_lm; uniform mat3 u_R; uniform float u_s; uniform vec3 u_t;
uniform vec4 u_view; uniform vec2 u_canvas; uniform float u_zc; uniform int u_bound;
vec3 lm(uint i){ return texelFetch(u_lm, ivec2(int(i), 0), 0).xyz; }
vec4 clipOf(vec2 c, float z){ return vec4(c.x / u_canvas.x * 2.0 - 1.0, 1.0 - c.y / u_canvas.y * 2.0, (z - u_zc) * u_view.w, 1.0); }
`;
const VS_SURF = VHEAD + `
layout(location=0) in vec3 a_p; layout(location=1) in vec3 a_n; layout(location=2) in vec3 a_t;
layout(location=4) in float a_a; layout(location=5) in uvec3 a_i;
uniform float u_off;
out vec3 v_n; out vec3 v_t; out float v_a;
void main(){
  vec3 N = normalize(u_R * a_n);
  vec3 P = u_bound == 1 ? lm(a_i.x) + N * (u_off * u_s) : u_s * (u_R * a_p) + u_t;
  v_n = N; v_t = normalize(u_R * a_t); v_a = a_a;
  gl_Position = clipOf(u_view.xy + P.xy * u_view.z, P.z);
}`;
const FS_SURF = `#version 300 es
precision highp float;
uniform vec3 u_L; uniform vec3 u_col; uniform vec3 u_spec; uniform float u_pow; uniform float u_exp; uniform float u_alpha;
in vec3 v_n; in vec3 v_t; in float v_a; out vec4 o;
void main(){
  float a = v_a * u_alpha;
  if (a < 0.02) discard;
  vec3 N = normalize(v_n), T = normalize(v_t), H = normalize(u_L + vec3(0.0, 0.0, -1.0));
  float nl = max(0.0, dot(N, u_L)) * 0.7 + 0.3;
  float th = dot(T, H);
  float kk = pow(sqrt(max(0.0, 1.0 - th * th)), u_pow) * smoothstep(-0.1, 0.4, dot(N, H));
  vec3 c = (u_col * nl + u_spec * kk) * u_exp;
  o = vec4(c * a, a);
}`;
const VS_STRAND = VHEAD + `
layout(location=0) in vec3 a_p; layout(location=1) in vec3 a_n; layout(location=2) in vec3 a_t;
layout(location=3) in vec4 a_info; layout(location=5) in uvec3 a_i; layout(location=6) in vec2 a_bary; layout(location=7) in vec3 a_root;
uniform float u_wcm; uniform float u_wmin; uniform float u_taper;
out vec3 v_T; out vec3 v_N; out float v_t; out float v_rnd;
void main(){
  vec3 P;
  if (u_bound == 1) {
    vec3 r = lm(a_i.x) * (1.0 - a_bary.x - a_bary.y) + lm(a_i.y) * a_bary.x + lm(a_i.z) * a_bary.y;
    P = r + u_s * (u_R * (a_p - a_root));
  } else {
    P = u_s * (u_R * a_p) + u_t;
  }
  vec3 T = normalize(u_R * a_t), N = normalize(u_R * a_n);
  vec2 c = u_view.xy + P.xy * u_view.z;
  vec2 td = T.xy; float tl = length(td);
  vec2 dir = tl > 1e-5 ? td / tl : vec2(0.0, 1.0);
  float w = u_wcm * a_info.w * u_s * u_view.z;
  w = clamp(w, u_wmin, 1.9) * (1.0 - u_taper * a_info.x);
  c += vec2(-dir.y, dir.x) * (a_info.y * 0.5 * w);
  gl_Position = clipOf(c, P.z);
  v_T = T; v_N = N; v_t = a_info.x; v_rnd = a_info.z;
}`;
const FS_STRAND = `#version 300 es
precision highp float;
uniform vec3 u_L; uniform vec3 u_base; uniform vec3 u_tip; uniform vec3 u_s1; uniform vec3 u_s2; uniform vec4 u_sh; uniform float u_exp;
in vec3 v_T; in vec3 v_N; in float v_t; in float v_rnd; out vec4 o;
float kk(vec3 T, vec3 H, float p){ float h = dot(T, H); return pow(sqrt(max(0.0, 1.0 - h * h)), p); }
void main(){
  vec3 T = normalize(v_T), N = normalize(v_N), H = normalize(u_L + vec3(0.0, 0.0, -1.0));
  vec3 H2 = normalize(vec3(0.0, -0.12, -1.0));
  float tl = dot(T, u_L);
  float diff = sqrt(max(0.0, 1.0 - tl * tl)) * (0.4 + 0.6 * max(0.0, dot(N, u_L)));
  vec3 T1 = normalize(T + N * (u_sh.z + (v_rnd - 0.5) * 0.36)), T2 = normalize(T + N * u_sh.w);
  float s1 = kk(T1, H, u_sh.x) + 0.3 * kk(T1, H2, u_sh.x), s2 = kk(T2, H, u_sh.y);
  float vis = smoothstep(-0.25, 0.35, dot(N, H));
  vec3 base = mix(u_base, u_tip, v_t) * (0.72 + 0.56 * v_rnd);
  float ao = mix(0.55, 1.0, smoothstep(0.0, 0.35, v_t));
  vec3 c = (base * (0.3 + diff) * ao + (u_s1 * s1 + u_s2 * s2 * (0.5 + v_rnd)) * vis * ao) * u_exp;
  o = vec4(c, 1.0);
}`;

/* ---------------- поза головы: подобие (масштаб, поворот, сдвиг) по устойчивым точкам */
const STABLE = [10, 109, 338, 67, 297, 103, 332, 54, 284, 21, 251, 162, 389, 127, 356, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1,
  33, 133, 362, 263, 226, 446, 130, 359, 116, 345, 117, 346, 118, 347, 123, 352, 50, 280, 205, 425, 36, 266,
  98, 327, 64, 294, 48, 278, 129, 358, 234, 454, 93, 323];
const STABLE_C = (() => { const a = STABLE.map(cv); const c = mul(a.reduce(add, [0, 0, 0]), 1 / a.length); return { a: a.map(p => sub(p, c)), c }; })();
function maxEig4(N){
  const a = N.slice(), v = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 10; sweep++) {
    let off = 0;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 4; q++) off += a[p * 4 + q] * a[p * 4 + q];
    if (off < 1e-20) break;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 4; q++) {
      const apq = a[p * 4 + q];
      if (Math.abs(apq) < 1e-30) continue;
      const th = (a[q * 4 + q] - a[p * 4 + p]) / (2 * apq);
      const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 4; k++) { const x = a[k * 4 + p], y = a[k * 4 + q]; a[k * 4 + p] = c * x - s * y; a[k * 4 + q] = s * x + c * y; }
      for (let k = 0; k < 4; k++) { const x = a[p * 4 + k], y = a[q * 4 + k]; a[p * 4 + k] = c * x - s * y; a[q * 4 + k] = s * x + c * y; }
      for (let k = 0; k < 4; k++) { const x = v[k * 4 + p], y = v[k * 4 + q]; v[k * 4 + p] = c * x - s * y; v[k * 4 + q] = s * x + c * y; }
    }
  }
  let b = 0;
  for (let i = 1; i < 4; i++) if (a[i * 4 + i] > a[b * 4 + b]) b = i;
  return [v[b], v[4 + b], v[8 + b], v[12 + b]];
}
function fitPose(L){ // L: Float32Array 478*3 в пикселях кадра → {s, R(9, по строкам), t}
  const n = STABLE.length;
  let cb = [0, 0, 0];
  const B = STABLE.map(i => [L[i * 3], L[i * 3 + 1], L[i * 3 + 2]]);
  for (const b of B) cb = add(cb, b);
  cb = mul(cb, 1 / n);
  let Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0, aa = 0;
  for (let k = 0; k < n; k++) {
    const a = STABLE_C.a[k], b = sub(B[k], cb);
    Sxx += a[0] * b[0]; Sxy += a[0] * b[1]; Sxz += a[0] * b[2];
    Syx += a[1] * b[0]; Syy += a[1] * b[1]; Syz += a[1] * b[2];
    Szx += a[2] * b[0]; Szy += a[2] * b[1]; Szz += a[2] * b[2];
    aa += dot(a, a);
  }
  const q = maxEig4([
    Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx,
    Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
    Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy,
    Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]);
  const [w, x, y, z] = q;
  const R = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  const Ra = a => [R[0] * a[0] + R[1] * a[1] + R[2] * a[2], R[3] * a[0] + R[4] * a[1] + R[5] * a[2], R[6] * a[0] + R[7] * a[1] + R[8] * a[2]];
  let num = 0;
  for (let k = 0; k < n; k++) num += dot(sub(B[k], cb), Ra(STABLE_C.a[k]));
  const s = num / aa;
  const t = sub(cb, mul(Ra(STABLE_C.c), s));
  return { s, R, t };
}

/* ---------------- сглаживание точек (фильтр One Euro: стоишь — не дрожит, двигаешься — не отстаёт) */
class Smoother {
  constructor(n){ this.n = n; this.x = null; this.dx = new Float32Array(n / 3); this.t = 0; }
  reset(){ this.x = null; }
  apply(raw, t, scale){
    if (!this.x || t - this.t > 0.6) { this.x = raw.slice(); this.dx.fill(0); this.t = t; return this.x; }
    const dt = clamp(t - this.t, 1 / 240, 0.2); this.t = t;
    const al = fc => { const tau = 1 / (2 * Math.PI * fc); return 1 / (1 + tau / dt); };
    const aD = al(1.2);
    for (let i = 0, j = 0; i < this.n; i += 3, j++) {
      const ex = raw[i] - this.x[i], ey = raw[i + 1] - this.x[i + 1], ez = raw[i + 2] - this.x[i + 2];
      const sp = Math.hypot(ex, ey, ez * 0.5) / dt / scale;   // см/с
      this.dx[j] += (sp - this.dx[j]) * aD;
      const a = al(Smoother.MIN + Smoother.BETA * this.dx[j]);
      this.x[i] += ex * a; this.x[i + 1] += ey * a; this.x[i + 2] += ez * a;
    }
    return this.x;
  }
}
Smoother.MIN = 1.6; Smoother.BETA = 0.5;

/* ---------------- WebGL */
function compile(gl, vs, fs){
  const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  const nu = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < nu; i++) { const info = gl.getActiveUniform(p, i); u[info.name] = gl.getUniformLocation(p, info.name); }
  return { p, u };
}
function vao(gl, attrs, idx){
  const o = gl.createVertexArray(); gl.bindVertexArray(o);
  for (const [loc, data, size, int] of attrs) {
    if (!data) continue;
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    if (int) gl.vertexAttribIPointer(loc, size, gl.UNSIGNED_SHORT, 0, 0);
    else gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }
  const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return { o, count: idx.length, type: idx instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT };
}

const LOOK = {
  hair: { base: [0.022, 0.018, 0.016], tip: [0.06, 0.048, 0.04], s1: [0.64, 0.63, 0.62], s2: [0.17, 0.14, 0.11], sh: [110, 22, -0.08, 0.16], wcm: 0.05, wmin: 0.85, taper: 0.7 },
  shell: { col: [0.012, 0.010, 0.009], spec: [0.1, 0.095, 0.09], pow: 30 },
  stache: { base: [0.095, 0.058, 0.036], tip: [0.24, 0.15, 0.09], s1: [0.2, 0.17, 0.14], s2: [0.18, 0.11, 0.06], sh: [60, 14, -0.05, 0.12], wcm: 0.036, wmin: 0.8, taper: 0.75 },
  stacheBase: { col: [0.085, 0.052, 0.034], spec: [0.05, 0.04, 0.03], pow: 20 },
  burns: { base: [0.03, 0.025, 0.021], tip: [0.075, 0.058, 0.046], s1: [0.2, 0.19, 0.18], s2: [0.12, 0.08, 0.05], sh: [50, 12, -0.05, 0.12], wcm: 0.034, wmin: 0.8, taper: 0.7 },
  babies: { base: [0.03, 0.025, 0.021], tip: [0.07, 0.055, 0.045], s1: [0.12, 0.12, 0.12], s2: [0.06, 0.05, 0.04], sh: [40, 12, -0.05, 0.12], wcm: 0.018, wmin: 0.45, taper: 0.8 },
  burnsBase: { col: [0.035, 0.028, 0.024], spec: [0.04, 0.04, 0.04], pow: 20 },
};

class Renderer {
  constructor(Q){
    const cvs = this.canvas = document.createElement("canvas");
    cvs.width = 720; cvs.height = 1280;
    const gl = this.gl = cvs.getContext("webgl2", { antialias: true, alpha: true, premultipliedAlpha: true, depth: true, preserveDrawingBuffer: false, powerPreference: "high-performance" });
    if (!gl) throw new Error("no webgl2");
    this.surf = compile(gl, VS_SURF, FS_SURF);
    this.str = compile(gl, VS_STRAND, FS_STRAND);
    gl.vertexAttribI4ui(5, 0, 0, 0, 0);
    // точки лица — в текстуру, их читает вершинный шейдер
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 478, 1);
    for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
    for (const k of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, k, gl.CLAMP_TO_EDGE);
    this.lmBuf = new Float32Array(478 * 4);

    const pomp = buildPompadour(Q), face = buildFace(Q);
    const F32 = a => new Float32Array(a);
    const sh = pomp.shell;
    this.shell = vao(gl, [[0, F32(sh.p), 3], [1, F32(sh.n), 3], [2, F32(sh.t), 3], [4, F32(sh.a), 1]], new Uint32Array(sh.idx));
    // окклюдеры: живая сетка лица (чуть внутрь) и череп
    const idxAll = new Uint16Array(468);
    for (let i = 0; i < 468; i++) idxAll[i] = i;
    const ones = new Float32Array(468).fill(1);
    const faceAttrs = w => [[1, face.N, 3], [2, face.N, 3], [4, w || ones, 1], [5, idxAll, 1, true]];
    this.faceOcc = vao(gl, faceAttrs(null), new Uint16Array(CANON_F));
    this.mBase = vao(gl, faceAttrs(face.mBase.w), new Uint16Array(face.mBase.idx));
    this.sBase = vao(gl, faceAttrs(face.sBase.w), new Uint16Array(face.sBase.idx));
    this.skull = vao(gl, ...ellipsoid(HC, HA.map(a => a - 0.6)));
    const S = (list, bound) => { const k = packStrands(list, bound); return vao(gl, [[0, k.P, 3], [1, k.N, 3], [2, k.T, 3], [3, k.I, 4], [5, k.TR, 3, true], [6, k.BA, 2], [7, k.RO, 3]], k.idx); };
    this.hair = S(pomp.strands, false);
    this.stache = S(face.mustache, true);
    this.burns = S(face.sideburns, true);
    this.babies = S(face.babies, true);
    this.stats = { hair: pomp.strands.length, stache: face.mustache.length, burns: face.sideburns.length };
  }
  upload(L){
    const b = this.lmBuf;
    for (let i = 0; i < 478; i++) { b[i * 4] = L[i * 3]; b[i * 4 + 1] = L[i * 3 + 1]; b[i * 4 + 2] = L[i * 3 + 2]; }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 478, 1, gl.RGBA, gl.FLOAT, b);
  }
  render(pose, view, exposure){
    const gl = this.gl, c = this.canvas;
    if (c.width !== view.W || c.height !== view.H) { c.width = view.W; c.height = view.H; }
    gl.viewport(0, 0, c.width, c.height);
    gl.clearColor(0, 0, 0, 0); gl.clearDepth(1);
    gl.depthMask(true); gl.colorMask(true, true, true, true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.CULL_FACE);
    const R = pose.R, Rc = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
    const L = nrm([0.22, -0.62, -0.75]);
    const common = (pr) => {
      const u = pr.u;
      gl.useProgram(pr.p);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tex); gl.uniform1i(u.u_lm, 0);
      gl.uniformMatrix3fv(u.u_R, false, Rc); gl.uniform1f(u.u_s, pose.s); gl.uniform3fv(u.u_t, pose.t);
      gl.uniform4f(u.u_view, view.dx, view.dy, view.k, 1 / (pose.s * 40));
      gl.uniform2f(u.u_canvas, c.width, c.height); gl.uniform1f(u.u_zc, pose.t[2]);
      gl.uniform3fv(u.u_L, L); gl.uniform1f(u.u_exp, exposure);
    };
    const draw = m => { gl.bindVertexArray(m.o); gl.drawElements(gl.TRIANGLES, m.count, m.type, 0); };
    const S = this.surf.u;
    common(this.surf);
    // 1) окклюдеры — только глубина (в отладке — серая голова)
    if (!this.debug) gl.colorMask(false, false, false, false);
    gl.uniform1f(S.u_alpha, 1);
    gl.uniform3fv(S.u_col, [0.62, 0.6, 0.58]); gl.uniform3fv(S.u_spec, [0, 0, 0]); gl.uniform1f(S.u_pow, 1);
    gl.uniform1i(S.u_bound, 1); gl.uniform1f(S.u_off, this.debug ? 0 : -0.35); draw(this.faceOcc);
    gl.uniform1i(S.u_bound, 0); draw(this.skull);
    gl.colorMask(true, true, true, true);
    // 2) тёмная масса причёски и подложки на коже
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const surf = (m, look, bound, off, depth) => {
      gl.uniform3fv(S.u_col, look.col); gl.uniform3fv(S.u_spec, look.spec); gl.uniform1f(S.u_pow, look.pow);
      gl.uniform1i(S.u_bound, bound); gl.uniform1f(S.u_off, off || 0); gl.depthMask(depth); draw(m);
    };
    surf(this.shell, LOOK.shell, 0, 0, true);
    surf(this.sBase, LOOK.burnsBase, 1, 0.04, false);
    surf(this.mBase, LOOK.stacheBase, 1, 0.05, false);
    gl.depthMask(true); gl.disable(gl.BLEND);
    // 3) пряди
    common(this.str);
    const T = this.str.u;
    const strands = (m, look, bound) => {
      gl.uniform3fv(T.u_base, look.base); gl.uniform3fv(T.u_tip, look.tip); gl.uniform3fv(T.u_s1, look.s1); gl.uniform3fv(T.u_s2, look.s2);
      gl.uniform4fv(T.u_sh, look.sh); gl.uniform1f(T.u_wcm, look.wcm); gl.uniform1f(T.u_wmin, look.wmin); gl.uniform1f(T.u_taper, look.taper);
      gl.uniform1i(T.u_bound, bound); draw(m);
    };
    strands(this.hair, LOOK.hair, 0);
    strands(this.burns, LOOK.burns, 1);
    strands(this.babies, LOOK.babies, 1);
    strands(this.stache, LOOK.stache, 1);
    gl.bindVertexArray(null);
  }
}
function ellipsoid(c, a){
  const p = [], n = [], t = [], al = [], idx = [];
  const NU = 40, NV = 24;
  for (let j = 0; j <= NV; j++) for (let i = 0; i < NU; i++) {
    const th = Math.PI * j / NV, ph = 2 * Math.PI * i / NU;
    const d = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
    p.push(c[0] + a[0] * d[0], c[1] + a[1] * d[1], c[2] + a[2] * d[2]); n.push(...d); t.push(0, 1, 0); al.push(1);
  }
  for (let j = 0; j < NV; j++) for (let i = 0; i < NU; i++) {
    const A = j * NU + i, B = j * NU + (i + 1) % NU, C = A + NU, D = B + NU;
    idx.push(A, C, B, B, C, D);
  }
  return [[[0, new Float32Array(p), 3], [1, new Float32Array(n), 3], [2, new Float32Array(t), 3], [4, new Float32Array(al), 1]], new Uint16Array(idx)];
}

/* ---------------- трекер: грузим движок и модель сами, чтобы показывать проценты */
async function fetchBytes(url, onBytes){
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
  if (!res.body || !res.body.getReader) { const b = new Uint8Array(await res.arrayBuffer()); onBytes(b.length); return b; }
  const rd = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); got += value.length; onBytes(value.length); }
  const out = new Uint8Array(got);
  let o = 0;
  for (const q of parts) { out.set(q, o); o += q.length; }
  return out;
}
async function makeTracker(mode, delegate, onProgress){
  const { FaceLandmarker, FilesetResolver } = await import(BASE + "mp/vision_bundle.js");
  let simd = true;
  try { simd = await FilesetResolver.isSimdSupported(); } catch (e) {}
  const name = simd ? "vision_wasm_internal" : "vision_wasm_nosimd_internal";
  const total = (simd ? 11153617 : 10481398) + 3758596;
  let got = 0;
  const tick = n => { got += n; if (onProgress) onProgress(Math.min(0.99, got / total)); };
  const [wasm, model] = await Promise.all([fetchBytes(BASE + `mp/wasm/${name}.wasm`, tick), fetchBytes(BASE + "mp/face_landmarker.task", tick)]);
  const wasmUrl = URL.createObjectURL(new Blob([wasm], { type: "application/wasm" }));
  const files = { wasmLoaderPath: BASE + `mp/wasm/${name}.js`, wasmBinaryPath: wasmUrl };
  const make = d => FaceLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetBuffer: model, delegate: d },
    runningMode: mode, numFaces: 1,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false,
  });
  // на iPhone GPU-режим MediaPipe ведёт себя нестабильно — там считаем на процессоре
  const ios = /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const want = delegate || (ios ? "CPU" : "GPU");
  let fl, used = want;
  try { fl = await make(want); }
  catch (e) { if (want === "CPU") throw e; used = "CPU"; fl = await make("CPU"); }
  return { fl, delegate: used, make };   // blob с движком не отзываем: он нужен, если придётся пересоздать трекер
}

/* ---------------- снаружи: createMask() → { track(), draw() } */
export async function createMask({ mode = "VIDEO", delegate = null, quality = 1, adaptive = true, debug = false, onProgress = null } = {}){
  const tb = performance.now();
  const r = new Renderer(quality);
  r.stats.buildMs = Math.round(performance.now() - tb);
  r.debug = debug;
  const tr = await makeTracker(mode, delegate, onProgress);
  if (mode === "VIDEO") {
    // прогрев: первый кадр в GPU-режиме долго собирает шейдеры — пусть это случится сейчас
    try { const w = document.createElement("canvas"); w.width = w.height = 64; w.getContext("2d").fillRect(0, 0, 64, 64); tr.fl.detectForVideo(w, 1); }
    catch (e) {
      if (tr.delegate === "GPU") { try { tr.fl.close(); } catch (e2) {} tr.fl = await tr.make("CPU"); tr.delegate = "CPU"; }
    }
  }
  const raw = new Float32Array(478 * 3);
  const sm = new Smoother(478 * 3);
  const lum = document.createElement("canvas"); lum.width = lum.height = 8;
  const lg = lum.getContext("2d", { willReadFrequently: true });
  const M = {
    raw, get delegate(){ return tr.delegate; }, stats: r.stats, found: false, vis: 0, exposure: 1, detMs: 0,
    pose: null, lm: null, srcW: 0, srcH: 0, miss: 0, lastTs: 0, frames: 0, skip: false,
    _set(lms, W, H, now, smoothIt){
      this.srcW = W; this.srcH = H;
      for (let i = 0; i < 478; i++) { const p = lms[i]; raw[i * 3] = p.x * W; raw[i * 3 + 1] = p.y * H; raw[i * 3 + 2] = p.z * W; }
      const s0 = this.pose ? this.pose.s : Math.hypot(raw[454 * 3] - raw[234 * 3], raw[454 * 3 + 1] - raw[234 * 3 + 1]) / 15.3;
      this.lm = smoothIt ? sm.apply(raw, now / 1000, Math.max(1, s0)) : raw.slice();
      this.pose = fitPose(this.lm);
      r.upload(this.lm);
      this.found = true; this.miss = 0;
    },
    trackImage(img){
      const res = tr.fl.detect(img);
      const lms = res.faceLandmarks && res.faceLandmarks[0];
      if (!lms) { this.found = false; return false; }
      this._set(lms, img.naturalWidth || img.width, img.naturalHeight || img.height, 0, false);
      this.vis = 1;
      return true;
    },
    _watch(video, now){
      // ищем лицо только на новом кадре камеры (requestVideoFrameCallback), а если колбэки молчат — на каждом
      if (!adaptive || !video.requestVideoFrameCallback) return true;
      if (this._rv !== video) {
        this._rv = video; this._fresh = true; this._seen = now;
        const cb = () => { this._fresh = true; this._seen = performance.now(); if (this._rv === video) video.requestVideoFrameCallback(cb); };
        video.requestVideoFrameCallback(cb);
      }
      if (!this._fresh && now - this._seen < 250) return false;
      this._fresh = false;
      return true;
    },
    track(video, now = performance.now()){
      if (!video || video.readyState < 2 || !video.videoWidth) return this.found;
      if (!this._watch(video, now)) return this.found;
      this.frames++;
      if (this.skip && this.frames % 2) return this.found;   // слабый телефон: ищем лицо через кадр
      const ts = Math.max(this.lastTs + 1, Math.round(now)); this.lastTs = ts;
      const t0 = performance.now();
      let res;
      try { res = tr.fl.detectForVideo(video, ts); }
      catch (e) {
        console.warn("mask: detect failed", e);
        this.found = false;
        if (tr.delegate === "GPU" && !this.switching) {   // GPU подвёл — тихо переходим на процессор
          this.switching = true;
          tr.make("CPU").then(fl => { try { tr.fl.close(); } catch (e2) {} tr.fl = fl; tr.delegate = "CPU"; this.lastTs = 0; }).catch(() => {}).finally(() => { this.switching = false; });
        }
        return false;
      }
      const dt = performance.now() - t0;
      this.detMs = this.detMs ? this.detMs * 0.9 + dt * 0.1 : dt;
      this.skip = adaptive && this.detMs > 26;
      const lms = res && res.faceLandmarks && res.faceLandmarks[0];
      if (!lms) { if (++this.miss > 4) { this.found = false; sm.reset(); } return this.found; }
      this._set(lms, video.videoWidth, video.videoHeight, now, true);
      if (this.frames % 12 === 0) this._light(video);
      return true;
    },
    _light(src){
      // яркость лица → насколько ярко блестят волосы
      try {
        const L = this.lm, xs = [234, 454, 10, 152].map(i => L[i * 3]), ys = [234, 454, 10, 152].map(i => L[i * 3 + 1]);
        const x0 = Math.max(0, Math.min(...xs)), x1 = Math.min(this.srcW, Math.max(...xs)), y0 = Math.max(0, Math.min(...ys)), y1 = Math.min(this.srcH, Math.max(...ys));
        if (x1 - x0 < 8 || y1 - y0 < 8) return;
        lg.drawImage(src, x0, y0, x1 - x0, y1 - y0, 0, 0, 8, 8);
        const d = lg.getImageData(0, 0, 8, 8).data;
        let s = 0;
        for (let i = 0; i < d.length; i += 4) s += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
        const Y = s / 64 / 255;
        this.exposure = this.exposure * 0.6 + clamp(Math.pow(Y / 0.5, 0.8), 0.5, 1.3) * 0.4;
      } catch (e) {}
    },
    // view: как камера легла на холст (coverDraw): dx, dy, k — пикселей холста на пиксель кадра
    draw(g, view, alpha = 1){
      const target = this.found ? 1 : 0;
      this.vis += (target - this.vis) * 0.35;
      if (this.vis < 0.02 || !this.pose) return;
      r.render(this.pose, view, this.exposure);
      g.save();
      g.globalAlpha = Math.min(1, this.vis) * alpha;
      if (view.mirror) { g.translate(view.W, 0); g.scale(-1, 1); }
      g.drawImage(r.canvas, 0, 0, view.W, view.H);
      g.restore();
    },
    debugTurn(yaw, pitch){ // повернуть найденное лицо (для проверки причёски с разных сторон)
      const P = this.pose, c = add(P.t, mul([P.R[0] * HC[0] + P.R[1] * HC[1] + P.R[2] * HC[2], P.R[3] * HC[0] + P.R[4] * HC[1] + P.R[5] * HC[2], P.R[6] * HC[0] + P.R[7] * HC[1] + P.R[8] * HC[2]], P.s));
      const cy = Math.cos(yaw * DEG), sy = Math.sin(yaw * DEG), cp = Math.cos(pitch * DEG), sp = Math.sin(pitch * DEG);
      const L = this.base0 || (this.base0 = this.lm.slice()), out = new Float32Array(L.length);
      for (let i = 0; i < 478; i++) {
        let x = L[i * 3] - c[0], y = L[i * 3 + 1] - c[1], z = L[i * 3 + 2] - c[2];
        [x, z] = [x * cy - z * sy, x * sy + z * cy];
        [y, z] = [y * cp - z * sp, y * sp + z * cp];
        out[i * 3] = x + c[0]; out[i * 3 + 1] = y + c[1]; out[i * 3 + 2] = z + c[2];
      }
      this.lm = out; this.pose = fitPose(out); r.upload(out);
    },
    faceBox(view){
      if (!this.lm) return null;
      const L = this.lm; let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
      for (let i = 0; i < 468; i++) { x0 = Math.min(x0, L[i * 3]); x1 = Math.max(x1, L[i * 3]); y0 = Math.min(y0, L[i * 3 + 1]); y1 = Math.max(y1, L[i * 3 + 1]); }
      const k = view ? view.k : 1, dx = view ? view.dx : 0, dy = view ? view.dy : 0;
      return [dx + x0 * k, dy + y0 * k, dx + x1 * k, dy + y1 * k];
    },
  };
  return M;
}
