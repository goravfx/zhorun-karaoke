// Маски участников «Жоруна»: всё снято с их фото.
// Причёски — фото анфас на объёме головы (2.5D) + оболочка по бокам и сзади; ирокез — объёмные шипы;
// усы, бакенбарды, зубы — наклейки на сетку лица; мешки, морщины, прыщи и грим — прямо по коже человека
// (кадр камеры под лицом умножается на детали образца, поэтому свет и цвет лица остаются его).
// Лицо ищет MediaPipe Face Landmarker в отдельном потоке, а на экран идёт тот же кадр камеры,
// на котором лицо нашли, — маска не отстаёт от головы.
import { CANON_V, CANON_F, CANON_UV } from "./face-canon.js";

const BASE = new URL("./", import.meta.url).href;
const VQ = "?v=9";   // версия ресурсов маски: после обновления браузер не смешает старые файлы с новыми
const DEG = Math.PI / 180;

/* ---------------- математика */
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = a => Math.hypot(a[0], a[1], a[2]);
const nrm = a => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const cv = i => [CANON_V[i * 3], CANON_V[i * 3 + 1], CANON_V[i * 3 + 2]];

/* голова: эллипсоид черепа и сферические координаты «зачёса» — те же, что при сборке текстур */
const HC = [0, 3.0, -4.6], HA = [7.9, 10.6, 9.9];
const TF = 15 * DEG;
const FP = [0, Math.sin(TF), Math.cos(TF)], E1 = [0, Math.cos(TF), -Math.sin(TF)];
function dirLM(l, m){
  const sm = Math.sin(m * DEG), cm = Math.cos(m * DEG), cl = Math.cos(l * DEG), sl = Math.sin(l * DEG);
  return [sm * sl, FP[1] * cm + sm * E1[1] * cl, FP[2] * cm + sm * E1[2] * cl];
}

/* ---------------- загрузка */
async function fetchBytes(url, onBytes){
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
  if (!res.body || !res.body.getReader) { const b = new Uint8Array(await res.arrayBuffer()); if (onBytes) onBytes(b.length); return b; }
  const rd = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); got += value.length; if (onBytes) onBytes(value.length); }
  const out = new Uint8Array(got);
  let o = 0;
  for (const q of parts) { out.set(q, o); o += q.length; }
  return out;
}
async function loadImage(url){
  const img = new Image();
  img.decoding = "async";
  img.src = url;
  await img.decode();
  return img;
}

/* ---------------- образы: из каких файлов состоит каждый (файлы — в папке образа) */
const LOOKS = {
  zhorun:    { ears: true, shell: { cut: 0.55, crown: 1.1 }, files: ["proj.webp", "proj.bin", "hair.webp", "shell.bin", "decal.webp", "decal_tris.bin", "burns.webp", "burns.bin"] },
  drummer:   { ears: true, files: ["spikes.webp", "spikes.bin", "det.webp"] },
  accordion: { ears: true, files: ["paint.webp"] },
  guitar:    { ears: false, fill: true, spec: 0.11, shell: { cut: 0.35, crown: 0.6 }, files: ["proj.webp", "proj.bin", "hair.webp", "shell.bin", "det.webp", "teeth.webp", "teeth.bin"] },
  trumpet:   { ears: false, fill: true, spec: 0.1, shell: { cut: 0.35, crown: 0.6 }, files: ["proj.webp", "proj.bin", "hair.webp", "shell.bin", "det.webp"] },
};
export const LOOK_NAMES = Object.keys(LOOKS);
// .bin лежат и сжатыми (.gz, втрое меньше): браузер распаковывает сам; старый браузер без распаковщика берёт обычные
async function fetchGz(url){
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
  return new Uint8Array(await new Response(res.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
}
async function loadLook(name, lenient = false){
  const A = {}, dir = BASE + name + "/";
  await Promise.all(LOOKS[name].files.map(async f => {
    try {
      if (/\.(webp|png|jpg)$/.test(f)) A[f] = await loadImage(dir + f + VQ);
      else if (typeof DecompressionStream === "function") { try { A[f] = await fetchGz(dir + f + ".gz" + VQ); } catch (e) { A[f] = await fetchBytes(dir + f + VQ); } }
      else A[f] = await fetchBytes(dir + f + VQ);
    } catch (e) { if (!lenient) throw e; }
  }));
  return A;
}

/* ---------------- поза головы по устойчивым точкам (Horn) */
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
const quatToR = ([w, x, y, z]) => [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
const Rmul = (R, a) => [R[0] * a[0] + R[1] * a[1] + R[2] * a[2], R[3] * a[0] + R[4] * a[1] + R[5] * a[2], R[6] * a[0] + R[7] * a[1] + R[8] * a[2]];
const RTmul = (R, a) => [R[0] * a[0] + R[3] * a[1] + R[6] * a[2], R[1] * a[0] + R[4] * a[1] + R[7] * a[2], R[2] * a[0] + R[5] * a[1] + R[8] * a[2]];
function fitPose(L){ // L: 478*3 в пикселях кадра → {s, q, R, t}
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
  const q = maxEig4([Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx, Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
    Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy, Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]);
  const R = quatToR(q);
  let num = 0;
  for (let k = 0; k < n; k++) num += dot(sub(B[k], cb), Rmul(R, STABLE_C.a[k]));
  const s = num / aa;
  return { s, q, R, t: sub(cb, mul(Rmul(R, STABLE_C.c), s)) };
}

/* ---------------- сглаживание: поза головы отдельно, мимика отдельно (One Euro) */
const alphaOf = (dt, fc) => { const tau = 1 / (2 * Math.PI * fc); return 1 / (1 + tau / dt); };
export class Smoother {
  constructor(){ this.reset(); }
  reset(){ this.pose = null; this.def = null; this.t = 0; this.motion = 0; this.out = new Float32Array(478 * 3); }
  apply(raw, tSec, smooth = true){
    const P = fitPose(raw);
    // мимика: отклонение точек от канонической формы, в см (в осях головы)
    if (!this.def) this.def = new Float32Array(468 * 3);
    const def = new Float32Array(468 * 3);
    for (let i = 0; i < 468; i++) {
      const d = RTmul(P.R, [raw[i * 3] - P.t[0], raw[i * 3 + 1] - P.t[1], raw[i * 3 + 2] - P.t[2]]);
      def[i * 3] = d[0] / P.s - CANON_V[i * 3]; def[i * 3 + 1] = d[1] / P.s - CANON_V[i * 3 + 1]; def[i * 3 + 2] = d[2] / P.s - CANON_V[i * 3 + 2];
    }
    const dt = clamp(tSec - this.t, 1 / 120, 0.25);
    const fresh = !smooth || !this.pose || tSec - this.t > 0.5;
    this.t = tSec;
    if (fresh) { this.pose = { s: P.s, q: P.q.slice(), t: P.t.slice() }; this.def.set(def); this.motion = 0; }
    else {
      const F = this.pose;
      let q = P.q;
      if (q[0] * F.q[0] + q[1] * F.q[1] + q[2] * F.q[2] + q[3] * F.q[3] < 0) q = q.map(x => -x);
      // скорость движения головы (см/с): сдвиг + поворот (радиан × 10 см)
      const dtr = len(sub(P.t, F.t)) / P.s;
      const dq = Math.hypot(q[0] - F.q[0], q[1] - F.q[1], q[2] - F.q[2], q[3] - F.q[3]) * 2;
      const sp = (dtr + dq * 10) / dt;
      this.motion += (sp - this.motion) * alphaOf(dt, 4.0);
      const a = alphaOf(dt, Smoother.POSE_MIN + Smoother.POSE_BETA * this.motion);
      F.s += (P.s - F.s) * a;
      for (let k = 0; k < 3; k++) F.t[k] += (P.t[k] - F.t[k]) * a;
      for (let k = 0; k < 4; k++) F.q[k] += (q[k] - F.q[k]) * a;
      const ql = Math.hypot(...F.q); for (let k = 0; k < 4; k++) F.q[k] /= ql;
      for (let i = 0; i < 468 * 3; i += 3) {
        const ex = def[i] - this.def[i], ey = def[i + 1] - this.def[i + 1], ez = def[i + 2] - this.def[i + 2];
        const ad = alphaOf(dt, Smoother.DEF_MIN + Smoother.DEF_BETA * Math.hypot(ex, ey, ez) / dt);
        this.def[i] += ex * ad; this.def[i + 1] += ey * ad; this.def[i + 2] += ez * ad;
      }
    }
    const F = this.pose, R = quatToR(F.q);
    for (let i = 0; i < 468; i++) {
      const c = [CANON_V[i * 3] + this.def[i * 3], CANON_V[i * 3 + 1] + this.def[i * 3 + 1], CANON_V[i * 3 + 2] + this.def[i * 3 + 2]];
      const p = Rmul(R, c);
      this.out[i * 3] = p[0] * F.s + F.t[0]; this.out[i * 3 + 1] = p[1] * F.s + F.t[1]; this.out[i * 3 + 2] = p[2] * F.s + F.t[2];
    }
    for (let i = 468 * 3; i < 478 * 3; i++) this.out[i] = raw[i];
    return { lm: this.out, pose: { s: F.s, R, t: F.t.slice(), q: F.q.slice() } };
  }
}
Smoother.POSE_MIN = 1.5; Smoother.POSE_BETA = 1.4; Smoother.DEF_MIN = 3.0; Smoother.DEF_BETA = 1.6;   // в движении фильтр почти не тормозит: картинка и так «из прошлого»

/* ---------------- WebGL */
const VHEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D u_lm; uniform mat3 u_R; uniform float u_s; uniform vec3 u_t;
uniform vec4 u_view; uniform vec2 u_canvas; uniform float u_zc;
vec3 lm(uint i){ return texelFetch(u_lm, ivec2(int(i), 0), 0).xyz; }
vec4 clipOf(vec3 P){ vec2 c = u_view.xy + P.xy * u_view.z; return vec4(c.x / u_canvas.x * 2.0 - 1.0, 1.0 - c.y / u_canvas.y * 2.0, (P.z - u_zc) * u_view.w, 1.0); }
`;
const VS_RIGID = VHEAD + `
layout(location=0) in vec3 a_p; layout(location=1) in vec3 a_n; layout(location=2) in vec2 a_uv; layout(location=3) in float a_a; layout(location=4) in vec3 a_t;
uniform vec3 u_K; uniform vec3 u_B;
out vec2 v_uv; out float v_a; out vec3 v_n; out vec3 v_t;
void main(){ vec3 P = u_s * (u_R * (a_p * u_K + u_B)) + u_t; v_n = u_R * a_n; v_t = u_R * a_t; v_uv = a_uv; v_a = a_a; gl_Position = clipOf(P); }`;
const VS_BOUND = VHEAD + `
layout(location=1) in vec3 a_n; layout(location=2) in vec2 a_uv; layout(location=3) in float a_off; layout(location=5) in uint a_i;
uniform float u_off;
out vec2 v_uv; out float v_a; out vec3 v_n;
void main(){ vec3 N = normalize(u_R * a_n); vec3 P = lm(a_i) + N * ((a_off + u_off) * u_s); v_n = N; v_uv = a_uv; v_a = 1.0; gl_Position = clipOf(P); }`;
// кожа: сетка лица ровно по точкам (+ полоса лба выше сетки — продолжение по двум точкам), a_cov — 0 по краю (без шва)
// v_src — откуда брать кожу для полосы лба: её зеркало вниз через край сетки (прячет свою линию волос под чужой причёской)
const VS_SKIN = VHEAD + `
layout(location=1) in vec3 a_n; layout(location=2) in vec2 a_uv; layout(location=3) in float a_cov; layout(location=4) in float a_band; layout(location=5) in uvec2 a_ii; layout(location=6) in float a_w; layout(location=7) in float a_w2;
out vec2 v_uv; out float v_a; out vec3 v_n; out vec2 v_src; out float v_band;
void main(){ vec3 N = normalize(u_R * a_n); v_n = N; v_uv = a_uv; v_a = a_cov; v_band = a_band; v_src = mix(lm(a_ii.x), lm(a_ii.y), a_w2).xy; gl_Position = clipOf(mix(lm(a_ii.x), lm(a_ii.y), a_w)); }`;
// полоса у точек лица (бакенбарды, зубы): смесь двух точек + смещение в осях головы
const VS_STRIP = VHEAD + `
layout(location=1) in vec3 a_n; layout(location=2) in vec2 a_uv; layout(location=4) in vec3 a_d; layout(location=5) in uvec2 a_ii; layout(location=6) in float a_w;
out vec2 v_uv; out float v_a; out vec3 v_n;
void main(){ vec3 N = normalize(u_R * a_n); vec3 P = mix(lm(a_ii.x), lm(a_ii.y), a_w) + (u_R * a_d) * u_s; v_n = N; v_uv = a_uv; v_a = 1.0; gl_Position = clipOf(P); }`;
// причёска «2.5D»: фото анфас на объёме причёски; u_K/u_B подгоняют её под лоб и виски человека
const VS_PROJ = VHEAD + `
layout(location=0) in vec3 a_p; layout(location=2) in vec2 a_uv; layout(location=3) in float a_tw; layout(location=4) in vec3 a_td;
layout(location=5) in uvec3 a_ti; layout(location=6) in vec3 a_tb;
uniform vec3 u_K; uniform vec3 u_B;
out vec2 v_uv;
void main(){
  vec3 Pr = u_s * (u_R * (a_p * u_K + u_B)) + u_t;                                   // объём — вместе с головой
  vec3 Pf = a_tb.x * lm(a_ti.x) + a_tb.y * lm(a_ti.y) + a_tb.z * lm(a_ti.z) + u_s * (u_R * a_td);   // у лба — на коже лба
  v_uv = a_uv; gl_Position = clipOf(mix(Pr, Pf, a_tw));
}`;
// шипы ирокеза: лента вдоль оси шипа, всегда развёрнута к камере (как пучок волос с любого ракурса)
const VS_SPIKE = VHEAD + `
layout(location=0) in vec3 a_c; layout(location=1) in vec3 a_tan; layout(location=2) in vec2 a_uv; layout(location=3) in float a_w; layout(location=4) in vec2 a_sh;
uniform vec3 u_K; uniform vec3 u_B;
out vec2 v_uv; out vec2 v_sh;
void main(){
  vec3 C = u_s * (u_R * (a_c * u_K + u_B)) + u_t;
  vec3 T = normalize(u_R * (a_tan * u_K));
  vec3 X = u_R * vec3(1.0, 0.0, 0.0);
  vec3 S = cross(T, vec3(0.0, 0.0, 1.0));
  float sl = length(S);
  S = sl > 1e-4 ? S / sl : X;
  if (dot(S, X) < 0.0) S = -S;
  S = normalize(mix(X, S, smoothstep(0.08, 0.35, sl)));   // шип смотрит прямо в камеру — лента не схлопывается
  v_uv = a_uv; v_sh = a_sh;
  gl_Position = clipOf(C + S * (a_w * u_s));
}`;
const FS_PROJ = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform float u_exp;
in vec2 v_uv; out vec4 o;
void main(){ vec4 c = texture(u_tex, v_uv); if (c.a < 0.004) discard; o = vec4(c.rgb * u_exp, c.a); }`;
const FS_SPIKE = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform float u_exp;
in vec2 v_uv; in vec2 v_sh; out vec4 o;
void main(){ vec4 c = texture(u_tex, v_uv) * v_sh.y; if (c.a < 0.004) discard; o = vec4(c.rgb * (v_sh.x * u_exp), c.a); }`;
const FS_DEPTH = `#version 300 es
precision mediump float; out vec4 o; void main(){ o = vec4(0.0); }`;
const FS_GRAY = `#version 300 es
precision mediump float; in vec3 v_n; out vec4 o;
void main(){ vec3 N = normalize(v_n); float l = 0.35 + 0.65 * max(0.0, dot(N, normalize(vec3(0.3, -0.5, -0.8)))); o = vec4(vec3(0.62, 0.6, 0.58) * l, 1.0); }`;
const FS_SHELL = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform float u_exp; uniform float u_spec;
in vec2 v_uv; in float v_a; in vec3 v_n; in vec3 v_t; out vec4 o;
void main(){
  vec4 c = texture(u_tex, v_uv);                  // цвет уже умножен на альфу
  vec3 N = normalize(v_n);
  float rim = smoothstep(-0.04, 0.02, -N.z);      // бока причёски плотные, изнанка не рисуется
  float k = v_a * rim;
  if (c.a * k < 0.02) discard;
  vec3 L = normalize(vec3(0.15, -0.75, -0.65));   // свет сверху-спереди (y вниз, z от камеры)
  float dif = 0.8 + 0.28 * max(0.0, dot(N, L));
  // блеск вдоль прядей (Kajiya–Kay): полоса бликов «едет» по причёске, когда голова поворачивается
  vec3 H = normalize(L + vec3(0.0, 0.0, -1.0));
  float tl = length(v_t), spec = 0.0;
  if (tl > 1e-4) { float th = dot(v_t / tl, H); spec = pow(sqrt(max(0.0, 1.0 - th * th)), 14.0) * smoothstep(0.0, 0.5, dot(N, H)); }
  vec3 col = c.rgb * dif + vec3(0.92, 0.9, 0.87) * (spec * u_spec * c.a);
  o = vec4(col * k * u_exp, c.a * k);
}`;
const FS_DECAL = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform float u_exp;
in vec2 v_uv; in vec3 v_n; out vec4 o;
void main(){
  vec4 c = texture(u_tex, v_uv);
  if (c.a < 0.01) discard;
  float rim = smoothstep(-0.2, 0.05, -normalize(v_n).z);
  o = vec4(c.rgb * u_exp, c.a) * rim;
}`;
// кожа: берём сам кадр камеры под лицом и меняем его — мешки, морщины, прыщи умножаются на кожу человека,
// грим ложится краской со светом и тенями его лица. Снаружи слоя кадр не трогаем.
const FS_SKIN = `#version 300 es
precision highp float;
uniform sampler2D u_cam; uniform vec4 u_cm;      // (dx, dy, 1/(k·w), 1/(k·h)): пиксель холста → uv кадра
uniform sampler2D u_det; uniform sampler2D u_paint;
uniform vec2 u_canvas; uniform vec3 u_skin; uniform float u_light; uniform vec2 u_camInv; uniform float u_fill;
in vec2 v_uv; in float v_a; in vec3 v_n; in vec2 v_src; in float v_band; out vec4 o;
void main(){
  vec2 cp = vec2(gl_FragCoord.x, u_canvas.y - gl_FragCoord.y);
  vec3 cam = texture(u_cam, (cp - u_cm.xy) * u_cm.zw).rgb;
  float fb = v_band * u_fill;
  if (fb > 0.001) {                              // полоса лба — кожа лба человека (зеркально снизу); если там прядь — просто цвет кожи
    vec3 fs = texture(u_cam, v_src * u_camInv).rgb;
    float lr = dot(fs, vec3(0.299, 0.587, 0.114)) / max(dot(u_skin, vec3(0.299, 0.587, 0.114)), 0.04);
    fs = mix(u_skin * 0.97, fs, smoothstep(0.5, 0.78, lr));
    cam = mix(cam, fs, fb);
  }
  vec4 d = texture(u_det, v_uv);                 // rgb — множитель/2 (0.5 — без изменений), a — где слой есть
  vec4 p = texture(u_paint, v_uv);               // краска (умножена на альфу)
  float rim = smoothstep(-0.25, 0.1, -normalize(v_n).z) * v_a;
  float A = max(d.a, fb) * rim, P = p.a * rim;
  if (A + P < 0.003) discard;
  vec3 base = cam * (d.rgb * 2.0);
  const vec3 Y = vec3(0.299, 0.587, 0.114);
  float sh = pow(clamp(dot(cam, Y) / max(dot(u_skin, Y), 0.04), 0.12, 1.6), 0.85);
  vec3 paint = p.rgb * (sh * u_light) + vec3(0.14) * max(0.0, sh - 1.12) * p.a;   // тени лица на гриме + жирный блеск
  vec3 col = paint * rim + base * A * (1.0 - P);
  o = vec4(col, P + A * (1.0 - P));
}`;
function program(gl, vs, fs){
  const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) { const info = gl.getActiveUniform(p, i); u[info.name] = gl.getUniformLocation(p, info.name); }
  return { p, u };
}
function vao(gl, attrs, idx, own, dynamic){
  const o = gl.createVertexArray(); gl.bindVertexArray(o);
  const bufs = [];
  for (const [loc, data, size, int] of attrs) {
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); bufs.push(b);
    gl.enableVertexAttribArray(loc);
    if (int) gl.vertexAttribIPointer(loc, size, gl.UNSIGNED_SHORT, 0, 0); else gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }
  const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, dynamic ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW); bufs.push(ib);
  gl.bindVertexArray(null);
  if (own) { own.vaos.push(o); own.bufs.push(...bufs); }
  return { o, ib, count: idx.length, type: idx instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT };
}
function texture(gl, img, clampS, own, premult = true){
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, premult);
  if (!premult) gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.BROWSER_DEFAULT_WEBGL);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, clampS ? gl.CLAMP_TO_EDGE : gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  if (own) own.texs.push(t);
  return t;
}
function solid(gl, rgba){
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(rgba));
  for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
  return t;
}
function faceNormals(){
  const n = new Float32Array(468 * 3);
  for (let f = 0; f < CANON_F.length; f += 3) {
    const ids = [CANON_F[f], CANON_F[f + 1], CANON_F[f + 2]];
    const [A, B, C] = ids.map(cv);
    let fn = cross(sub(B, A), sub(C, A));
    if (dot(fn, sub(mul(add(add(A, B), C), 1 / 3), [0, 0, -3])) < 0) fn = mul(fn, -1);
    for (const i of ids) { n[i * 3] += fn[0]; n[i * 3 + 1] += fn[1]; n[i * 3 + 2] += fn[2]; }
  }
  for (let i = 0; i < 468; i++) n.set(nrm([n[i * 3], n[i * 3 + 1], n[i * 3 + 2]]), i * 3);
  return n;
}
const OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
// сетка для слоя кожи: лицо MediaPipe + полоса лба над ним (два ряда: 1 и 2 см) — грим и морщины не обрываются
// посреди лба, а под чужой причёской своя линия волос закрывается кожей лба (зеркало вниз через край сетки)
const TOP = [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389];
const EXT = [1.5, 2.6];
function bandUV(uc, ui, ww){   // продолжение uv за край сетки, но не за край текстуры
  let f = 1;
  for (const q of [0, 1]) { const d = (ui[q] - uc[q]) * (ww - 1), room = d > 0 ? 0.998 - ui[q] : ui[q] - 0.002; if (Math.abs(d) > 1e-6) f = Math.min(f, Math.max(0, room) / Math.abs(d)); }
  return [ui[0] + (ui[0] - uc[0]) * (ww - 1) * f, ui[1] + (ui[1] - uc[1]) * (ww - 1) * f];
}
function skinMesh(gl, N){
  const NT = TOP.length, nv = 468 + NT * EXT.length;
  const ii = new Uint16Array(nv * 2), w = new Float32Array(nv), w2 = new Float32Array(nv), band = new Float32Array(nv), uv = new Float32Array(nv * 2), n = new Float32Array(nv * 3), cov = new Float32Array(nv).fill(1);
  for (let i = 0; i < 468; i++) { ii[i * 2] = ii[i * 2 + 1] = i; uv[i * 2] = CANON_UV[i * 2]; uv[i * 2 + 1] = 1 - CANON_UV[i * 2 + 1]; }
  n.set(N);
  for (const i of OVAL) cov[i] = 0;
  const onOval = new Set(OVAL), idx = Array.from(CANON_F);
  const row = r => 468 + r * NT;
  TOP.forEach((i, k) => {
    const a = TOP[Math.max(0, k - 1)], b = TOP[Math.min(NT - 1, k + 1)], Vi = cv(i);
    let out = nrm(cross(sub(cv(b), cv(a)), [N[i * 3], N[i * 3 + 1], N[i * 3 + 2]]));
    if (out[0] * Vi[0] + out[1] * (Vi[1] - 1) < 0) out = mul(out, -1);
    const tgt = sub(Vi, mul(out, 1.6));
    let c = -1, best = 1e9;
    for (let j = 0; j < 468; j++) { if (onOval.has(j)) continue; const d = len(sub(cv(j), tgt)); if (d < best) { best = d; c = j; } }
    const dist = Math.max(0.5, len(sub(Vi, cv(c))));
    EXT.forEach((ex, r) => {
      const e = row(r) + k, ww = 1 + ex / dist;
      ii[e * 2] = c; ii[e * 2 + 1] = i; w[e] = ww; w2[e] = 2 - ww; band[e] = 1;
      uv.set(bandUV([uv[c * 2], uv[c * 2 + 1]], [uv[i * 2], uv[i * 2 + 1]], ww), e * 2);
      n.set(N.subarray(i * 3, i * 3 + 3), e * 3);
      cov[e] = r === EXT.length - 1 || k === 0 || k === NT - 1 ? 0 : 1;
    });
    band[i] = 1; cov[i] = k === 0 || k === NT - 1 ? 0 : 1;
    if (k < NT - 1) {
      const e0 = row(0) + k, e1 = row(1) + k;
      idx.push(i, TOP[k + 1], e0, TOP[k + 1], e0 + 1, e0, e0, e0 + 1, e1, e0 + 1, e1 + 1, e1);
    }
  });
  return vao(gl, [[1, n, 3], [2, uv, 2], [3, cov, 1], [4, band, 1], [5, ii, 2, true], [6, w, 1], [7, w2, 1]], new Uint16Array(idx));
}
function ellipsoid(c, a){
  const p = [], n = [], idx = [], NU = 36, NV = 20;
  for (let j = 0; j <= NV; j++) for (let i = 0; i < NU; i++) {
    const th = Math.PI * j / NV, ph = 2 * Math.PI * i / NU, d = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
    p.push(c[0] + a[0] * d[0], c[1] + a[1] * d[1], c[2] + a[2] * d[2]); n.push(...d);
  }
  for (let j = 0; j < NV; j++) for (let i = 0; i < NU; i++) { const A = j * NU + i, B = j * NU + (i + 1) % NU; idx.push(A, A + NU, B, B, A + NU, B + NU); }
  return { p: new Float32Array(p), n: new Float32Array(n), idx: new Uint16Array(idx) };
}
function earOcc(side){ // ухо: эллипсоид за краем лица, верх отклонён назад — за ушами прячутся волосы с затылка
  const e = ellipsoid([0, 0, 0], [1.3, 3.1, 2.0]), a = 15 * DEG, c = Math.cos(a), s = Math.sin(a);
  for (let i = 0; i < e.p.length; i += 3) {
    const y = e.p[i + 1], z = e.p[i + 2], ny = e.n[i + 1], nz = e.n[i + 2];
    e.p[i] += side * 8.0; e.p[i + 1] = y * c + z * s + 1.2; e.p[i + 2] = -y * s + z * c - 6.1;   // спереди не заходит на бакенбарды
    e.n[i + 1] = ny * c + nz * s; e.n[i + 2] = -ny * s + nz * c;
  }
  return e;
}
function neck(){ // шея: закрывает то, что у затылка уходит ниже челюсти
  const p = [], n = [], idx = [], NU = 32, ys = [-3.6, -9, -16, -26];
  for (const y of ys) for (let i = 0; i < NU; i++) { const a = 2 * Math.PI * i / NU; p.push(5.4 * Math.cos(a), y, -4.8 + 5.8 * Math.sin(a)); n.push(Math.cos(a), 0, Math.sin(a)); }
  for (let j = 0; j < ys.length - 1; j++) for (let i = 0; i < NU; i++) { const A = j * NU + i, B = j * NU + (i + 1) % NU; idx.push(A, A + NU, B, B, A + NU, B + NU); }
  return { p: new Float32Array(p), n: new Float32Array(n), idx: new Uint16Array(idx) };
}
const u8 = (b, o, n) => new Uint8Array(b.buffer, b.byteOffset + o, n);
const f32 = (b, o, n) => new Float32Array(b.slice(o, o + n * 4).buffer);
const u16 = (b, o, n) => new Uint16Array(b.slice(o, o + n * 2).buffer);
const u32 = (b, o, n) => new Uint32Array(b.slice(o, o + n * 4).buffer);

class Renderer {
  constructor(debug){
    const cvs = this.canvas = document.createElement("canvas");
    cvs.width = 720; cvs.height = 1280;
    const gl = this.gl = cvs.getContext("webgl2", { antialias: false, alpha: true, premultipliedAlpha: true, depth: true, preserveDrawingBuffer: false, powerPreference: "high-performance" });
    if (!gl) throw new Error("no webgl2");
    this.debug = debug;
    this.gray = !!debug && !debug.nogray;        // отладка: голова-окклюдер видна серой
    this.pRigidDepth = program(gl, VS_RIGID, this.gray ? FS_GRAY : FS_DEPTH);
    this.pBoundDepth = program(gl, VS_BOUND, this.gray ? FS_GRAY : FS_DEPTH);
    this.pShell = program(gl, VS_RIGID, FS_SHELL);
    this.pDecal = program(gl, VS_BOUND, FS_DECAL);
    this.pStrip = program(gl, VS_STRIP, FS_DECAL);
    this.pProj = program(gl, VS_PROJ, FS_PROJ);
    this.pSkin = program(gl, VS_SKIN, FS_SKIN);
    this.pSpike = program(gl, VS_SPIKE, FS_SPIKE);
    this.lmTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.lmTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 478, 1);
    for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
    this.lmBuf = new Float32Array(478 * 4);
    // кадр камеры для слоя кожи
    this.camTex = gl.createTexture(); this.camW = this.camH = 0;
    gl.bindTexture(gl.TEXTURE_2D, this.camTex);
    for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.LINEAR);
    for (const k of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, k, gl.CLAMP_TO_EDGE);
    this.neutralTex = solid(gl, [128, 128, 128, 0]); this.clearTex = solid(gl, [0, 0, 0, 0]);
    // сетка лица: окклюдер, слой кожи, наклейки
    const N = this.faceN = faceNormals(), uv = this.faceUV = new Float32Array(468 * 2), ids = this.faceIds = new Uint16Array(468);
    for (let i = 0; i < 468; i++) { ids[i] = i; uv[i * 2] = CANON_UV[i * 2]; uv[i * 2 + 1] = 1 - CANON_UV[i * 2 + 1]; }
    const occOff = new Float32Array(468);
    for (let i = 0; i < 468; i++) { const ax = Math.abs(CANON_V[i * 3]); occOff[i] = -0.7 * clamp((ax - 5.5) / 2.1, 0, 1); }
    this.faceOcc = vao(gl, [[1, N, 3], [2, uv, 2], [3, occOff, 1], [5, ids, 1, true]], new Uint16Array(CANON_F));
    this.skinMesh = skinMesh(gl, N);
    const sk = ellipsoid(HC, HA.map(a => a - 0.6)), nk = neck();   // больше нельзя: на широком лице череп вылезает сквозь виски и съедает бакенбарды
    const rig = m => vao(gl, [[0, m.p, 3], [1, m.n, 3], [2, new Float32Array(m.p.length / 3 * 2), 2], [3, new Float32Array(m.p.length / 3).fill(1), 1]], m.idx);
    this.skull = rig(sk); this.neck = rig(nk); this.ears = [rig(earOcc(-1)), rig(earOcc(1))];
    this.useShell = !(debug && debug.noback);
    this.looks = {};
  }
  // образ: набор слоёв из файлов его папки
  addLook(name, A, cfg){
    const gl = this.gl, own = { texs: [], vaos: [], bufs: [] };
    const L = { name, cfg, own, stats: {} };
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    const grow = (x, y, z, k = 1) => { lo[0] = Math.min(lo[0], x * k); lo[1] = Math.min(lo[1], y * k); lo[2] = Math.min(lo[2], z * k); hi[0] = Math.max(hi[0], x * k); hi[1] = Math.max(hi[1], y * k); hi[2] = Math.max(hi[2], z * k); };
    if (A["shell.bin"]) { L.shell = this._shell(A["shell.bin"], cfg.shell || {}, own, grow); L.hairTex = texture(gl, A["hair.webp"], false, own); L.stats.shellTris = L.shell.count / 3; }
    if (A["proj.bin"]) {
      // [nv, ni, nfit] uint32; nv×3 позиции; nv×2 uv; ni индексов; nfit точек лба + их места у образца; привязка к лицу
      const pb = A["proj.bin"], ph = u32(pb, 0, 3), pnv = ph[0], pni = ph[1], pnf = ph[2];
      let po = 12;
      const pp = f32(pb, po, pnv * 3); po += pnv * 12;
      const puv = f32(pb, po, pnv * 2); po += pnv * 8;
      const pix = u32(pb, po, pni); po += pni * 4;
      L.fitIdx = u32(pb, po, pnf); po += pnf * 4;
      L.fitRef = f32(pb, po, pnf * 3); po += pnf * 12;
      const pti = u16(pb, po, pnv * 3); po += pnv * 6;
      const ptb = f32(pb, po, pnv * 3); po += pnv * 12;
      const ptd = f32(pb, po, pnv * 3); po += pnv * 12;
      const ptw = f32(pb, po, pnv);
      L.proj = vao(gl, [[0, pp, 3], [2, puv, 2], [3, ptw, 1], [4, ptd, 3], [5, pti, 3, true], [6, ptb, 3]], pnv < 65536 ? new Uint16Array(pix) : pix, own);
      L.projTex = texture(gl, A["proj.webp"], true, own);
      for (let i = 0; i < pnv; i++) grow(pp[i * 3], pp[i * 3 + 1], pp[i * 3 + 2], 1.2);
      L.stats.projTris = pni / 3;
    }
    if (A["decal.webp"]) {   // наклейка на сетку лица (усы Жоруна): чуть над кожей
      const off = new Float32Array(468);
      for (let i = 0; i < 468; i++) { const [x, y] = cv(i), ax = Math.abs(x); off[i] = (ax < 4.3 && y < -1.6 && y > -9.2) ? 0.22 : y > 5 ? 0.12 : 0.1; }
      const tris = A["decal_tris.bin"];
      L.decal = vao(gl, [[1, this.faceN, 3], [2, this.faceUV, 2], [3, off, 1], [5, this.faceIds, 1, true]], new Uint16Array(tris.buffer, tris.byteOffset, tris.byteLength / 2), own);
      L.decalTex = texture(gl, A["decal.webp"], false, own);
      L.stats.decalTris = L.decal.count / 3;
    }
    if (A["burns.bin"]) { L.burns = this._strip(A["burns.bin"], own); L.burnTex = texture(gl, A["burns.webp"], true, own); }
    if (A["teeth.bin"]) { L.teeth = this._strip(A["teeth.bin"], own); L.teethTex = texture(gl, A["teeth.webp"], true, own); }
    const det = A["det.webp"] || A["det.png"];
    if (det || A["paint.webp"]) {
      L.skin = true;
      L.detTex = det ? texture(gl, det, true, own, false) : this.neutralTex;
      L.paintTex = A["paint.webp"] ? texture(gl, A["paint.webp"], true, own) : this.clearTex;
    }
    if (A["spikes.bin"]) { L.spikes = this._spikes(A["spikes.bin"], own, grow, L); L.spikeTex = texture(gl, A["spikes.webp"], true, own); L.stats.spikes = L.spikes.n; }
    L.box = [];
    // всегда — всё лицо с полосой лба (грим и кожа доходят до линии волос)
    grow(-9, -10.5, -7); grow(9, 12, 9);
    for (const x of [lo[0], hi[0]]) for (const y of [Math.min(lo[1], -9.6), hi[1]]) for (const z of [lo[2], hi[2]]) L.box.push([x, y, z]);
    this.looks[name] = L;
    return L;
  }
  dropLook(name){
    const L = this.looks[name], gl = this.gl;
    if (!L) return;
    for (const t of L.own.texs) gl.deleteTexture(t);
    for (const b of L.own.bufs) gl.deleteBuffer(b);
    for (const v of L.own.vaos) gl.deleteVertexArray(v);
    delete this.looks[name];
  }
  // оболочка причёски из сетки радиусов (λ шаг 360/NL, μ от 12° шаг 2.5°)
  _shell(bin, sc, own, grow){
    const gl = this.gl, dv = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
    const NL = dv.getUint16(0, true), NM = dv.getUint16(2, true);
    const r = j => dv.getUint16(4 + j * 2, true) / 1000, va = j => bin[4 + NL * NM * 2 + j] / 255;
    const cols = NL + 1, LS = 360 / NL, cut = sc.cut != null ? sc.cut : 0.55, crown = sc.crown != null ? sc.crown : 1.1;
    const f0 = sc.fade ? sc.fade[0] : 0.4, f1 = sc.fade ? sc.fade[1] : 0.12;
    const P = new Float32Array(NM * cols * 3), Al = new Float32Array(NM * cols), UVs = new Float32Array(NM * cols * 2);
    const grid = (j, i) => {
      const ii = ((i % NL) + NL) % NL, jj = clamp(j, 0, NM - 1), d = dirLM(-180 + ii * LS, Math.min(179.8, 12 + jj * 2.5));
      const r0 = 1 / Math.hypot(d[0] / HA[0], d[1] / HA[1], d[2] / HA[2]), rr = r(jj * NL + ii);
      // объём сзади и по бокам меньше фото (спереди край — пряди с фото), макушка ниже: когда голова
      // наклонена к телефону, она не торчит над причёской гладкой каской
      return add(HC, mul(d, Math.max(r0 + 0.2, rr - cut - crown * Math.max(0, d[1]))));
    };
    for (let j = 0; j < NM; j++) for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      P.set(grid(j, i), k * 3); Al[k] = va(j * NL + (i % NL));
      UVs[k * 2] = i / NL; UVs[k * 2 + 1] = (Math.min(179.8, 12 + j * 2.5) - 12) / 168;
    }
    const Nn = new Float32Array(NM * cols * 3);
    for (let j = 0; j < NM; j++) for (let i = 0; i < cols; i++) {
      let n = cross(sub(grid(j, i + 1), grid(j, i - 1)), sub(grid(j + 1, i), grid(j - 1, i)));
      const d = dirLM(-180 + (i % NL) * LS, Math.min(179.8, 12 + j * 2.5));
      if (dot(n, d) < 0) n = mul(n, -1);
      Nn.set(nrm(n), (j * cols + i) * 3);
    }
    // спереди причёска — это фото; объём нужен только там, куда фото не смотрит (бока и затылок при повороте)
    for (let k = 0; k < NM * cols; k++) {
      const tt = clamp((Nn[k * 3 + 2] - f0) / (f1 - f0), 0, 1);
      Al[k] *= tt * tt * (3 - 2 * tt);
    }
    const idx = [];
    for (let j = 0; j < NM - 1; j++) for (let i = 0; i < NL; i++) {
      const a = j * cols + i, b = a + 1, c = a + cols, d = c + 1;
      if (Math.max(Al[a], Al[b], Al[c], Al[d]) <= 0.001) continue;
      idx.push(a, c, b, b, c, d);
      for (const q of [a, b, c, d]) grow(P[q * 3], P[q * 3 + 1], P[q * 3 + 2]);
    }
    // направление прядей — от лба к затылку (вдоль μ)
    const Tt = new Float32Array(NM * cols * 3);
    for (let j = 0; j < NM; j++) for (let i = 0; i < cols; i++) Tt.set(nrm(sub(grid(j + 1, i), grid(j - 1, i))), (j * cols + i) * 3);
    return vao(gl, [[0, P, 3], [1, Nn, 3], [2, UVs, 2], [3, Al, 1], [4, Tt, 3]], new Uint32Array(idx), own);
  }
  // полоса: [nv, ni] uint16; nv×(i0, i1) uint16; nv×(w, d3, n3, uv2) float32; ni индексов
  _strip(bb, own){
    const nv = bb[0] | (bb[1] << 8), ni = bb[2] | (bb[3] << 8);
    const bi = u16(bb, 4, nv * 2), bf = f32(bb, 4 + nv * 4, nv * 9), bidx = u16(bb, 4 + nv * 40, ni);
    const bw = new Float32Array(nv), bd = new Float32Array(nv * 3), bn = new Float32Array(nv * 3), buv = new Float32Array(nv * 2);
    for (let i = 0; i < nv; i++) { const o = i * 9; bw[i] = bf[o]; bd.set(bf.subarray(o + 1, o + 4), i * 3); bn.set(bf.subarray(o + 4, o + 7), i * 3); buv.set(bf.subarray(o + 7, o + 9), i * 2); }
    return vao(this.gl, [[1, bn, 3], [2, buv, 2], [4, bd, 3], [5, bi, 2, true], [6, bw, 1]], bidx, own);
  }
  // шипы: [nv, ni, ns, nfit] uint32; nv×(c3, t3, uv2, w, яркость, плотность) float32; ni uint16 индексов; ns×(начало, число, середина3);
  // nfit точек лба + их места в осях головы (подгонка под голову человека, как у причёски)
  _spikes(sb, own, grow, L){
    const h = u32(sb, 0, 4), nv = h[0], ni = h[1], ns = h[2], nf = h[3];
    let o = 16;
    const vf = f32(sb, o, nv * 11); o += nv * 44;
    const idx = u16(sb, o, ni); o += ni * 2; o += (4 - (o & 3)) & 3;
    const sp = [];
    const sd = new DataView(sb.buffer, sb.byteOffset + o, ns * 20);
    for (let k = 0; k < ns; k++) sp.push({ a: sd.getUint32(k * 20, true), n: sd.getUint32(k * 20 + 4, true), m: [sd.getFloat32(k * 20 + 8, true), sd.getFloat32(k * 20 + 12, true), sd.getFloat32(k * 20 + 16, true)] });
    o += ns * 20;
    if (nf) { L.fitIdx = u32(sb, o, nf); o += nf * 4; L.fitRef = f32(sb, o, nf * 3); }
    const C = new Float32Array(nv * 3), T = new Float32Array(nv * 3), UV = new Float32Array(nv * 2), W = new Float32Array(nv), SH = new Float32Array(nv * 2);
    for (let i = 0; i < nv; i++) {
      const q = i * 11;
      C.set(vf.subarray(q, q + 3), i * 3); T.set(vf.subarray(q + 3, q + 6), i * 3); UV.set(vf.subarray(q + 6, q + 8), i * 2); W[i] = vf[q + 8]; SH[i * 2] = vf[q + 9]; SH[i * 2 + 1] = vf[q + 10];
      const w = Math.abs(W[i]); grow(C[i * 3] - w, C[i * 3 + 1] - w, C[i * 3 + 2] - w); grow(C[i * 3] + w, C[i * 3 + 1] + w, C[i * 3 + 2] + w);
    }
    const m = vao(this.gl, [[0, C, 3], [1, T, 3], [2, UV, 2], [3, W, 1], [4, SH, 2]], idx.slice(), own, true);
    m.src = idx; m.sp = sp; m.n = ns; m.order = ""; m.buf = new Uint16Array(ni);
    return m;
  }
  upload(L){
    const b = this.lmBuf;
    for (let i = 0; i < 478; i++) { b[i * 4] = L[i * 3]; b[i * 4 + 1] = L[i * 3 + 1]; b[i * 4 + 2] = L[i * 3 + 2]; }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.lmTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 478, 1, gl.RGBA, gl.FLOAT, b);
  }
  uploadCam(src){
    const gl = this.gl, w = src.videoWidth || src.naturalWidth || src.width, h = src.videoHeight || src.naturalHeight || src.height;
    if (!w || !h) return false;
    gl.bindTexture(gl.TEXTURE_2D, this.camTex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    if (w !== this.camW || h !== this.camH) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src); this.camW = w; this.camH = h; }
    else gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, src);
    return true;
  }
  bbox(L, pose, view, lm){
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    const put = (x, y) => { const cx = view.dx + x * view.k, cy = view.dy + y * view.k; x0 = Math.min(x0, cx); x1 = Math.max(x1, cx); y0 = Math.min(y0, cy); y1 = Math.max(y1, cy); };
    for (const p of L.box) { const q = Rmul(pose.R, p); put(q[0] * pose.s + pose.t[0], q[1] * pose.s + pose.t[1]); }
    for (const i of [10, 152, 234, 454]) put(lm[i * 3], lm[i * 3 + 1]);
    const pad = 6;
    x0 = Math.floor(clamp(x0 - pad, 0, view.W)); y0 = Math.floor(clamp(y0 - pad, 0, view.H));
    x1 = Math.ceil(clamp(x1 + pad, 0, view.W)); y1 = Math.ceil(clamp(y1 + pad, 0, view.H));
    return [x0, y0, x1 - x0, y1 - y0];
  }
  render(L, pose, view, exposure, box, fit, skin){
    const gl = this.gl, c = this.canvas;
    if (c.width !== view.W || c.height !== view.H) { c.width = view.W; c.height = view.H; }
    const [bx, by, bw, bh] = box;
    gl.viewport(0, 0, c.width, c.height);
    gl.enable(gl.SCISSOR_TEST); gl.scissor(bx, c.height - by - bh, bw, bh);
    gl.clearColor(0, 0, 0, 0); gl.clearDepth(1);
    gl.depthMask(true); gl.colorMask(true, true, true, true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.CULL_FACE);
    const R = pose.R, Rc = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
    const K = fit ? fit.K : [1, 1, 1], B = fit ? fit.B : [0, 0, 0];
    const use = (pr, withFit) => {
      const u = pr.u;
      gl.useProgram(pr.p);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.lmTex); gl.uniform1i(u.u_lm, 0);
      gl.uniformMatrix3fv(u.u_R, false, Rc); gl.uniform1f(u.u_s, pose.s); gl.uniform3fv(u.u_t, pose.t);
      gl.uniform4f(u.u_view, view.dx, view.dy, view.k, 1 / (pose.s * 40));
      gl.uniform2f(u.u_canvas, c.width, c.height); gl.uniform1f(u.u_zc, pose.t[2]);
      if (u.u_exp) gl.uniform1f(u.u_exp, exposure);
      if (u.u_K) { gl.uniform3fv(u.u_K, withFit ? K : [1, 1, 1]); gl.uniform3fv(u.u_B, withFit ? B : [0, 0, 0]); }
      return u;
    };
    const tex = (u, unit, t) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(u, unit); };
    const draw = m => { gl.bindVertexArray(m.o); gl.drawElements(gl.TRIANGLES, m.count, m.type, 0); };
    const dbg = this.debug || {};
    // 1) окклюдеры: лицо, череп и шея — только глубина (в отладке — серая голова)
    gl.disable(gl.BLEND);
    if (!this.gray) gl.colorMask(false, false, false, false);
    let u = use(this.pBoundDepth); gl.uniform1f(u.u_off, this.gray ? 0 : -0.25); draw(this.faceOcc);
    gl.colorMask(true, true, true, true);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    // 2) кожа: мешки, морщины, прыщи, грим — по кадру камеры (до черепа: он у глаз и переносицы выступает из лица)
    if (L.skin && skin && !dbg.noskin) {
      gl.depthMask(false);
      u = use(this.pSkin);
      tex(u.u_det, 1, L.detTex); tex(u.u_paint, 2, L.paintTex); tex(u.u_cam, 3, this.camTex);
      gl.uniform4fv(u.u_cm, skin.cm); gl.uniform3fv(u.u_skin, skin.rgb); gl.uniform1f(u.u_light, skin.light);
      gl.uniform2f(u.u_camInv, 1 / this.camW, 1 / this.camH); gl.uniform1f(u.u_fill, L.cfg.fill ? 1 : 0);
      draw(this.skinMesh);
      gl.depthMask(true);
    }
    if (!this.gray) gl.colorMask(false, false, false, false);
    use(this.pRigidDepth); draw(this.skull); draw(this.neck);
    if (L.cfg.ears !== false) for (const e of this.ears) draw(e);
    gl.colorMask(true, true, true, true);
    // 3) объём причёски (бока и затылок)
    if (L.shell && this.useShell) {
      u = use(this.pShell, true);
      gl.uniform1f(u.u_spec, L.cfg.spec != null ? L.cfg.spec : 0.075);
      tex(u.u_tex, 1, L.hairTex);
      draw(L.shell);
    }
    gl.depthMask(false);
    // 4) наклейки на лицо: усы, бакенбарды, зубы
    if (L.decal && !dbg.nodecal) { u = use(this.pDecal); gl.uniform1f(u.u_off, 0); tex(u.u_tex, 1, L.decalTex); draw(L.decal); }
    if (L.burns && !dbg.nodecal) { u = use(this.pStrip); tex(u.u_tex, 1, L.burnTex); draw(L.burns); }
    if (L.teeth && !dbg.nodecal) { u = use(this.pStrip); tex(u.u_tex, 1, L.teethTex); draw(L.teeth); }
    // 5) шипы ирокеза — от дальних к ближним
    if (L.spikes) {
      const m = L.spikes, zs = m.sp.map((s, k) => [R[6] * s.m[0] + R[7] * s.m[1] + R[8] * s.m[2], k]).sort((a, b) => b[0] - a[0]);
      const order = zs.map(z => z[1]).join(",");
      if (order !== m.order) {
        m.order = order; let o = 0;
        for (const [, k] of zs) { const s = m.sp[k]; m.buf.set(m.src.subarray(s.a, s.a + s.n), o); o += s.n; }
        gl.bindVertexArray(m.o); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.ib); gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, m.buf);
      }
      u = use(this.pSpike, true); tex(u.u_tex, 1, L.spikeTex); draw(m);
    }
    // 6) причёска с фото — поверх всего, что за ней
    if (L.proj && !dbg.noshell) { u = use(this.pProj, true); tex(u.u_tex, 1, L.projTex); draw(L.proj); }
    gl.depthMask(true);
    gl.bindVertexArray(null);
    gl.disable(gl.SCISSOR_TEST);
  }
}

/* ---------------- трекер: отдельный поток (worker) или, если не вышло, основной */
async function makeTracker(mode, delegate, onProgress, useWorker){
  const { FaceLandmarker, FilesetResolver } = await import(BASE + "mp/vision_bundle.js");
  let simd = true;
  try { simd = await FilesetResolver.isSimdSupported(); } catch (e) {}
  const name = simd ? "vision_wasm_internal" : "vision_wasm_nosimd_internal";
  const total = (simd ? 11153617 : 10481398) + 3758596;
  let got = 0;
  const tick = n => { got += n; if (onProgress) onProgress(Math.min(0.99, got / total)); };
  const [wasm, model] = await Promise.all([fetchBytes(BASE + `mp/wasm/${name}.wasm`, tick), fetchBytes(BASE + "mp/face_landmarker.task", tick)]);
  const wasmUrl = URL.createObjectURL(new Blob([wasm], { type: "application/wasm" }));
  const loader = BASE + `mp/wasm/${name}.js`;
  // на iPhone GPU-режим MediaPipe нестабилен — там считаем на процессоре
  const ios = /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  // процессор (XNNPACK) по умолчанию: заводится быстро и одинаково везде; видеокарту пробуем потом, в превью
  const want = delegate || "CPU";
  if (useWorker && mode === "VIDEO" && typeof Worker !== "undefined" && typeof OffscreenCanvas !== "undefined" && typeof createImageBitmap === "function") {
    let w = null;
    try {
      w = new Worker(BASE + "track-worker.js" + VQ);
      const ready = await new Promise((res, rej) => {
        const to = setTimeout(() => rej(new Error("worker timeout")), 90000);   // медленный телефон честно грузит дольше — не бросаем поток раньше времени
        w.onmessage = e => { if (e.data.type === "ready") { clearTimeout(to); res(e.data); } else if (e.data.type === "error") { clearTimeout(to); rej(new Error(e.data.error)); } };
        w.onerror = e => { clearTimeout(to); rej(new Error(e.message || "worker error")); };
        const mcopy = model.slice().buffer;
        w.postMessage({ type: "init", bundle: BASE + "mp/vision_bundle_cjs.js", loader, wasm: wasmUrl, model: mcopy, delegate: want }, [mcopy]);
      });
      w.onerror = null;
      return { kind: "worker", worker: w, delegate: ready.delegate, ios };
    } catch (e) { console.warn("mask: worker failed, using main thread", e); try { w && w.terminate(); } catch (e2) {} }
  }
  const files = { wasmLoaderPath: loader, wasmBinaryPath: wasmUrl };
  const make = d => FaceLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetBuffer: model, delegate: d }, runningMode: mode, numFaces: 1,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false,
  });
  let fl, used = want;
  try { fl = await make(want); } catch (e) { if (want === "CPU") throw e; used = "CPU"; fl = await make("CPU"); }
  return { kind: "main", fl, delegate: used, make };
}

/* ---------------- снаружи */
export async function createMask({ look = "zhorun", mode = "VIDEO", delegate = null, debug = false, onProgress = null, worker = true, fresh = true, ring: ringSize = 8, maxDelay = 240 } = {}){
  if (!LOOKS[look]) look = "zhorun";
  const r = new Renderer(debug);
  const [, tr] = await Promise.all([loadLook(look, !!(debug && debug.lenient)).then(A => r.addLook(look, A, LOOKS[look])), makeTracker(mode, delegate, onProgress, worker)]);
  const sm = new Smoother();
  const raw = new Float32Array(478 * 3);
  const lum = document.createElement("canvas"); lum.width = 32; lum.height = 8;
  const lg = lum.getContext("2d", { willReadFrequently: true });
  const small = document.createElement("canvas"), sg = small.getContext("2d");
  // кольцо последних кадров камеры: на экран идёт кадр чуть из прошлого — тот, для которого лицо уже найдено,
  // а маска между найденными кадрами плавно доводится (как в масках Instagram: картинка и маска всегда вместе)
  const RING = Math.max(4, ringSize | 0), HIST = 12;
  const ring = Array.from({ length: RING }, () => { const c = document.createElement("canvas"); return { c, g: c.getContext("2d"), t: -1, w: 0, h: 0 }; });
  let ringN = 0, seq = 0;
  const lerpBuf = new Float32Array(478 * 3);
  function lerpPose(a, b, w){
    let q = b.q; if (a.q[0] * q[0] + a.q[1] * q[1] + a.q[2] * q[2] + a.q[3] * q[3] < 0) q = q.map(x => -x);
    const qq = [0, 1, 2, 3].map(k => a.q[k] + (q[k] - a.q[k]) * w), ql = Math.hypot(...qq);
    for (let k = 0; k < 4; k++) qq[k] /= ql;
    for (let i = 0; i < 478 * 3; i++) lerpBuf[i] = a.lm[i] + (b.lm[i] - a.lm[i]) * w;
    return { s: a.s + (b.s - a.s) * w, q: qq, R: quatToR(qq), t: [0, 1, 2].map(k => a.tt[k] + (b.tt[k] - a.tt[k]) * w), lm: lerpBuf };
  }

  const M = {
    raw, get delegate(){ return tr.delegate; }, kind: tr.kind,
    look, looks: LOOK_NAMES, _lp: {}, skin: [0.7, 0.52, 0.44],
    get stats(){ const L = r.looks[this.look]; return L ? L.stats : {}; },
    // сменить образ: трекер тот же, грузятся только файлы образа (сотни КБ)
    setLook(name){
      if (!LOOKS[name]) return Promise.reject(new Error("unknown look " + name));
      this._want = name;
      const done = () => { if (this._want === name) { this.look = name; for (const k of Object.keys(r.looks)) if (k !== name) r.dropLook(k); } return name; };
      if (r.looks[name]) return Promise.resolve(done());
      const p = this._lp[name] || (this._lp[name] = loadLook(name).then(A => { if (!r.looks[name]) r.addLook(name, A, LOOKS[name]); }).finally(() => { delete this._lp[name]; }));
      return p.then(done);
    },
    found: false, vis: 0, exposure: 1, pose: null, lm: null, srcW: 0, srcH: 0, miss: 0,
    latency: 0, detMs: 0, results: 0, lastResult: 0, inflight: null, rate: 0, bmpMs: 0,
    delay: 90, frozen: false, busy: false, hist: [], sent: -1,
    get ready(){ return ringN > 0; },
    _ingest(lmNorm, W, H, tMs, smooth){
      this.srcW = W; this.srcH = H;
      for (let i = 0; i < 478; i++) { raw[i * 3] = lmNorm[i * 3] * W; raw[i * 3 + 1] = lmNorm[i * 3 + 1] * H; raw[i * 3 + 2] = lmNorm[i * 3 + 2] * W; }
      const o = sm.apply(raw, tMs / 1000, smooth);
      this.lm = o.lm; this.pose = o.pose;
      this.found = true; this.miss = 0;
      if (smooth) {
        const h = this.hist;
        h.push({ t: tMs, s: o.pose.s, q: o.pose.q.slice(), tt: o.pose.t.slice(), lm: o.lm.slice() });
        if (h.length > HIST) h.shift();
      }
    },
    _lost(){ if (++this.miss > 3) { this.found = false; sm.reset(); this.hist.length = 0; } },
    // новый кадр камеры → в кольцо
    _grab(video, t){
      const vw = video.videoWidth, vh = video.videoHeight;
      if (!vw || !vh) return;
      const k = Math.min(1, 1280 / Math.max(vw, vh)), w = Math.round(vw * k), h = Math.round(vh * k);
      const sl = ring[ringN % RING];
      if (sl.c.width !== w || sl.c.height !== h) { sl.c.width = w; sl.c.height = h; }
      sl.g.drawImage(video, 0, 0, w, h);
      sl.t = t; sl.w = w; sl.h = h;
      ringN++;
    },
    _hook(video){
      if (this._rv === video) return;
      this._rv = video; this._rvfc = false;
      if (fresh && video.requestVideoFrameCallback) {
        this._rvfc = true;
        const cb = now => {
          if (this._rv !== video) return;
          if (video.readyState >= 2 && performance.now() - this._pumpAt < 600) { this._grab(video, now); this._send(video); }
          video.requestVideoFrameCallback(cb);
        };
        video.requestVideoFrameCallback(cb);
      }
    },
    // вызывается каждый кадр отрисовки
    pump(video, now = performance.now()){
      if (!video || video.readyState < 2 || !video.videoWidth) return;
      this._pumpAt = now;
      this._hook(video);
      if (!this._rvfc && video.currentTime !== this._ct) { this._ct = video.currentTime; this._grab(video, now); }
      if (this.inflight && now - this.inflight.t0 > 1500) this.inflight = null;
      this._send();
    },
    // самый свежий кадр — в поиск лица (по одному: пока ищем, свежие кадры копятся в кольце)
    _send(fromVideo = null){
      if (this.inflight || !ringN) return;
      const n = ringN - 1;
      if (n === this.sent) return;
      this.sent = n;
      const sl = ring[n % RING], id = ++seq, now = performance.now();
      const kk = Math.min(1, 480 / Math.max(sl.w, sl.h)), sw = Math.round(sl.w * kk), sh = Math.round(sl.h * kk);
      this.inflight = { id, t0: now, t: sl.t, slot: sl, vw: sl.w, vh: sl.h };
      if (tr.kind === "worker") {
        const post = bmp => { const d = performance.now() - now; this.bmpMs = this.bmpMs ? this.bmpMs * 0.9 + d * 0.1 : d; tr.worker.postMessage({ type: "frame", id, ts: Math.round(sl.t), bmp }, [bmp]); };
        const viaCanvas = () => { if (small.width !== sw || small.height !== sh) { small.width = sw; small.height = sh; } sg.drawImage(sl.c, 0, 0, sw, sh); return createImageBitmap(small); };
        const src = fromVideo || sl.c;
        (this._resizeOK !== false ? createImageBitmap(src, { resizeWidth: sw, resizeHeight: sh, resizeQuality: "low" }).catch(() => { this._resizeOK = false; return viaCanvas(); }) : viaCanvas())
          .then(post).catch(() => { this.inflight = null; });
      } else {
        if (small.width !== sw || small.height !== sh) { small.width = sw; small.height = sh; }
        sg.drawImage(sl.c, 0, 0, sw, sh);
        const t0 = performance.now();
        let res = null;
        try { res = tr.fl.detectForVideo(small, Math.round(sl.t)); } catch (e) { console.warn("mask: detect", e); }
        const f = res && res.faceLandmarks && res.faceLandmarks[0];
        let lm = null;
        if (f) { lm = new Float32Array(f.length * 3); for (let i = 0; i < f.length; i++) { lm[i * 3] = f[i].x; lm[i * 3 + 1] = f[i].y; lm[i * 3 + 2] = f[i].z; } }
        this._result(id, lm, performance.now() - t0);
      }
    },
    _result(id, lm, ms, skip){
      const f = this.inflight;
      if (!f || f.id !== id) return;
      this.inflight = null;
      if (!skip) {
        const now = performance.now();
        // медианы по последним кадрам: прогрев и случайные долгие кадры не сбивают решения
        const H = this._st || (this._st = { lat: [], det: [], at: [], gap: [] });
        const push = (a, v) => { a.push(v); if (a.length > 20) a.shift(); };
        push(H.lat, now - f.t); push(H.det, ms); push(H.at, now);
        if (this._lastT != null) push(H.gap, f.t - this._lastT);
        this._lastT = f.t;
        const med = a => { const b = a.slice().sort((x, y) => x - y); return b[b.length >> 1]; };
        this.latency = med(H.lat); this.detMs = med(H.det);
        this.rate = H.at.length > 2 ? (H.at.length - 1) * 1000 / Math.max(1, now - H.at[0]) : 0;
        this.results++; this._sinceSwitch = (this._sinceSwitch || 0) + 1;
        this.lastResult = now;
        if (lm) { this._ingest(lm, f.vw, f.vh, f.t, true); if (this.results % 15 === 1) this._light(f.slot.c); }
        else this._lost();
        this._adapt();
        this._tune();
      }
      this._send();     // сразу следующий кадр: поиск лица не простаивает
    },
    // насколько показывать картинку «из прошлого»: чтобы для показанного кадра лицо уже было найдено
    _adapt(){
      const H = this._st;
      if (this.frozen || !H || H.lat.length < 4) return;
      const pq = (a, q) => { const b = a.slice().sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(b.length * q))]; };
      const target = clamp(pq(H.lat, 0.8) + 0.6 * (H.gap.length ? pq(H.gap, 0.8) : 33) + 10, 45, maxDelay);
      this.delay += (target - this.delay) * (target > this.delay ? 0.3 : 0.04);
    },
    // кадр для экрана: на delay мс раньше «сейчас»
    frameFor(now){
      if (!ringN) return null;
      const target = now - this.delay;
      let best = null, oldest = null;
      for (let i = Math.max(0, ringN - RING); i < ringN; i++) {
        const sl = ring[i % RING];
        if (!oldest || sl.t < oldest.t) oldest = sl;
        if (sl.t <= target && (!best || sl.t > best.t)) best = sl;
      }
      const sl = best || oldest;
      return { src: sl.c, w: sl.w, h: sl.h, t: sl.t };
    },
    // поза и точки лица на момент кадра t: между двумя найденными — плавно, после последнего — чуть вперёд
    _poseAt(t){
      const h = this.hist;
      if (!h.length) return null;
      let i = h.length - 1;
      if (t >= h[i].t) {
        if (i > 0) {
          const a = h[i - 1], b = h[i], dt = b.t - a.t, ex = Math.min(t - b.t, 70, dt) * 0.8;
          if (dt > 5 && ex > 1) return lerpPose(a, b, 1 + ex / dt);
        }
        const b = h[i];
        return { s: b.s, q: b.q, R: quatToR(b.q), t: b.tt, lm: b.lm };
      }
      while (i > 0 && h[i - 1].t > t) i--;
      if (i === 0) { const a = h[0]; return { s: a.s, q: a.q, R: quatToR(a.q), t: a.tt, lm: a.lm }; }
      const a = h[i - 1], b = h[i];
      return lerpPose(a, b, (t - a.t) / Math.max(1, b.t - a.t));
    },
    // на Android пробуем и видеокарту, и процессор: оставляем то, что быстрее на этом телефоне
    _tune(){
      if (tr.kind !== "worker" || this._tuneDone || tr.ios || this.busy || !this.allowGpuTrial) return;
      if (this._switching || this._sinceSwitch < 12) return;
      if (!this._trial) {
        if (this.detMs <= 26) { if (this.results > 40) this._tuneDone = true; return; }
        this._trial = { from: tr.delegate, ms: this.detMs };
        this._switching = true;
        tr.worker.postMessage({ type: "switch", delegate: tr.delegate === "GPU" ? "CPU" : "GPU" });
      } else {
        if (tr.delegate !== this._trial.from && this.detMs > this._trial.ms * 0.9) {
          this._switching = true;
          tr.worker.postMessage({ type: "switch", delegate: this._trial.from });
        }
        this._tuneDone = true;
      }
    },
    // свет и цвет кожи: среднее по лицу (яркость) и три пятна кожи — лоб и щёки (цвет для грима)
    _light(src, direct = false){
      if (!src) return;
      try {
        const L = this.lm, xs = [234, 454, 10, 152].map(i => L[i * 3]), ys = [234, 454, 10, 152].map(i => L[i * 3 + 1]);
        const x0 = Math.max(0, Math.min(...xs)), x1 = Math.min(this.srcW, Math.max(...xs)), y0 = Math.max(0, Math.min(...ys)), y1 = Math.min(this.srcH, Math.max(...ys));
        if (x1 - x0 < 8 || y1 - y0 < 8) return;
        lg.clearRect(0, 0, 32, 8);
        lg.drawImage(src, x0, y0, x1 - x0, y1 - y0, 0, 0, 8, 8);
        const ps = Math.max(2, (this.pose ? this.pose.s : 10) * 0.7);
        [151, 50, 280].forEach((i, k) => lg.drawImage(src, L[i * 3] - ps, L[i * 3 + 1] - ps, 2 * ps, 2 * ps, 8 + k * 8, 0, 8, 8));
        const d = lg.getImageData(0, 0, 32, 8).data;
        let s = 0;
        const pc = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for (let y = 0; y < 8; y++) for (let x = 0; x < 32; x++) {
          const i = (y * 32 + x) * 4;
          if (x < 8) s += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
          else { const p = pc[(x >> 3) - 1]; p[0] += d[i]; p[1] += d[i + 1]; p[2] += d[i + 2]; }
        }
        const Y = s / 64 / 255, ex = clamp(Math.pow(Y / 0.52, 0.7), 0.55, 1.35);
        this.exposure = direct ? ex : this.exposure * 0.5 + ex * 0.5;
        const med = k => { const v = pc.map(p => p[k] / 64 / 255).sort((a, b) => a - b); return v[1]; };
        const sk = [med(0), med(1), med(2)];
        if (sk[0] + sk[1] + sk[2] > 0.06) this.skin = direct ? sk : this.skin.map((v, k) => v * 0.6 + sk[k] * 0.4);
      } catch (e) {}
    },
    trackImage(img){
      const res = tr.fl.detect(img);
      const f = res.faceLandmarks && res.faceLandmarks[0];
      if (!f) { this.found = false; return false; }
      const lm = new Float32Array(f.length * 3);
      for (let i = 0; i < f.length; i++) { lm[i * 3] = f[i].x; lm[i * 3 + 1] = f[i].y; lm[i * 3 + 2] = f[i].z; }
      this._ingest(lm, img.naturalWidth || img.width, img.naturalHeight || img.height, 0, false);
      this._light(img, true);
      this.vis = 1;
      return true;
    },
    // view: как кадр лёг на холст (как coverDraw): dx, dy, k — пикселей холста на пиксель кадра; live — поверх живого видео
    // src — кадр, который лежит на холсте под маской (нужен слою кожи)
    draw(g, view, alpha = 1, tFrame = null, src = null){
      const LK = r.looks[this.look];
      if (!LK) return;
      const P = tFrame != null && this.hist.length ? this._poseAt(tFrame) : (this.pose ? { s: this.pose.s, q: this.pose.q, R: this.pose.R, t: this.pose.t, lm: this.lm } : null);
      this.vis += ((this.found && P ? 1 : 0) - this.vis) * 0.4;
      if (this.vis < 0.02 || !P) return;
      const pose = { s: P.s, R: P.R, t: P.t, q: P.q }, lm = P.lm;
      r.upload(lm);
      const box = r.bbox(LK, pose, view, lm);
      if (box[2] < 2 || box[3] < 2) return;
      let skin = null;
      if (LK.skin && src && r.uploadCam(src)) {
        const Ys = 0.299 * this.skin[0] + 0.587 * this.skin[1] + 0.114 * this.skin[2];
        skin = { cm: [view.dx, view.dy, 1 / (view.k * r.camW), 1 / (view.k * r.camH)], rgb: this.skin, light: clamp(0.55 + 0.6 * Ys, 0.75, 1.05) };
      }
      r.render(LK, pose, view, this.exposure, box, this._fit(LK), skin);
      g.save();
      g.globalAlpha = Math.min(1, this.vis) * alpha;
      if (view.mirror) { g.translate(view.W, 0); g.scale(-1, 1); }
      g.drawImage(r.canvas, box[0], box[1], box[2], box[3], box[0], box[1], box[2], box[3]);
      g.restore();
    },
    // причёска образца → под лоб и виски этого человека: x, y — масштаб и сдвиг, z — сдвиг (по сглаженной форме лица)
    _fit(LK){
      LK = LK || r.looks[this.look];
      const d = sm.def;
      if (!d || !LK || !LK.fitIdx) return null;
      const I = LK.fitIdx, Z = LK.fitRef, n = I.length;
      const K = [1, 1, 1], B = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        let mz = 0, mu = 0;
        for (let j = 0; j < n; j++) { mz += Z[j * 3 + k]; mu += CANON_V[I[j] * 3 + k] + d[I[j] * 3 + k]; }
        mz /= n; mu /= n;
        let kk = 1;
        if (k < 2) {
          let cv = 0, vz = 0;
          for (let j = 0; j < n; j++) { const a = Z[j * 3 + k] - mz, b = CANON_V[I[j] * 3 + k] + d[I[j] * 3 + k] - mu; cv += a * b; vz += a * a; }
          kk = clamp(cv / Math.max(vz, 1e-6), 0.85, 1.2);
        }
        K[k] = kk; B[k] = mu - kk * mz;
      }
      return { K, B };
    },
    debugTurn(yaw, pitch){
      const P = this.pose, c = add(P.t, mul(Rmul(P.R, HC), P.s));
      const cy = Math.cos(yaw * DEG), sy = Math.sin(yaw * DEG), cp = Math.cos(pitch * DEG), sp = Math.sin(pitch * DEG);
      const L = this.base0 || (this.base0 = this.lm.slice()), out = new Float32Array(L.length);
      for (let i = 0; i < 478; i++) {
        let x = L[i * 3] - c[0], y = L[i * 3 + 1] - c[1], z = L[i * 3 + 2] - c[2];
        [x, z] = [x * cy - z * sy, x * sy + z * cy];
        [y, z] = [y * cp - z * sp, y * sp + z * cp];
        out[i * 3] = x + c[0]; out[i * 3 + 1] = y + c[1]; out[i * 3 + 2] = z + c[2];
      }
      const o = sm.apply(out, 0, false);
      this.lm = o.lm.slice(); this.pose = o.pose; r.upload(this.lm);
    },
    faceBox(view){
      if (!this.lm) return null;
      const L = this.lm; let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
      for (let i = 0; i < 468; i++) { x0 = Math.min(x0, L[i * 3]); x1 = Math.max(x1, L[i * 3]); y0 = Math.min(y0, L[i * 3 + 1]); y1 = Math.max(y1, L[i * 3 + 1]); }
      const k = view ? view.k : 1, dx = view ? view.dx : 0, dy = view ? view.dy : 0;
      return [dx + x0 * k, dy + y0 * k, dx + x1 * k, dy + y1 * k];
    },
  };
  if (tr.kind === "worker") tr.worker.onmessage = e => {
    const d = e.data;
    if (d.type === "result") M._result(d.id, d.lm, d.ms, d.skip);
    else if (d.type === "switched") {
      M._switching = false;
      if (d.delegate !== tr.delegate) { tr.delegate = d.delegate; M._sinceSwitch = 0; if (M._st) for (const k of Object.keys(M._st)) M._st[k].length = 0; }
      else if (M._trial && !M._tuneDone) M._tuneDone = true;   // другой режим не завёлся — остаёмся
    }
  };
  return M;
}
