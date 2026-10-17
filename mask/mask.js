// Маска «Жорун»: кок, усы-подкова, бакенбарды и прядь на лбу сняты с фото самого Жоруна.
// Кок — оболочка по силуэтам головы с разных ракурсов, на неё спроецированы фото;
// усы и бакенбарды — «наклейка» на сетку лица, поэтому двигаются вместе с губами и щеками.
// Лицо ищет MediaPipe Face Landmarker в отдельном потоке, а на экран идёт тот же кадр камеры,
// на котором лицо нашли, — маска не отстаёт от головы.
import { CANON_V, CANON_F, CANON_UV } from "./face-canon.js";

const BASE = new URL("./", import.meta.url).href;
const VQ = "?v=4";   // версия ресурсов маски: после обновления браузер не смешает старые файлы с новыми
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
Smoother.POSE_MIN = 1.5; Smoother.POSE_BETA = 0.5; Smoother.DEF_MIN = 3.0; Smoother.DEF_BETA = 1.0;

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
out vec2 v_uv; out float v_a; out vec3 v_n; out vec3 v_t;
void main(){ vec3 P = u_s * (u_R * a_p) + u_t; v_n = u_R * a_n; v_t = u_R * a_t; v_uv = a_uv; v_a = a_a; gl_Position = clipOf(P); }`;
const VS_BOUND = VHEAD + `
layout(location=1) in vec3 a_n; layout(location=2) in vec2 a_uv; layout(location=3) in float a_off; layout(location=5) in uint a_i;
uniform float u_off;
out vec2 v_uv; out float v_a; out vec3 v_n;
void main(){ vec3 N = normalize(u_R * a_n); vec3 P = lm(a_i) + N * ((a_off + u_off) * u_s); v_n = N; v_uv = a_uv; v_a = 1.0; gl_Position = clipOf(P); }`;
// бакенбарды: смесь двух точек контура лица + смещение в осях головы
const VS_STRIP = VHEAD + `
layout(location=1) in vec3 a_n; layout(location=2) in vec2 a_uv; layout(location=4) in vec3 a_d; layout(location=5) in uvec2 a_ii; layout(location=6) in float a_w;
out vec2 v_uv; out float v_a; out vec3 v_n;
void main(){ vec3 N = normalize(u_R * a_n); vec3 P = mix(lm(a_ii.x), lm(a_ii.y), a_w) + (u_R * a_d) * u_s; v_n = N; v_uv = a_uv; v_a = 1.0; gl_Position = clipOf(P); }`;
const FS_DEPTH = `#version 300 es
precision mediump float; out vec4 o; void main(){ o = vec4(0.0); }`;
const FS_GRAY = `#version 300 es
precision mediump float; in vec3 v_n; out vec4 o;
void main(){ vec3 N = normalize(v_n); float l = 0.35 + 0.65 * max(0.0, dot(N, normalize(vec3(0.3, -0.5, -0.8)))); o = vec4(vec3(0.62, 0.6, 0.58) * l, 1.0); }`;
const FS_SHELL = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform float u_exp;
in vec2 v_uv; in float v_a; in vec3 v_n; in vec3 v_t; out vec4 o;
void main(){
  vec4 c = texture(u_tex, v_uv);                  // цвет уже умножен на альфу
  vec3 N = normalize(v_n);
  float rim = smoothstep(-0.04, 0.02, -N.z);      // бока причёски плотные, изнанка не рисуется
  float k = v_a * rim;
  if (c.a * k < 0.02) discard;
  vec3 L = normalize(vec3(0.15, -0.75, -0.65));   // свет сверху-спереди (y вниз, z от камеры)
  float dif = 0.8 + 0.28 * max(0.0, dot(N, L));
  // блеск вдоль прядей (Kajiya–Kay): полоса бликов «едет» по коку, когда голова поворачивается
  vec3 H = normalize(L + vec3(0.0, 0.0, -1.0));
  float tl = length(v_t), spec = 0.0;
  if (tl > 1e-4) { float th = dot(v_t / tl, H); spec = pow(sqrt(max(0.0, 1.0 - th * th)), 14.0) * smoothstep(0.0, 0.5, dot(N, H)); }
  vec3 col = c.rgb * dif + vec3(0.92, 0.9, 0.87) * (spec * 0.075 * c.a);
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
function vao(gl, attrs, idx){
  const o = gl.createVertexArray(); gl.bindVertexArray(o);
  for (const [loc, data, size, int] of attrs) {
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    if (int) gl.vertexAttribIPointer(loc, size, gl.UNSIGNED_SHORT, 0, 0); else gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }
  const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return { o, count: idx.length, type: idx instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT };
}
function texture(gl, img, clampS){
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, clampS ? gl.CLAMP_TO_EDGE : gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
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

class Renderer {
  constructor(assets, debug){
    const cvs = this.canvas = document.createElement("canvas");
    cvs.width = 720; cvs.height = 1280;
    const gl = this.gl = cvs.getContext("webgl2", { antialias: false, alpha: true, premultipliedAlpha: true, depth: true, preserveDrawingBuffer: false, powerPreference: "high-performance" });
    if (!gl) throw new Error("no webgl2");
    this.debug = debug;
    this.pRigidDepth = program(gl, VS_RIGID, debug ? FS_GRAY : FS_DEPTH);
    this.pBoundDepth = program(gl, VS_BOUND, debug ? FS_GRAY : FS_DEPTH);
    this.pShell = program(gl, VS_RIGID, FS_SHELL);
    this.pDecal = program(gl, VS_BOUND, FS_DECAL);
    this.pStrip = program(gl, VS_STRIP, FS_DECAL);
    this.lmTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.lmTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, 478, 1);
    for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
    this.lmBuf = new Float32Array(478 * 4);
    this.hairTex = texture(gl, assets.hair);
    this.decalTex = texture(gl, assets.decal);
    this.burnTex = texture(gl, assets.burnImg, true);

    // оболочка кока из сетки радиусов (λ шаг 360/NL, μ от 12° шаг 2.5°)
    const dv = new DataView(assets.shell.buffer, assets.shell.byteOffset, assets.shell.byteLength);
    const NL = dv.getUint16(0, true), NM = dv.getUint16(2, true);
    const r = j => dv.getUint16(4 + j * 2, true) / 1000, va = j => assets.shell[4 + NL * NM * 2 + j] / 255;
    const cols = NL + 1, LS = 360 / NL;
    const P = new Float32Array(NM * cols * 3), A = new Float32Array(NM * cols), UVs = new Float32Array(NM * cols * 2);
    const grid = (j, i) => { const ii = ((i % NL) + NL) % NL, jj = clamp(j, 0, NM - 1); const d = dirLM(-180 + ii * LS, Math.min(179.8, 12 + jj * 2.5)); return add(HC, mul(d, r(jj * NL + ii))); };
    for (let j = 0; j < NM; j++) for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      P.set(grid(j, i), k * 3); A[k] = va(j * NL + (i % NL));
      UVs[k * 2] = i / NL; UVs[k * 2 + 1] = (Math.min(179.8, 12 + j * 2.5) - 12) / 168;
    }
    const Nn = new Float32Array(NM * cols * 3);
    for (let j = 0; j < NM; j++) for (let i = 0; i < cols; i++) {
      let n = cross(sub(grid(j, i + 1), grid(j, i - 1)), sub(grid(j + 1, i), grid(j - 1, i)));
      const d = dirLM(-180 + (i % NL) * LS, Math.min(179.8, 12 + j * 2.5));
      if (dot(n, d) < 0) n = mul(n, -1);
      Nn.set(nrm(n), (j * cols + i) * 3);
    }
    const idx = [];
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (let j = 0; j < NM - 1; j++) for (let i = 0; i < NL; i++) {
      const a = j * cols + i, b = a + 1, c = a + cols, d = c + 1;
      if (Math.max(A[a], A[b], A[c], A[d]) <= 0.001) continue;
      idx.push(a, c, b, b, c, d);
      for (const q of [a, b, c, d]) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P[q * 3 + k]); hi[k] = Math.max(hi[k], P[q * 3 + k]); }
    }
    this.box = [];
    for (const x of [lo[0], hi[0]]) for (const y of [Math.min(lo[1], -9.6), hi[1]]) for (const z of [lo[2], hi[2]]) this.box.push([x, y, z]);
    // направление прядей — от лба к затылку (вдоль μ)
    const Tt = new Float32Array(NM * cols * 3);
    for (let j = 0; j < NM; j++) for (let i = 0; i < cols; i++) Tt.set(nrm(sub(grid(j + 1, i), grid(j - 1, i))), (j * cols + i) * 3);
    this.shell = vao(gl, [[0, P, 3], [1, Nn, 3], [2, UVs, 2], [3, A, 1], [4, Tt, 3]], new Uint32Array(idx));
    // сетка лица: наклейка и окклюдер
    const N = faceNormals(), uv = new Float32Array(468 * 2), off = new Float32Array(468), ids = new Uint16Array(468);
    for (let i = 0; i < 468; i++) {
      ids[i] = i; uv[i * 2] = CANON_UV[i * 2]; uv[i * 2 + 1] = 1 - CANON_UV[i * 2 + 1];
      const [x, y] = cv(i), ax = Math.abs(x);
      off[i] = (ax < 4.3 && y < -1.6 && y > -9.2) ? 0.22 : y > 5 ? 0.12 : 0.1;   // усы чуть выше кожи — у них есть объём
    }
    this.decal = vao(gl, [[1, N, 3], [2, uv, 2], [3, off, 1], [5, ids, 1, true]], new Uint16Array(assets.tris));
    const occOff = new Float32Array(468);
    for (let i = 0; i < 468; i++) { const ax = Math.abs(CANON_V[i * 3]); occOff[i] = -0.7 * clamp((ax - 5.5) / 2.1, 0, 1); }
    this.faceOcc = vao(gl, [[1, N, 3], [2, uv, 2], [3, occOff, 1], [5, ids, 1, true]], new Uint16Array(CANON_F));
    const sk = ellipsoid(HC, HA.map(a => a - 0.6)), nk = neck();   // больше нельзя: на широком лице череп вылезает сквозь виски и съедает бакенбарды
    const rig = m => vao(gl, [[0, m.p, 3], [1, m.n, 3], [2, new Float32Array(m.p.length / 3 * 2), 2], [3, new Float32Array(m.p.length / 3).fill(1), 1]], m.idx);
    this.skull = rig(sk); this.neck = rig(nk); this.ears = [rig(earOcc(-1)), rig(earOcc(1))];
    // бакенбарды: [nv, ni] uint16; nv×(i0, i1) uint16; nv×(w, d3, n3, uv2) float32; ni индексов
    const bb = assets.burns, nv = bb[0] | (bb[1] << 8), ni = bb[2] | (bb[3] << 8);
    const bi = new Uint16Array(bb.slice(4, 4 + nv * 4).buffer), bf = new Float32Array(bb.slice(4 + nv * 4, 4 + nv * 40).buffer);
    const bidx = new Uint16Array(bb.slice(4 + nv * 40, 4 + nv * 40 + ni * 2).buffer);
    const bw = new Float32Array(nv), bd = new Float32Array(nv * 3), bn = new Float32Array(nv * 3), buv = new Float32Array(nv * 2);
    for (let i = 0; i < nv; i++) { const o = i * 9; bw[i] = bf[o]; bd.set(bf.subarray(o + 1, o + 4), i * 3); bn.set(bf.subarray(o + 4, o + 7), i * 3); buv.set(bf.subarray(o + 7, o + 9), i * 2); }
    this.burns = vao(gl, [[1, bn, 3], [2, buv, 2], [4, bd, 3], [5, bi, 2, true], [6, bw, 1]], bidx);
    this.stats = { shellTris: idx.length / 3, decalTris: assets.tris.length / 3, burnTris: ni / 3 };
  }
  upload(L){
    const b = this.lmBuf;
    for (let i = 0; i < 478; i++) { b[i * 4] = L[i * 3]; b[i * 4 + 1] = L[i * 3 + 1]; b[i * 4 + 2] = L[i * 3 + 2]; }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.lmTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 478, 1, gl.RGBA, gl.FLOAT, b);
  }
  bbox(pose, view, L){
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    const put = (x, y) => { const cx = view.dx + x * view.k, cy = view.dy + y * view.k; x0 = Math.min(x0, cx); x1 = Math.max(x1, cx); y0 = Math.min(y0, cy); y1 = Math.max(y1, cy); };
    for (const p of this.box) { const q = Rmul(pose.R, p); put(q[0] * pose.s + pose.t[0], q[1] * pose.s + pose.t[1]); }
    for (const i of [10, 152, 234, 454]) put(L[i * 3], L[i * 3 + 1]);
    const pad = 6;
    x0 = Math.floor(clamp(x0 - pad, 0, view.W)); y0 = Math.floor(clamp(y0 - pad, 0, view.H));
    x1 = Math.ceil(clamp(x1 + pad, 0, view.W)); y1 = Math.ceil(clamp(y1 + pad, 0, view.H));
    return [x0, y0, x1 - x0, y1 - y0];
  }
  render(pose, view, exposure, box){
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
    const use = (pr) => {
      const u = pr.u;
      gl.useProgram(pr.p);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.lmTex); gl.uniform1i(u.u_lm, 0);
      gl.uniformMatrix3fv(u.u_R, false, Rc); gl.uniform1f(u.u_s, pose.s); gl.uniform3fv(u.u_t, pose.t);
      gl.uniform4f(u.u_view, view.dx, view.dy, view.k, 1 / (pose.s * 40));
      gl.uniform2f(u.u_canvas, c.width, c.height); gl.uniform1f(u.u_zc, pose.t[2]);
      if (u.u_exp) gl.uniform1f(u.u_exp, exposure);
      return u;
    };
    const draw = m => { gl.bindVertexArray(m.o); gl.drawElements(gl.TRIANGLES, m.count, m.type, 0); };
    // 1) окклюдеры: лицо, череп и шея — только глубина (в отладке — серая голова)
    gl.disable(gl.BLEND);
    if (!this.debug) gl.colorMask(false, false, false, false);
    let u = use(this.pBoundDepth); gl.uniform1f(u.u_off, this.debug ? 0 : -0.25); draw(this.faceOcc);
    use(this.pRigidDepth); draw(this.skull); draw(this.neck); for (const e of this.ears) draw(e);
    gl.colorMask(true, true, true, true);
    // 2) кок
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    u = use(this.pShell);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.hairTex); gl.uniform1i(u.u_tex, 1);
    if (!(this.debug && this.debug.noshell)) draw(this.shell);
    // 3) усы, бакенбарды, прядь на лбу
    gl.depthMask(false);
    u = use(this.pDecal); gl.uniform1f(u.u_off, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.decalTex); gl.uniform1i(u.u_tex, 1);
    if (!(this.debug && this.debug.nodecal)) draw(this.decal);
    u = use(this.pStrip);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.burnTex); gl.uniform1i(u.u_tex, 1);
    if (!(this.debug && this.debug.nodecal)) draw(this.burns);
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
  const want = delegate || (ios ? "CPU" : "GPU");
  if (useWorker && mode === "VIDEO" && typeof Worker !== "undefined" && typeof OffscreenCanvas !== "undefined" && typeof createImageBitmap === "function") {
    try {
      const w = new Worker(BASE + "track-worker.js" + VQ);
      const ready = await new Promise((res, rej) => {
        const to = setTimeout(() => rej(new Error("worker timeout")), 25000);
        w.onmessage = e => { if (e.data.type === "ready") { clearTimeout(to); res(e.data); } else if (e.data.type === "error") { clearTimeout(to); rej(new Error(e.data.error)); } };
        w.onerror = e => { clearTimeout(to); rej(new Error(e.message || "worker error")); };
        const mcopy = model.slice().buffer;
        w.postMessage({ type: "init", bundle: BASE + "mp/vision_bundle_cjs.js", loader, wasm: wasmUrl, model: mcopy, delegate: want }, [mcopy]);
      });
      w.onerror = null;
      return { kind: "worker", worker: w, delegate: ready.delegate, ios };
    } catch (e) { console.warn("mask: worker failed, using main thread", e); }
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
export async function createMask({ mode = "VIDEO", delegate = null, debug = false, onProgress = null, worker = true, fresh = true } = {}){
  const [hair, decal, burnImg, burns, shell, tris] = await Promise.all([
    loadImage(BASE + "zhorun/hair.webp" + VQ), loadImage(BASE + "zhorun/decal.webp" + VQ), loadImage(BASE + "zhorun/burns.webp" + VQ), fetchBytes(BASE + "zhorun/burns.bin" + VQ),
    fetchBytes(BASE + "zhorun/shell.bin" + VQ), fetchBytes(BASE + "zhorun/decal_tris.bin" + VQ)]);
  const r = new Renderer({ hair, decal, burnImg, burns, shell, tris: new Uint16Array(tris.buffer, tris.byteOffset, tris.byteLength / 2) }, debug);
  const tr = await makeTracker(mode, delegate, onProgress, worker);
  const sm = new Smoother();
  const raw = new Float32Array(478 * 3);
  const lum = document.createElement("canvas"); lum.width = lum.height = 8;
  const lg = lum.getContext("2d", { willReadFrequently: true });
  const small = document.createElement("canvas"), sg = small.getContext("2d");
  const holds = [0, 1].map(() => { const c = document.createElement("canvas"); return { c, g: c.getContext("2d") }; });
  let holdIdx = 0, seq = 0;

  const M = {
    raw, get delegate(){ return tr.delegate; }, kind: tr.kind, stats: r.stats,
    found: false, vis: 0, exposure: 1, pose: null, lm: null, srcW: 0, srcH: 0, miss: 0,
    frame: null, frameW: 0, frameH: 0, latency: 0, detMs: 0, results: 0, lastResult: 0, inflight: null,
    rate: 0, sync: false, vel: null, tCap: 0, _prev: null,
    _ingest(lmNorm, W, H, nowMs, smooth){
      this.srcW = W; this.srcH = H;
      for (let i = 0; i < 478; i++) { raw[i * 3] = lmNorm[i * 3] * W; raw[i * 3 + 1] = lmNorm[i * 3 + 1] * H; raw[i * 3 + 2] = lmNorm[i * 3 + 2] * W; }
      const o = sm.apply(raw, nowMs / 1000, smooth);
      this.lm = o.lm; this.pose = o.pose;
      // скорость головы — чтобы на живом видео «довести» маску до текущего момента
      const pv = this._prev, P = this.pose;
      if (smooth && pv && nowMs - pv.t > 5 && nowMs - pv.t < 300) {
        const dt = (nowMs - pv.t) / 1000;
        let q = P.q; if (q[0] * pv.q[0] + q[1] * pv.q[1] + q[2] * pv.q[2] + q[3] * pv.q[3] < 0) q = q.map(x => -x);
        const nv = { t: [0, 1, 2].map(k => (P.t[k] - pv.tt[k]) / dt), s: (P.s / pv.s - 1) / dt, q: [0, 1, 2, 3].map(k => (q[k] - pv.q[k]) / dt) };
        const v = this.vel;
        this.vel = v ? { t: v.t.map((x, k) => x * 0.4 + nv.t[k] * 0.6), s: v.s * 0.4 + nv.s * 0.6, q: v.q.map((x, k) => x * 0.4 + nv.q[k] * 0.6) } : nv;
      } else this.vel = null;
      this._prev = { t: nowMs, tt: P.t.slice(), s: P.s, q: P.q.slice() };
      this.tCap = nowMs;
      this.found = true; this.miss = 0;
    },
    _lost(){ if (++this.miss > 3) { this.found = false; sm.reset(); this.vel = null; this._prev = null; } },
    // кадр камеры → в поиск лица (в отдельном потоке; результат придёт сам)
    pump(video, now = performance.now()){
      if (!video || video.readyState < 2 || !video.videoWidth) return;
      this._vid = video;
      if (this.inflight) { if (now - this.inflight.t0 > 1500) { this.inflight = null; this.frame = null; } return; }
      if (!this._fresh(video, now)) return;
      const vw = video.videoWidth, vh = video.videoHeight;
      const kk = Math.min(1, 480 / Math.max(vw, vh)), sw = Math.round(vw * kk), sh = Math.round(vh * kk);
      const id = ++seq, ts = Math.round(now);
      // быстрый телефон: держим кадр и показываем его вместе с маской (идеальное совпадение);
      // медленный: показываем живое видео, а маску «доводим» по скорости головы
      let hold = null;
      if (this.sync) {
        hold = holds[holdIdx]; holdIdx ^= 1;
        if (hold.c.width !== vw || hold.c.height !== vh) { hold.c.width = vw; hold.c.height = vh; }
        hold.g.drawImage(video, 0, 0, vw, vh);
      }
      this.inflight = { id, t0: now, hold, vw, vh };
      const src = hold ? hold.c : video;
      if (tr.kind === "worker") {
        const send = bmp => { const d = performance.now() - now; this.bmpMs = this.bmpMs ? this.bmpMs * 0.9 + d * 0.1 : d; tr.worker.postMessage({ type: "frame", id, ts, bmp }, [bmp]); };
        const viaCanvas = () => { if (small.width !== sw || small.height !== sh) { small.width = sw; small.height = sh; } sg.drawImage(src, 0, 0, sw, sh); return createImageBitmap(small); };
        (this._resizeOK !== false ? createImageBitmap(src, { resizeWidth: sw, resizeHeight: sh, resizeQuality: "low" }).catch(() => { this._resizeOK = false; return viaCanvas(); }) : viaCanvas())
          .then(send).catch(() => { this.inflight = null; });
      } else {
        if (small.width !== sw || small.height !== sh) { small.width = sw; small.height = sh; }
        sg.drawImage(src, 0, 0, sw, sh);
        const t0 = performance.now();
        let res = null;
        try { res = tr.fl.detectForVideo(small, ts); } catch (e) { console.warn("mask: detect", e); }
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
      if (skip) return;
      const now = performance.now();
      this.frame = f.hold ? f.hold.c : null; this.frameW = f.vw; this.frameH = f.vh;
      // медианы по последним кадрам: первые (прогрев) и случайные долгие кадры не сбивают решения
      const H = this._hist || (this._hist = { lat: [], det: [], at: [] });
      const push = (a, v) => { a.push(v); if (a.length > 15) a.shift(); };
      push(H.lat, now - f.t0); push(H.det, ms); push(H.at, now);
      const med = a => { const b = a.slice().sort((x, y) => x - y); return b[b.length >> 1]; };
      this.latency = med(H.lat); this.detMs = med(H.det);
      this.rate = H.at.length > 2 ? (H.at.length - 1) * 1000 / Math.max(1, now - H.at[0]) : 0;
      this.results++; this._sinceSwitch = (this._sinceSwitch || 0) + 1;
      this.lastResult = now;
      // держать кадр стоит, только если поиск быстрый; с запасом, чтобы режим не мигал туда-сюда
      if (this.results > 10) this.sync = this.sync ? (this.latency < 58 && this.rate > 18) : (this.latency < 42 && this.rate > 24);
      if (lm) { this._ingest(lm, f.vw, f.vh, f.t0, true); if (this.results % 15 === 1) this._light(f.hold ? f.hold.c : this._vid || null); }
      else this._lost();
      this._tune();
    },
    // на Android пробуем и видеокарту, и процессор: оставляем то, что быстрее на этом телефоне
    _tune(){
      if (tr.kind !== "worker" || this._tuneDone || tr.ios) return;
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
    _fresh(video, now){
      if (!fresh || !video.requestVideoFrameCallback) return true;
      if (this._rv !== video) {
        this._rv = video; this._new = true; this._seen = now;
        const cb = () => { this._new = true; this._seen = performance.now(); if (this._rv === video) video.requestVideoFrameCallback(cb); };
        video.requestVideoFrameCallback(cb);
      }
      if (!this._new && now - this._seen < 250) return false;
      this._new = false;
      return true;
    },
    _light(src){
      if (!src) return;
      try {
        const L = this.lm, xs = [234, 454, 10, 152].map(i => L[i * 3]), ys = [234, 454, 10, 152].map(i => L[i * 3 + 1]);
        const x0 = Math.max(0, Math.min(...xs)), x1 = Math.min(this.srcW, Math.max(...xs)), y0 = Math.max(0, Math.min(...ys)), y1 = Math.min(this.srcH, Math.max(...ys));
        if (x1 - x0 < 8 || y1 - y0 < 8) return;
        lg.drawImage(src, x0, y0, x1 - x0, y1 - y0, 0, 0, 8, 8);
        const d = lg.getImageData(0, 0, 8, 8).data;
        let s = 0;
        for (let i = 0; i < d.length; i += 4) s += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
        const Y = s / 64 / 255;
        this.exposure = this.exposure * 0.5 + clamp(Math.pow(Y / 0.52, 0.7), 0.55, 1.35) * 0.5;
      } catch (e) {}
    },
    trackImage(img){
      const res = tr.fl.detect(img);
      const f = res.faceLandmarks && res.faceLandmarks[0];
      if (!f) { this.found = false; return false; }
      const lm = new Float32Array(f.length * 3);
      for (let i = 0; i < f.length; i++) { lm[i * 3] = f[i].x; lm[i * 3 + 1] = f[i].y; lm[i * 3 + 2] = f[i].z; }
      this._ingest(lm, img.naturalWidth || img.width, img.naturalHeight || img.height, 0, false);
      this.vis = 1;
      return true;
    },
    // поза «на сейчас»: от кадра, где нашли лицо, до живого кадра на экране голова успела сдвинуться
    _predict(now){
      const P = this.pose, v = this.vel;
      if (!P || !v) return null;
      const dt = clamp((now - this.tCap) / 1000, 0, 0.12) * 0.85;
      if (dt < 0.002) return null;
      const t = [P.t[0] + v.t[0] * dt, P.t[1] + v.t[1] * dt, P.t[2] + v.t[2] * dt];
      const s = P.s * (1 + clamp(v.s * dt, -0.1, 0.1));
      const q = P.q.map((x, k) => x + v.q[k] * dt), ql = Math.hypot(...q);
      for (let k = 0; k < 4; k++) q[k] /= ql;
      const R = quatToR(q), L = this.lm, out = this._plm || (this._plm = new Float32Array(478 * 3)), ks = s / P.s;
      for (let i = 0; i < 478; i++) {
        const d = RTmul(P.R, [L[i * 3] - P.t[0], L[i * 3 + 1] - P.t[1], L[i * 3 + 2] - P.t[2]]), e = Rmul(R, d);
        out[i * 3] = e[0] * ks + t[0]; out[i * 3 + 1] = e[1] * ks + t[1]; out[i * 3 + 2] = e[2] * ks + t[2];
      }
      return { pose: { s, R, t, q }, lm: out };
    },
    // view: как кадр лёг на холст (как coverDraw): dx, dy, k — пикселей холста на пиксель кадра; live — поверх живого видео
    draw(g, view, alpha = 1){
      this.vis += ((this.found ? 1 : 0) - this.vis) * 0.4;
      if (this.vis < 0.02 || !this.pose) return;
      const pr = view.live ? this._predict(performance.now()) : null;
      const pose = pr ? pr.pose : this.pose, lm = pr ? pr.lm : this.lm;
      r.upload(lm);
      const box = r.bbox(pose, view, lm);
      if (box[2] < 2 || box[3] < 2) return;
      r.render(pose, view, this.exposure, box);
      g.save();
      g.globalAlpha = Math.min(1, this.vis) * alpha;
      if (view.mirror) { g.translate(view.W, 0); g.scale(-1, 1); }
      g.drawImage(r.canvas, box[0], box[1], box[2], box[3], box[0], box[1], box[2], box[3]);
      g.restore();
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
      if (d.delegate !== tr.delegate) { tr.delegate = d.delegate; M._sinceSwitch = 0; if (M._hist) { M._hist.det.length = 0; M._hist.lat.length = 0; M._hist.at.length = 0; } }
      else if (M._trial && !M._tuneDone) M._tuneDone = true;   // другой режим не завёлся — остаёмся
    }
  };
  return M;
}
