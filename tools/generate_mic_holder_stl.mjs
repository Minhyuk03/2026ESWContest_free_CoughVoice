import fs from 'node:fs';
import path from 'node:path';

const triangles = [];

function tri(a, b, c) {
  triangles.push([a, b, c]);
}

function box(x0, x1, y0, y1, z0, z1) {
  const p = [
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
  ];
  const faces = [
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
    [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
  ];
  for (const [a, b, c] of faces) tri(p[a], p[b], p[c]);
}

function openRing({cx, cz, y0, y1, innerR, outerR, startDeg, endDeg, steps}) {
  const rings = [];
  for (let i = 0; i <= steps; i += 1) {
    const a = (startDeg + (endDeg - startDeg) * i / steps) * Math.PI / 180;
    rings.push({
      ifront: [cx + innerR * Math.cos(a), y0, cz + innerR * Math.sin(a)],
      iback: [cx + innerR * Math.cos(a), y1, cz + innerR * Math.sin(a)],
      ofront: [cx + outerR * Math.cos(a), y0, cz + outerR * Math.sin(a)],
      oback: [cx + outerR * Math.cos(a), y1, cz + outerR * Math.sin(a)],
    });
  }
  for (let i = 0; i < steps; i += 1) {
    const a = rings[i];
    const b = rings[i + 1];
    tri(a.ofront, b.ofront, b.oback); tri(a.ofront, b.oback, a.oback);
    tri(a.ifront, b.iback, b.ifront); tri(a.ifront, a.iback, b.iback);
    tri(a.ifront, b.ifront, b.ofront); tri(a.ifront, b.ofront, a.ofront);
    tri(a.iback, b.oback, b.iback); tri(a.iback, a.oback, b.oback);
  }
  const s = rings[0];
  const e = rings[rings.length - 1];
  tri(s.ifront, s.ofront, s.oback); tri(s.ifront, s.oback, s.iback);
  tri(e.ifront, e.oback, e.ofront); tri(e.ifront, e.iback, e.oback);
}

function normal(a, b, c) {
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const n = [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
  ];
  const len = Math.hypot(...n) || 1;
  return n.map((value) => value / len);
}

// Raspberry Pi 5 official PCB footprint, with a practical 3 mm print thickness.
box(0, 85, 0, 56, 0, 3);

// Center support: 12 mm wide, 12 mm deep. It overlaps the cradle for slicer union.
box(36.5, 48.5, 22, 34, 3, 41.5);

// Open circular cradle: Ø20 mm inner opening, 3 mm wall, 12 mm axial depth.
// The center is exactly 50 mm above the top face of the base (z = 53 mm).
openRing({
  cx: 42.5,
  cz: 53,
  y0: 22,
  y1: 34,
  innerR: 10,
  outerR: 13,
  startDeg: 140,
  endDeg: 400,
  steps: 52,
});

let stl = 'solid rpi5_mic_holder\n';
for (const [a, b, c] of triangles) {
  const n = normal(a, b, c);
  stl += `  facet normal ${n.join(' ')}\n    outer loop\n`;
  stl += `      vertex ${a.join(' ')}\n      vertex ${b.join(' ')}\n      vertex ${c.join(' ')}\n`;
  stl += '    endloop\n  endfacet\n';
}
stl += 'endsolid rpi5_mic_holder\n';

const output = path.resolve('RaspberryPi5_MicHolder_85x56_H50_D20.stl');
fs.writeFileSync(output, stl);
console.log(output);
