import fs from "node:fs/promises";

export async function loadWeights(binPath, manifestPath) {
  const [bin, manifestRaw] = await Promise.all([
    fs.readFile(binPath),
    fs.readFile(manifestPath, "utf8"),
  ]);

  const manifest = JSON.parse(manifestRaw);
  const weights = {};

  for (const [name, info] of Object.entries(manifest)) {
    if (info.byte_offset % 4 !== 0 || info.byte_length % 4 !== 0) {
      throw new Error(`Float32 hizalama hatası: ${name}`);
    }
    const end = info.byte_offset + info.byte_length;
    if (end > bin.byteLength) {
      throw new Error(`Manifest weights.bin sınırını aşıyor: ${name}`);
    }
    const start = bin.byteOffset + info.byte_offset;
    weights[name] = new Float32Array(bin.buffer, start, info.byte_length / 4);
    weights[name]._shape = info.shape;
  }
  return weights;
}
