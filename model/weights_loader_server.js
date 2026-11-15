import fs from "node:fs/promises";

/**
 * weights.bin + weights_manifest.json okur (Node.js sunucusu).
 *
 * v13 (1B model): dosya 2 GB'ı aşabilir, fs.readFile tek parçada okuyamaz. Bu yüzden
 * her tensör kendi byte aralığından (dosya tutamacıyla) ayrı ayrı okunur.
 * Manifest'te tensör başına "dtype" olabilir:
 *   - "float32" (varsayılan; eski dosyalar bu şekilde): olduğu gibi okunur
 *   - "float16": okunurken float32'ye açılır (model.js matematiği aynı kalır)
 * NOT: Float32'ye açılınca bellek ~4.4 GB tutar (1B model) - sunucu RAM'ini buna göre seç.
 */

const READ_CHUNK = 256 * 1024 * 1024; // 256 MB'lık parçalarla oku

function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let man = h & 0x3ff;
    let bits;
    if (exp === 0) {
      if (man === 0) {
        bits = sign; // ±0
      } else {
        // altnormal sayı: normalleştir
        exp = 1;
        while ((man & 0x400) === 0) {
          man <<= 1;
          exp--;
        }
        man &= 0x3ff;
        bits = sign | ((exp + 112) << 23) | (man << 13);
      }
    } else if (exp === 31) {
      bits = sign | 0x7f800000 | (man << 13); // inf / NaN
    } else {
      bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    u32[0] = bits;
    out[i] = f32[0];
  }
  return out;
}

export { f16ToF32 };

export async function loadWeights(binPath, manifestPath, { onProgress = null } = {}) {
  const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
  const fh = await fs.open(binPath, "r");
  const weights = {};
  try {
    const { size } = await fh.stat();
    const names = Object.keys(manifest);
    let done = 0;
    for (const name of names) {
      const info = manifest[name];
      const dtype = info.dtype || "float32";
      const elemSize = dtype === "float16" ? 2 : 4;
      if (info.byte_length % elemSize !== 0) {
        throw new Error(`Hizalama hatası (${dtype}): ${name}`);
      }
      if (info.byte_offset + info.byte_length > size) {
        throw new Error(`Manifest weights.bin sınırını aşıyor: ${name}`);
      }
      const ab = new ArrayBuffer(info.byte_length);
      const u8 = new Uint8Array(ab);
      for (let off = 0; off < info.byte_length; off += READ_CHUNK) {
        const len = Math.min(READ_CHUNK, info.byte_length - off);
        const { bytesRead } = await fh.read(u8, off, len, info.byte_offset + off);
        if (bytesRead !== len) throw new Error(`Dosya beklenenden kısa: ${name}`);
      }
      const arr = dtype === "float16" ? f16ToF32(new Uint16Array(ab)) : new Float32Array(ab);
      arr._shape = info.shape;
      weights[name] = arr;
      done++;
      if (onProgress && done % 20 === 0) onProgress(done, names.length);
    }
  } finally {
    await fh.close();
  }
  return weights;
}
