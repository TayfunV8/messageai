/**
 * export_to_web_format.py'nin ürettiği weights.bin + weights_manifest.json
 * dosyalarını tarayıcıda okuyup model.js'in beklediği
 * { "tensor.adi": Float32Array, ... } formatına çevirir.
 */

export async function loadWeights(binUrl, manifestUrl) {
  const [binResp, manifestResp] = await Promise.all([fetch(binUrl), fetch(manifestUrl)]);

  if (!binResp.ok) throw new Error(`Ağırlık dosyası indirilemedi: ${binUrl} (HTTP ${binResp.status})`);
  if (!manifestResp.ok) throw new Error(`Manifest indirilemedi: ${manifestUrl} (HTTP ${manifestResp.status})`);

  const buffer = await binResp.arrayBuffer();
  const manifest = await manifestResp.json();

  const weights = {};
  for (const [name, info] of Object.entries(manifest)) {
    // byte_offset her zaman 4'ün katı olmalı (float32 hizalaması) -
    // export scripti tensörleri sırayla art arda yazdığı için garanti.
    const floatOffset = info.byte_offset / 4;
    const floatLength = info.byte_length / 4;
    weights[name] = new Float32Array(buffer, info.byte_offset, floatLength);
    weights[name]._shape = info.shape; // hata ayıklama için saklıyoruz
  }

  return weights;
}
