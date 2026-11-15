/**
 * Sıfırdan Decoder-only Transformer - JavaScript portu.
 *
 * ÖNEMLİ: Bu dosya SADECE TAHMİN (inference) yapar, EĞİTİM YAPMAZ.
 * Eğitim (backpropagation, gradyan hesabı) hâlâ Colab'da Python/PyTorch
 * ile yapılıyor. Burada olan şey: PC'de eğitilip dışa aktarılmış hazır
 * sayıları (ağırlıkları) okuyup "bir sonraki kelime ne olur" hesaplamak.
 *
 * Matematik olarak brain/model.py ile BİREBİR AYNI olmak zorunda (RoPE,
 * RMSNorm, SwiGLU, causal attention) - yoksa PC'de öğrenilen sayılar
 * burada yanlış yorumlanır ve model saçma metin üretir.
 *
 * DÜRÜSTLÜK NOTU: Bu dosyadaki generate()/_sample() HİÇBİR ZAMAN kural
 * tabanlı (if-else / kelime eşleştirme) bir cevap üretme mekanizması
 * içermedi - üretim HER ZAMAN modelin kendi öğrendiği ağırlıklarıyla
 * (forward pass + olasılıksal örnekleme) yapılıyor. index.html'deki
 * "sohbet mi soru mu" ayrımı ve kriz-güvenlik kontrolü bu dosyanın
 * DIŞINDA, modelin ne üreteceğine karışmayan ayrı bir orkestrasyon
 * katmanı - bu dosya değişmedi.
 */

// ---------------- Temel matris/vektör yardımcıları ----------------
function matmulTransposed(x, m, k, w, n) {
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      const xOff = i * k;
      const wOff = j * k;
      for (let p = 0; p < k; p++) {
        sum += x[xOff + p] * w[wOff + p];
      }
      out[i * n + j] = sum;
    }
  }
  return out;
}

function rmsNorm(x, m, dim, weight, eps = 1e-6) {
  const out = new Float32Array(m * dim);
  for (let i = 0; i < m; i++) {
    let sumSq = 0;
    const off = i * dim;
    for (let j = 0; j < dim; j++) sumSq += x[off + j] * x[off + j];
    const denom = Math.sqrt(sumSq / dim + eps);
    for (let j = 0; j < dim; j++) out[off + j] = (x[off + j] / denom) * weight[j];
  }
  return out;
}

function silu(v) {
  return v / (1 + Math.exp(-v));
}

function softmaxInPlace(arr, offset, len) {
  let max = -Infinity;
  for (let i = 0; i < len; i++) max = Math.max(max, arr[offset + i]);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const e = Math.exp(arr[offset + i] - max);
    arr[offset + i] = e;
    sum += e;
  }
  for (let i = 0; i < len; i++) arr[offset + i] /= sum;
}

// ---------------- RoPE ----------------
function precomputeRope(headDim, maxSeqLen) {
  const half = headDim / 2;
  const base = 10000;
  const invFreq = new Float32Array(half);
  for (let i = 0; i < half; i++) invFreq[i] = 1 / Math.pow(base, (2 * i) / headDim);

  const cos = new Float32Array(maxSeqLen * half);
  const sin = new Float32Array(maxSeqLen * half);
  for (let t = 0; t < maxSeqLen; t++) {
    for (let i = 0; i < half; i++) {
      const angle = t * invFreq[i];
      cos[t * half + i] = Math.cos(angle);
      sin[t * half + i] = Math.sin(angle);
    }
  }
  return { cos, sin, half };
}

function applyRope(x, T, headDim, rope, offset) {
  const { cos, sin, half } = rope;
  const out = new Float32Array(T * headDim);
  for (let t = 0; t < T; t++) {
    const pos = offset + t;
    for (let i = 0; i < half; i++) {
      const x1 = x[t * headDim + 2 * i];
      const x2 = x[t * headDim + 2 * i + 1];
      const c = cos[pos * half + i];
      const s = sin[pos * half + i];
      out[t * headDim + 2 * i] = x1 * c - x2 * s;
      out[t * headDim + 2 * i + 1] = x1 * s + x2 * c;
    }
  }
  return out;
}

// ---------------- Model ----------------
export class GptModel {
  constructor(config, weights) {
    this.cfg = config;
    this.w = weights;
    this.headDim = config.n_embd / config.n_head;
    this.rope = precomputeRope(this.headDim, config.block_size);
  }

  _embed(tokenId) {
    const emb = this.w["tok_emb.weight"];
    const off = tokenId * this.cfg.n_embd;
    return emb.subarray(off, off + this.cfg.n_embd);
  }

  _blockForward(layerIdx, x, T, cache) {
    const { n_embd, n_head } = this.cfg;
    const headDim = this.headDim;
    const prefix = `blocks.${layerIdx}.`;

    const xNorm = rmsNorm(x, T, n_embd, this.w[prefix + "ln1.weight"]);
    const qkvW = this.w[prefix + "attn.qkv.weight"];
    const qkv = matmulTransposed(xNorm, T, n_embd, qkvW, 3 * n_embd);

    const pastLen = cache.len;
    const totalLen = pastLen + T;

    // Head'lere ayır, RoPE uygula, cache'e ekle
    // HIZ OPTİMİZASYONU: cache boyutu değişmediğinde (ya da mevcut buffer
    // zaten yeterince büyükse) yeni bellek ayırmak yerine mevcut buffer'ı
    // tekrar kullanıyoruz - token üretirken her adımda yeni Float32Array
    // oluşturmak gereksiz çöp toplama (GC) yüküne yol açıyordu. Hesaplanan
    // DEĞERLER birebir aynı, sadece bellek ayırma/kopyalama azaldı.
    const cacheSize = totalLen * n_head * headDim;
    let newK, newV;
    if (cache.k && cache.v && cache.k.length >= cacheSize && cache.v.length >= cacheSize) {
      newK = cache.k;
      newV = cache.v;
    } else {
      newK = new Float32Array(cacheSize);
      newV = new Float32Array(cacheSize);
      if (cache.k) newK.set(cache.k.subarray(0, pastLen * n_head * headDim), 0);
      if (cache.v) newV.set(cache.v.subarray(0, pastLen * n_head * headDim), 0);
    }

    const attnOut = new Float32Array(T * n_embd);

    for (let h = 0; h < n_head; h++) {
      const qh = new Float32Array(T * headDim);
      const kh = new Float32Array(T * headDim);
      const vh = new Float32Array(T * headDim);
      for (let t = 0; t < T; t++) {
        const base = t * 3 * n_embd;
        for (let d = 0; d < headDim; d++) {
          qh[t * headDim + d] = qkv[base + 0 * n_embd + h * headDim + d];
          kh[t * headDim + d] = qkv[base + 1 * n_embd + h * headDim + d];
          vh[t * headDim + d] = qkv[base + 2 * n_embd + h * headDim + d];
        }
      }

      const qRot = applyRope(qh, T, headDim, this.rope, pastLen);
      const kRot = applyRope(kh, T, headDim, this.rope, pastLen);

      for (let t = 0; t < T; t++) {
        const pos = pastLen + t;
        for (let d = 0; d < headDim; d++) {
          newK[(pos * n_head + h) * headDim + d] = kRot[t * headDim + d];
          newV[(pos * n_head + h) * headDim + d] = vh[t * headDim + d];
        }
      }

      const scale = 1 / Math.sqrt(headDim);
      const scores = new Float32Array(T * totalLen);
      for (let ti = 0; ti < T; ti++) {
        for (let tj = 0; tj < totalLen; tj++) {
          if (pastLen === 0 && T > 1 && tj > ti) {
            scores[ti * totalLen + tj] = -Infinity;
            continue;
          }
          let sum = 0;
          for (let d = 0; d < headDim; d++) {
            sum += qRot[ti * headDim + d] * newK[(tj * n_head + h) * headDim + d];
          }
          scores[ti * totalLen + tj] = sum * scale;
        }
        softmaxInPlace(scores, ti * totalLen, totalLen);
      }

      for (let ti = 0; ti < T; ti++) {
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let tj = 0; tj < totalLen; tj++) {
            sum += scores[ti * totalLen + tj] * newV[(tj * n_head + h) * headDim + d];
          }
          attnOut[ti * n_embd + h * headDim + d] = sum;
        }
      }
    }

    cache.k = newK;
    cache.v = newV;
    cache.len = totalLen;

    const projW = this.w[prefix + "attn.proj.weight"];
    const projOut = matmulTransposed(attnOut, T, n_embd, projW, n_embd);
    for (let i = 0; i < T * n_embd; i++) x[i] += projOut[i];

    const xNorm2 = rmsNorm(x, T, n_embd, this.w[prefix + "ln2.weight"]);
    const w1 = this.w[prefix + "mlp.w1.weight"];
    const w2 = this.w[prefix + "mlp.w2.weight"];
    const w3 = this.w[prefix + "mlp.w3.weight"];
    const hidden = w1.length / n_embd;

    const a = matmulTransposed(xNorm2, T, n_embd, w1, hidden);
    const b = matmulTransposed(xNorm2, T, n_embd, w2, hidden);
    const gated = new Float32Array(T * hidden);
    for (let i = 0; i < T * hidden; i++) gated[i] = silu(a[i]) * b[i];
    const mlpOut = matmulTransposed(gated, T, hidden, w3, n_embd);

    for (let i = 0; i < T * n_embd; i++) x[i] += mlpOut[i];

    return x;
  }

  forward(ids, caches) {
    const { n_embd, vocab_size, n_layer } = this.cfg;
    const T = ids.length;

    let x = new Float32Array(T * n_embd);
    for (let t = 0; t < T; t++) {
      x.set(this._embed(ids[t]), t * n_embd);
    }

    for (let l = 0; l < n_layer; l++) {
      x = this._blockForward(l, x, T, caches[l]);
    }

    const xNorm = rmsNorm(x, T, n_embd, this.w["ln_f.weight"]);

    const embW = this.w["tok_emb.weight"];
    const lastRow = xNorm.subarray((T - 1) * n_embd, T * n_embd);
    const logits = new Float32Array(vocab_size);
    for (let v = 0; v < vocab_size; v++) {
      let sum = 0;
      const off = v * n_embd;
      for (let d = 0; d < n_embd; d++) sum += lastRow[d] * embW[off + d];
      logits[v] = sum;
    }
    return logits;
  }

  /**
   * Top-k + top-p (nucleus) + sıcaklık örneklemesi - SAF, ağırlık tabanlı
   * olasılıksal örnekleme. Hiçbir aşamada "şu kelime geldiyse şunu seç"
   * türünden bir kural YOK - tamamen modelin logit çıktısına dayalı.
   *
   * `recentIds` verilirse (Set), o token'ların logit'i `repetitionPenalty`
   * ile cezalandırılır - küçük modellerin sık düştüğü "aynı kelimeyi/
   * kalıbı sonsuz tekrar etme" sorununu azaltmak için (bu da bir "kural"
   * değil, modelin kendi olasılık dağılımını yumuşatan standart bir
   * örnekleme tekniği - LLM'lerin neredeyse hepsinde kullanılır).
   *
   * `topP` (nucleus sampling, varsayılan 0.9): top-k ile sınırlanmış
   * adaylar arasından, kümülatif olasılığı topP'yi geçene kadar en
   * olası olanları alır, geri kalanını eler - bu, topK'nın "sabit sayı"
   * sınırını olasılığa duyarlı hale getirir (bazen 5 aday yeterlidir,
   * bazen 30 gerekir - topP buna dinamik karar verir).
   */
  _sample(logits, temperature, topK, recentIds = null, repetitionPenalty = 1.0, topP = 1.0) {
    const scaled = Array.from(logits, (v, i) => {
      let val = v / Math.max(temperature, 1e-6);
      if (recentIds && repetitionPenalty !== 1.0 && recentIds.has(i)) {
        val = val > 0 ? val / repetitionPenalty : val * repetitionPenalty;
      }
      return val;
    });
    const indexed = scaled.map((v, i) => [i, v]);
    indexed.sort((a, b) => b[1] - a[1]);
    let top = indexed.slice(0, Math.max(1, Math.min(topK, indexed.length)));

    const maxLogit = top[0][1];
    const exps = top.map(([, v]) => Math.exp(v - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    let probs = exps.map((e) => e / sum);

    // top-p (nucleus): kümülatif olasılık topP'yi geçince kes
    if (topP < 1.0) {
      let cum = 0;
      let cutoff = probs.length;
      for (let i = 0; i < probs.length; i++) {
        cum += probs[i];
        if (cum >= topP) {
          cutoff = i + 1;
          break;
        }
      }
      top = top.slice(0, cutoff);
      probs = probs.slice(0, cutoff);
      const renorm = probs.reduce((a, b) => a + b, 0);
      probs = probs.map((p) => p / renorm);
    }

    let r = Math.random();
    for (let i = 0; i < probs.length; i++) {
      if (r < probs[i]) return top[i][0];
      r -= probs[i];
    }
    return top[top.length - 1][0];
  }

  generate(promptIds, maxNewTokens, {
    temperature = 0.45,
    topK = 40,
    topP = 0.9,
    onToken = null,
    repetitionPenalty = 1.3,
  } = {}) {
    const caches = Array.from({ length: this.cfg.n_layer }, () => ({ k: null, v: null, len: 0 }));
    const allIds = [...promptIds];
    const RECENT_WINDOW = 64;
    const recentIds = new Set();
    const pushRecent = (id) => {
      recentIds.add(id);
      if (recentIds.size > RECENT_WINDOW) recentIds.delete(recentIds.values().next().value);
    };

    const blockSize = this.cfg.block_size;
    let promptWindow = promptIds.length > blockSize ? promptIds.slice(-blockSize) : promptIds;

    let logits = this.forward(promptWindow, caches);
    let nextId = this._sample(logits, temperature, topK, recentIds, repetitionPenalty, topP);
    allIds.push(nextId);
    pushRecent(nextId);
    if (onToken && onToken(nextId)) return allIds.slice(promptIds.length);

    for (let i = 0; i < maxNewTokens - 1; i++) {
      const totalLen = caches[0].len + 1;
      if (totalLen > blockSize) {
        for (const c of caches) {
          c.k = null;
          c.v = null;
          c.len = 0;
        }
        const window = allIds.slice(-blockSize);
        logits = this.forward(window, caches);
      } else {
        logits = this.forward([nextId], caches);
      }
      nextId = this._sample(logits, temperature, topK, recentIds, repetitionPenalty, topP);
      allIds.push(nextId);
      pushRecent(nextId);
      if (onToken && onToken(nextId)) break;
    }

    return allIds.slice(promptIds.length);
  }
}
