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
 * Performans notu: Bu saf JavaScript, GPU kullanmıyor (WebGL/WebGPU yok).
 * Küçük modellerde (birkaç milyon parametre) tarayıcıda saniyeler
 * içinde çalışır; büyük modellerde yavaşlar. Hedefimiz zaten küçük/hafif
 * model olduğu için bu sorun teşkil etmiyor.
 */

// ---------------- Temel matris/vektör yardımcıları ----------------
// Not: Basitlik ve okunabilirlik için düz JS dizileri (Float32Array)
// kullanıyoruz; harici bir matematik kütüphanesi (tensorflow.js vb.) YOK.

/** (m x k) * (k x n) = (m x n) matris çarpımı. Ağırlık matrisleri
 * PyTorch geleneğiyle (out_features, in_features) şeklinde saklanır,
 * bu yüzden w'yi transpoze etmeden "x . w^T" mantığıyla çarpıyoruz. */
function matmulTransposed(x, m, k, w, n) {
  // x: (m,k) düz dizi, w: (n,k) düz dizi (her satır bir çıkış nöronu)
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

/** x: Float32Array, mantıksal şekli (T, headDim) - tek bir head için.
 * offset: bu parçanın dizideki başlangıç pozisyonu (KV-cache offset'i). */
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
  /**
   * @param {object} config { vocab_size, block_size, n_layer, n_head, n_embd }
   * @param {object} weights isim -> Float32Array (export scriptinden gelen manifest'e göre)
   */
  constructor(config, weights) {
    this.cfg = config;
    this.w = weights; // örn. this.w["blocks.0.attn.qkv.weight"]
    this.headDim = config.n_embd / config.n_head;
    this.rope = precomputeRope(this.headDim, config.block_size);
  }

  /** Embedding tablosundan tek bir token'ın vektörünü okur. */
  _embed(tokenId) {
    const emb = this.w["tok_emb.weight"]; // (vocab_size, n_embd) düz dizi
    const off = tokenId * this.cfg.n_embd;
    return emb.subarray(off, off + this.cfg.n_embd);
  }

  /**
   * Tek bir Block'u T token için çalıştırır.
   * x: Float32Array (T * n_embd)
   * cache: { k: Float32Array|null, v: Float32Array|null, len: number } - katmana özel
   * returns: yeni x (T * n_embd)
   */
  _blockForward(layerIdx, x, T, cache) {
    const { n_embd, n_head } = this.cfg;
    const headDim = this.headDim;
    const prefix = `blocks.${layerIdx}.`;

    // --- ln1 + attention ---
    const xNorm = rmsNorm(x, T, n_embd, this.w[prefix + "ln1.weight"]);
    const qkvW = this.w[prefix + "attn.qkv.weight"]; // (3*n_embd, n_embd)
    const qkv = matmulTransposed(xNorm, T, n_embd, qkvW, 3 * n_embd); // (T, 3*n_embd)

    const pastLen = cache.len;
    const totalLen = pastLen + T;

    // Head'lere ayır, RoPE uygula, cache'e ekle
    const newK = new Float32Array(totalLen * n_head * headDim);
    const newV = new Float32Array(totalLen * n_head * headDim);
    if (cache.k) newK.set(cache.k.subarray(0, pastLen * n_head * headDim), 0);
    if (cache.v) newV.set(cache.v.subarray(0, pastLen * n_head * headDim), 0);

    const attnOut = new Float32Array(T * n_embd);

    for (let h = 0; h < n_head; h++) {
      // Bu head için q/k/v'yi çıkar: qkv düzeni (T, 3, n_head, headDim)
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

      // Cache'e yeni k/v'yi yaz
      for (let t = 0; t < T; t++) {
        const pos = pastLen + t;
        for (let d = 0; d < headDim; d++) {
          newK[(pos * n_head + h) * headDim + d] = kRot[t * headDim + d];
          newV[(pos * n_head + h) * headDim + d] = vh[t * headDim + d];
        }
      }

      // Attention skorları: (T, totalLen)
      const scale = 1 / Math.sqrt(headDim);
      const scores = new Float32Array(T * totalLen);
      for (let ti = 0; ti < T; ti++) {
        for (let tj = 0; tj < totalLen; tj++) {
          // Sadece prefill'de (pastLen===0 ve T>1) gelecek pozisyonlar maskelenir
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

      // Attention çıktısı: scores @ v
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

    const projW = this.w[prefix + "attn.proj.weight"]; // (n_embd, n_embd)
    const projOut = matmulTransposed(attnOut, T, n_embd, projW, n_embd);
    for (let i = 0; i < T * n_embd; i++) x[i] += projOut[i]; // residual

    // --- ln2 + MLP (SwiGLU) ---
    const xNorm2 = rmsNorm(x, T, n_embd, this.w[prefix + "ln2.weight"]);
    const w1 = this.w[prefix + "mlp.w1.weight"];
    const w2 = this.w[prefix + "mlp.w2.weight"];
    const w3 = this.w[prefix + "mlp.w3.weight"];
    const hidden = w1.length / n_embd; // w1 şekli (hidden, n_embd)

    const a = matmulTransposed(xNorm2, T, n_embd, w1, hidden);
    const b = matmulTransposed(xNorm2, T, n_embd, w2, hidden);
    const gated = new Float32Array(T * hidden);
    for (let i = 0; i < T * hidden; i++) gated[i] = silu(a[i]) * b[i];
    const mlpOut = matmulTransposed(gated, T, hidden, w3, n_embd);

    for (let i = 0; i < T * n_embd; i++) x[i] += mlpOut[i]; // residual

    return x;
  }

  /**
   * ids dizisini işleyip son pozisyon için logits (vocab_size uzunluğunda) döndürür.
   * caches: her katman için { k, v, len } - üretim boyunca dışarıda tutulur.
   */
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

    // Weight tying: çıkış = xNorm @ tok_emb.weight^T
    const embW = this.w["tok_emb.weight"]; // (vocab_size, n_embd)
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

  /** Basit top-k + sıcaklık örneklemesi. */
  _sample(logits, temperature, topK) {
    const scaled = Array.from(logits, (v) => v / Math.max(temperature, 1e-6));
    const indexed = scaled.map((v, i) => [i, v]);
    indexed.sort((a, b) => b[1] - a[1]);
    const top = indexed.slice(0, Math.max(1, Math.min(topK, indexed.length)));

    const maxLogit = top[0][1];
    const exps = top.map(([, v]) => Math.exp(v - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / sum);

    let r = Math.random();
    for (let i = 0; i < probs.length; i++) {
      if (r < probs[i]) return top[i][0];
      r -= probs[i];
    }
    return top[top.length - 1][0];
  }

  /**
   * KV-cache'li üretim. promptIds: encode edilmiş başlangıç dizisi.
   * onToken: (isteğe bağlı) her yeni token üretildiğinde çağrılır - UI'da
   * kelime kelime akan bir yazı efekti için kullanışlı.
   */
  generate(promptIds, maxNewTokens, { temperature = 0.8, topK = 40, onToken = null } = {}) {
    const caches = Array.from({ length: this.cfg.n_layer }, () => ({ k: null, v: null, len: 0 }));
    const allIds = [...promptIds];

    const blockSize = this.cfg.block_size;
    let promptWindow = promptIds.length > blockSize ? promptIds.slice(-blockSize) : promptIds;

    let logits = this.forward(promptWindow, caches);
    let nextId = this._sample(logits, temperature, topK);
    allIds.push(nextId);
    if (onToken) onToken(nextId);

    for (let i = 0; i < maxNewTokens - 1; i++) {
      const totalLen = caches[0].len + 1;
      if (totalLen > blockSize) {
        // Pencereleme: cache'i sıfırla, son block_size token ile yeniden prefill
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
      nextId = this._sample(logits, temperature, topK);
      allIds.push(nextId);
      if (onToken) onToken(nextId);
    }

    return allIds.slice(promptIds.length);
  }
}
