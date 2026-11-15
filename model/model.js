/**
 * Sıfırdan Decoder-only Transformer - JavaScript portu.
 *
 * ÖNEMLİ: Bu dosya SADECE TAHMİN (inference) yapar, EĞİTİM YAPMAZ.
 * Eğitim (backpropagation, gradyan hesabı) hâlâ Colab'da Python/PyTorch
 * ile yapılıyor. Burada olan şey: PC'de eğitilip dışa aktarılmış hazır
 * sayıları (ağırlıkları) okuyup "bir sonraki kelime ne olur" hesaplamak.
 *
 * Matematik olarak eğitim defterindeki PyTorch modeliyle BİREBİR AYNI olmak
 * zorunda (RoPE, RMSNorm, SwiGLU, causal attention, v13: Grouped Query Attention) - yoksa PC'de öğrenilen sayılar
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
  // out[i][j] = sum_p x[i][p] * w[j][p]   (w: n x k, satır-öncelikli)
  // v13: iç döngü 4'lü açıldı (aynı matematik, daha az döngü yükü).
  const out = new Float32Array(m * n);
  const k4 = k - (k % 4);
  for (let i = 0; i < m; i++) {
    const xOff = i * k;
    for (let j = 0; j < n; j++) {
      const wOff = j * k;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      let p = 0;
      for (; p < k4; p += 4) {
        s0 += x[xOff + p] * w[wOff + p];
        s1 += x[xOff + p + 1] * w[wOff + p + 1];
        s2 += x[xOff + p + 2] * w[wOff + p + 2];
        s3 += x[xOff + p + 3] * w[wOff + p + 3];
      }
      let sum = s0 + s1 + s2 + s3;
      for (; p < k; p++) sum += x[xOff + p] * w[wOff + p];
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
    this.nKv = config.n_kv_head || config.n_head;           // GQA: key/value head sayısı
    if (config.n_head % this.nKv !== 0) throw new Error("n_head, n_kv_head'e tam bölünmeli");
    this.nRep = config.n_head / this.nKv;                    // her KV head'i paylaşan query head sayısı
    this.rope = precomputeRope(this.headDim, config.block_size);
  }

  _embed(tokenId) {
    const emb = this.w["tok_emb.weight"];
    const off = tokenId * this.cfg.n_embd;
    return emb.subarray(off, off + this.cfg.n_embd);
  }

  _qkvWeights(prefix) {
    const { n_embd } = this.cfg;
    const qkv = this.w[prefix + "attn.qkv.weight"];
    if (qkv) {
      // ESKİ biçim (v12, tek qkv matrisi, GQA yok): satır aralıklarını ayır
      const c = n_embd * n_embd;
      return [qkv.subarray(0, c), qkv.subarray(c, 2 * c), qkv.subarray(2 * c, 3 * c)];
    }
    return [this.w[prefix + "attn.q.weight"], this.w[prefix + "attn.k.weight"], this.w[prefix + "attn.v.weight"]];
  }

  _blockForward(layerIdx, x, T, cache) {
    const { n_embd, n_head } = this.cfg;
    const nKv = this.nKv;
    const nRep = this.nRep;
    const headDim = this.headDim;
    const qDim = n_head * headDim;
    const kvDim = nKv * headDim;
    const prefix = `blocks.${layerIdx}.`;

    const xNorm = rmsNorm(x, T, n_embd, this.w[prefix + "ln1.weight"]);
    const [qW, kW, vW] = this._qkvWeights(prefix);
    const q = matmulTransposed(xNorm, T, n_embd, qW, qDim);
    const k = matmulTransposed(xNorm, T, n_embd, kW, kvDim);
    const v = matmulTransposed(xNorm, T, n_embd, vW, kvDim);

    const pastLen = cache.len;
    const totalLen = pastLen + T;

    // KV-cache: sadece nKv head saklanır (GQA'nın bellek kazancı). Kapasite 128'lik
    // adımlarla büyür (her token'da yeniden ayırıp kopyalamamak için).
    const stride = nKv * headDim;
    const needed = totalLen * stride;
    let newK = cache.k;
    let newV = cache.v;
    if (!newK || newK.length < needed) {
      const capTokens = Math.min(this.cfg.block_size, Math.ceil(totalLen / 128) * 128);
      const cap = Math.max(needed, capTokens * stride);
      const nk = new Float32Array(cap);
      const nv = new Float32Array(cap);
      if (cache.k) {
        nk.set(cache.k.subarray(0, pastLen * stride), 0);
        nv.set(cache.v.subarray(0, pastLen * stride), 0);
      }
      newK = nk;
      newV = nv;
    }

    // 1) K ve V: RoPE (sadece K) uygula, cache'e yaz - KV head başına BİR kez
    for (let g = 0; g < nKv; g++) {
      const kh = new Float32Array(T * headDim);
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < headDim; d++) {
          kh[t * headDim + d] = k[t * kvDim + g * headDim + d];
          newV[((pastLen + t) * nKv + g) * headDim + d] = v[t * kvDim + g * headDim + d];
        }
      }
      const kRot = applyRope(kh, T, headDim, this.rope, pastLen);
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < headDim; d++) {
          newK[((pastLen + t) * nKv + g) * headDim + d] = kRot[t * headDim + d];
        }
      }
    }

    // 2) Her query head: RoPE(q), KV head = floor(h / nRep) ile dikkat
    const attnOut = new Float32Array(T * qDim);
    const scale = 1 / Math.sqrt(headDim);
    for (let h = 0; h < n_head; h++) {
      const g = Math.floor(h / nRep);
      const qh = new Float32Array(T * headDim);
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < headDim; d++) qh[t * headDim + d] = q[t * qDim + h * headDim + d];
      }
      const qRot = applyRope(qh, T, headDim, this.rope, pastLen);

      const scores = new Float32Array(T * totalLen);
      for (let ti = 0; ti < T; ti++) {
        const lastVisible = pastLen + ti; // causal: bu konumdan sonrası görünmez
        for (let tj = 0; tj < totalLen; tj++) {
          if (tj > lastVisible) {
            scores[ti * totalLen + tj] = -Infinity;
            continue;
          }
          let sum = 0;
          const kOff = (tj * nKv + g) * headDim;
          for (let d = 0; d < headDim; d++) sum += qRot[ti * headDim + d] * newK[kOff + d];
          scores[ti * totalLen + tj] = sum * scale;
        }
        softmaxInPlace(scores, ti * totalLen, totalLen);
      }

      for (let ti = 0; ti < T; ti++) {
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let tj = 0; tj < totalLen; tj++) {
            const p = scores[ti * totalLen + tj];
            if (p !== 0) sum += p * newV[(tj * nKv + g) * headDim + d];
          }
          attnOut[ti * qDim + h * headDim + d] = sum;
        }
      }
    }

    cache.k = newK;
    cache.v = newV;
    cache.len = totalLen;

    const projW = this.w[prefix + "attn.proj.weight"];
    const projOut = matmulTransposed(attnOut, T, qDim, projW, n_embd);
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
    temperature = 0.75,
    topK = 50,
    topP = 0.9,
    onToken = null,
    repetitionPenalty = 1.15,
    stopIds = null, // Set<number>: bunlardan biri üretilince durur (token çıktıya EKLENMEZ)
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
    if (stopIds && stopIds.has(nextId)) return allIds.slice(promptIds.length);
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
      if (stopIds && stopIds.has(nextId)) break;
      allIds.push(nextId);
      pushRecent(nextId);
      if (onToken && onToken(nextId)) break;
    }

    return allIds.slice(promptIds.length);
  }
}
