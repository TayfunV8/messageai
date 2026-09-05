/**
 * Sıfırdan BPE Tokenizer - JavaScript portu.
 *
 * KRİTİK: brain/bpe_tokenizer.py ile birebir aynı encode/decode mantığını
 * izler. Model PC'de bu tokenizer'ın ürettiği ID'lerle eğitildiği için,
 * burada farklı bir sıra/algoritma kullanılırsa üretilen metin anlamsız
 * çıkar. Bu yüzden JS tarafında YENİDEN EĞİTİM yapılmıyor - PC'de
 * (Colab'da) öğrenilmiş merges.json dosyası olduğu gibi yükleniyor.
 */

export class BpeTokenizer {
  constructor() {
    /** @type {Map<string, number>} "id1,id2" -> yeni_id (rank sırasına göre) */
    this.merges = new Map();
    /** @type {Map<number, number>} rank hızlı erişim: "id1,id2" -> rank */
    this.mergeRank = new Map();
    /** @type {Map<number, Uint8Array>} yeni_id -> byte dizisi (decode için) */
    this.vocab = new Map();
    for (let i = 0; i < 256; i++) {
      this.vocab.set(i, new Uint8Array([i]));
    }
  }

  /**
   * PC'de tokenizer.save() ile üretilen `_merges.json` dosyasını yükler.
   * Format: {"a,b": yeni_id, ...}
   */
  static async loadFromUrl(url) {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Tokenizer indirilemedi: ${url} (HTTP ${resp.status})`);
    }
    const raw = await resp.json();
    return BpeTokenizer.fromJson(raw);
  }

  static fromJson(raw) {
    const tok = new BpeTokenizer();
    // Python tarafında merge sırası new_id ile birebir örtüşür (256'dan
    // başlayıp artan sırada öğrenilmiş). Bunu koruyarak rank ataması yapıyoruz.
    const entries = Object.entries(raw).map(([key, newId]) => {
      const [a, b] = key.split(",").map(Number);
      return { a, b, newId };
    });
    entries.sort((x, y) => x.newId - y.newId);

    entries.forEach(({ a, b, newId }, rank) => {
      const combined = new Uint8Array(tok.vocab.get(a).length + tok.vocab.get(b).length);
      combined.set(tok.vocab.get(a), 0);
      combined.set(tok.vocab.get(b), tok.vocab.get(a).length);
      tok.vocab.set(newId, combined);
      tok.merges.set(`${a},${b}`, newId);
      tok.mergeRank.set(`${a},${b}`, rank);
    });

    return tok;
  }

  get vocabSize() {
    return this.vocab.size;
  }

  /**
   * Python'daki encode() ile birebir aynı algoritma: her adımda ardışık
   * ikililer arasından en düşük rank'li (en erken öğrenilmiş) merge'i
   * uygula, uygulanabilir ikili kalmayana kadar tekrarla.
   */
  encode(text) {
    let ids = Array.from(new TextEncoder().encode(text));

    while (ids.length >= 2) {
      let bestPair = null;
      let bestRank = Infinity;

      for (let i = 0; i < ids.length - 1; i++) {
        const key = `${ids[i]},${ids[i + 1]}`;
        if (this.mergeRank.has(key)) {
          const rank = this.mergeRank.get(key);
          if (rank < bestRank) {
            bestRank = rank;
            bestPair = [ids[i], ids[i + 1]];
          }
        }
      }

      if (bestPair === null) break;

      const newId = this.merges.get(`${bestPair[0]},${bestPair[1]}`);
      ids = this._mergePair(ids, bestPair, newId);
    }

    return ids;
  }

  _mergePair(ids, pair, newId) {
    const merged = [];
    let i = 0;
    while (i < ids.length) {
      if (i < ids.length - 1 && ids[i] === pair[0] && ids[i + 1] === pair[1]) {
        merged.push(newId);
        i += 2;
      } else {
        merged.push(ids[i]);
        i += 1;
      }
    }
    return merged;
  }

  /** Python'daki decode() ile aynı: id dizisi -> byte dizisi -> UTF-8 string. */
  decode(ids) {
    let totalLen = 0;
    const chunks = ids.map((id) => {
      const bytes = this.vocab.get(id) ?? new Uint8Array(0);
      totalLen += bytes.length;
      return bytes;
    });
    const out = new Uint8Array(totalLen);
    let offset = 0;
    for (const chunk of chunks) {
      out.set(chunk, offset);
      offset += chunk.length;
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(out);
  }
}
