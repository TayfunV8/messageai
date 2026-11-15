/**
 * Vektör Hafıza Deposu - "kendi FAISS'imiz".
 *
 * NEDEN FAISS/ChromaDB DEĞİL: Bu kütüphaneler binlerce/milyonlarca vektör
 * arasında ultra hızlı arama için (approximate nearest neighbor) var.
 * Bizim senaryomuzda (bir kullanıcının yüklediği birkaç PDF, en fazla
 * birkaç bin chunk) doğrudan tüm vektörlerle karşılaştırma (brute-force)
 * milisaniyeler sürer - ekstra bir kütüphaneye gerek yok. Chunk sayısı
 * on binleri geçerse bu yaklaşım yavaşlar, o zaman gerçek bir ANN
 * kütüphanesi gerekir (bizim ölçeğimizde değil).
 */

import { buildIdf, embed, cosineSimilarity } from "./embeddings.js";

/**
 * Uzun bir metni, aralarında örtüşme (overlap) olan parçalara böler.
 * Örtüşme önemli: bir cümle tam sınırda kesilirse anlamını kaybetmesin diye.
 */
export function chunkText(text, chunkSize = 500, overlap = 100) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    chunks.push(text.slice(start, end));
    if (end === text.length) break;
    start = end - overlap;
  }
  return chunks;
}

export class VectorMemory {
  constructor() {
    /** @type {Array<{id: number, text: string, docName: string, vector: Float32Array}>} */
    this.entries = [];
    this.idf = new Map();
    this._nextId = 0;
  }

  /**
   * Bir dokümanı (PDF'ten çıkarılmış metin) belleğe ekler. Doküman
   * chunk'lara bölünür, IDF tablosu yeniden hesaplanır (yeni kelime
   * dağılımını yansıtsın diye), her chunk için embedding üretilir.
   */
  addDocument(docName, fullText, chunkSize = 500, overlap = 100) {
    const chunks = chunkText(fullText, chunkSize, overlap);

    // IDF tablosunu TÜM chunk'lar (eski + yeni) üzerinden yeniden hesapla.
    // Küçük ölçekte (bizim senaryomuz) bunu her ekleme sonrası yapmak ucuz.
    const allTexts = [...this.entries.map((e) => e.text), ...chunks];
    this.idf = buildIdf(allTexts);

    // Eski vektörleri de yeni IDF ile yeniden hesapla (tutarlılık için)
    for (const entry of this.entries) {
      entry.vector = embed(entry.text, this.idf);
    }

    for (const chunkTextValue of chunks) {
      this.entries.push({
        id: this._nextId++,
        text: chunkTextValue,
        docName,
        vector: embed(chunkTextValue, this.idf),
      });
    }

    return chunks.length;
  }

  /** En alakalı topK chunk'ı döndürür (skorla birlikte). */
  search(query, topK = 4) {
    if (this.entries.length === 0) return [];
    const queryVec = embed(query, this.idf);

    const scored = this.entries.map((e) => ({
      text: e.text,
      docName: e.docName,
      score: cosineSimilarity(queryVec, e.vector),
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  get documentCount() {
    return new Set(this.entries.map((e) => e.docName)).size;
  }

  get chunkCount() {
    return this.entries.length;
  }

  clear() {
    this.entries = [];
    this.idf = new Map();
    this._nextId = 0;
  }
}
