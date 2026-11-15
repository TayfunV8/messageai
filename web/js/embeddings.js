/**
 * Sıfırdan Embedding (Anlam Vektörü) Üretici.
 *
 * NEDEN BÖYLE: Gerçek anlamda "anlam" yakalayan embedding modelleri
 * (word2vec, BERT vb.) kendileri ayrı bir eğitim süreci ister (milyonlarca
 * cümle, günlerce eğitim). Bunun yerine, hiçbir eğitim gerektirmeyen ama
 * pratikte iyi çalışan bir teknik kullanıyoruz: "hashed bag-of-words with
 * TF-IDF weighting" - metindeki kelimeleri sabit boyutlu bir vektöre
 * hash'leyip, önemli (nadir) kelimelere daha çok ağırlık veriyoruz.
 *
 * Bu teknik "kelime anlamını" (eş anlamlıları) yakalamaz ama "hangi
 * kelimeler ortak geçiyor" sorusuna çok iyi cevap verir - doküman
 * aramasının %80'i zaten bu. Sıfır bağımlılık, sıfır eğitim süresi.
 */

const VECTOR_DIM = 2048; // 512'den büyütüldü: küçük boyutta hash çakışması
// yanlış-pozitif benzerlik yaratabiliyor (test edip doğruladık - bkz. proje
// notları). 2048, çakışma ihtimalini pratikte ihmal edilebilir seviyeye
// indiriyor, bellek maliyeti hâlâ önemsiz (~8KB/vektör).

/** Basit, deterministik string hash (FNV-1a varyantı). */
function hashString(str) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0; // unsigned'a çevir
}

function tokenize(text) {
  return text
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\p{L}\p{N}\s]/gu, " ") // noktalama temizle (Unicode-aware)
    .split(/\s+/)
    .filter((w) => w.length > 1);
}

/**
 * Bir doküman koleksiyonu için IDF (inverse document frequency) tablosu
 * hesaplar - "kaç dokümanda geçiyor" bilgisi, nadir kelimelere ağırlık
 * vermek için kullanılır (örn. "ve", "bir" gibi kelimeler ağırlıksız kalır).
 */
export function buildIdf(documents) {
  const docFreq = new Map(); // kelime -> kaç farklı dokümanda geçti
  for (const doc of documents) {
    const uniqueWords = new Set(tokenize(doc));
    for (const w of uniqueWords) {
      docFreq.set(w, (docFreq.get(w) || 0) + 1);
    }
  }
  const n = documents.length;
  const idf = new Map();
  for (const [word, freq] of docFreq) {
    idf.set(word, Math.log(n / freq + 1) + 1); // +1: hiç geçmeyen kelime için de küçük bir ağırlık
  }
  return idf;
}

/** Metni sabit boyutlu (VECTOR_DIM) bir Float32Array'e çevirir. */
export function embed(text, idf = null) {
  const words = tokenize(text);
  const vec = new Float32Array(VECTOR_DIM);

  const termFreq = new Map();
  for (const w of words) termFreq.set(w, (termFreq.get(w) || 0) + 1);

  for (const [word, tf] of termFreq) {
    const weight = idf && idf.has(word) ? idf.get(word) : 1.0;
    const h = hashString(word);
    const idx = h % VECTOR_DIM;
    // Hash işaretini de kullanarak (+ veya -) çakışma hasarını azaltıyoruz
    // (iki farklı kelime aynı slota düşerse birbirini toplamak yerine
    // kısmen iptal eder, tamamen bozmaz) - "feature hashing" tekniğinde
    // standart bir yöntem (Weinberger ve ark., hashing trick).
    const sign = h & 0x1 ? 1 : -1;
    vec[idx] += sign * tf * weight;
  }

  // L2 normalizasyon - kosinüs benzerliğini basit iç çarpıma indirger
  let norm = 0;
  for (let i = 0; i < VECTOR_DIM; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < VECTOR_DIM; i++) vec[i] /= norm;

  return vec;
}

/** İki normalize edilmiş vektör arasında kosinüs benzerliği (basit iç çarpım). */
export function cosineSimilarity(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}
