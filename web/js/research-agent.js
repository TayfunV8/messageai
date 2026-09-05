/**
 * Canlı Araştırma Ajanı - JavaScript portu.
 *
 * research_agent.py + web_search.py + scraper.py'nin üçünü birleştiren
 * tarayıcı sürümü. HTML ayrıştırma için tarayıcının kendi `DOMParser`'ını
 * kullanıyoruz - Python/Rust'taki gibi elle state-machine parser yazmaya
 * gerek yok, tarayıcı bunu bizim için zaten yapıyor.
 *
 * CORS notu: DuckDuckGo ve çoğu site, tarayıcıdan doğrudan `fetch()`
 * isteğine izin vermez. Bu yüzden istekler `proxy/worker.js` (Cloudflare
 * Worker) üzerinden geçiriliyor. PROXY_URL'i kendi Worker adresinle
 * değiştirmen gerekiyor (kurulum talimatı worker.js dosyasının başında).
 */

const PROXY_URL = "https://messageai.messageai.workers.dev"; // <-- BURAYI DEĞİŞTİR

import { buildIdf, embed, cosineSimilarity } from "./embeddings.js";

function proxied(targetUrl) {
  return `${PROXY_URL}/?url=${encodeURIComponent(targetUrl)}`;
}

/**
 * DuckDuckGo HTML arayüzünden arama yapar.
 * @returns {Promise<Array<{title: string, url: string, snippet: string}>>}
 */
export async function webSearch(query, maxResults = 5) {
  const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
  const resp = await fetch(proxied(searchUrl));
  if (!resp.ok) {
    throw new Error(`Arama başarısız (HTTP ${resp.status}). Proxy adresini kontrol et.`);
  }
  const html = await resp.text();

  const doc = new DOMParser().parseFromString(html, "text/html");
  const results = [];

  for (const a of doc.querySelectorAll("a.result__a")) {
    if (results.length >= maxResults) break;
    const title = a.textContent.trim();
    let url = a.getAttribute("href") || "";
    url = cleanDdgUrl(url);

    // Snippet genelde aynı sonuç bloğundaki bir sonraki eleman
    const resultBlock = a.closest(".result, .web-result") || a.parentElement;
    const snippetEl = resultBlock ? resultBlock.querySelector(".result__snippet") : null;
    const snippet = snippetEl ? snippetEl.textContent.trim() : "";

    if (title && url) results.push({ title, url, snippet });
  }

  return results;
}

function cleanDdgUrl(href) {
  if (href.includes("uddg=")) {
    try {
      const parsed = new URL(href, "https://duckduckgo.com");
      const uddg = parsed.searchParams.get("uddg");
      if (uddg) return decodeURIComponent(uddg);
    } catch {
      /* olduğu gibi bırak */
    }
  }
  return href;
}

const SKIP_TAGS = new Set(["SCRIPT", "STYLE", "NAV", "FOOTER", "HEADER", "NOSCRIPT", "SVG", "FORM"]);

/**
 * Bir URL'den temiz (gürültüsüz) metin çıkarır.
 * @returns {Promise<string|null>} başarısız olursa null (o kaynağı atla)
 */
export async function fetchPageText(url, maxChars = 3000) {
  try {
    const resp = await fetch(proxied(url));
    if (!resp.ok) return null;
    const contentType = resp.headers.get("Content-Type") || "";
    if (!contentType.includes("text")) return null;

    const html = await resp.text();
    const doc = new DOMParser().parseFromString(html, "text/html");

    // Gürültü elemanlarını kaldır
    for (const tag of SKIP_TAGS) {
      doc.querySelectorAll(tag.toLowerCase()).forEach((el) => el.remove());
    }

    const text = doc.body ? doc.body.textContent : "";
    const cleaned = text
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .join("\n");

    return cleaned.slice(0, maxChars);
  } catch {
    return null;
  }
}

/**
 * Sorguyu arayıp ilk sonuçların içeriğini toplar.
 */
/**
 * Genel olarak daha güvenilir kabul edilen kaynak türleri: resmi kurumlar
 * (.gov, .edu), ansiklopedik/referans siteler, tanınmış haber kuruluşları.
 * Bu bir "doğruluk garantisi" değil - sadece rastgele bir blogdan önce
 * bu tür kaynakları öncelemek için basit bir sezgisel (heuristic) sıralama.
 */
const TRUSTED_DOMAIN_PATTERNS = [
  /\.gov(\.\w+)?$/,
  /\.edu(\.\w+)?$/,
  /(^|\.)wikipedia\.org$/,
  /(^|\.)britannica\.com$/,
  /(^|\.)reuters\.com$/,
  /(^|\.)bbc\.(com|co\.uk)$/,
  /(^|\.)tdk\.gov\.tr$/,
  /(^|\.)who\.int$/,
  /(^|\.)nature\.com$/,
  /(^|\.)developer\.mozilla\.org$/,
];

function trustScore(url) {
  try {
    const hostname = new URL(url).hostname;
    return TRUSTED_DOMAIN_PATTERNS.some((re) => re.test(hostname)) ? 1 : 0;
  } catch {
    return 0;
  }
}

function sortByTrust(hits) {
  // Stabil sıralama: güvenilir olanlar öne alınır, kendi aralarındaki
  // (ve güvenilir olmayanların kendi aralarındaki) orijinal DDG sırası
  // korunur - yani "en alakalıyı at, sadece güvenilirliği ata" değil,
  // "eşit koşulda güvenilir olanı öne çek" mantığı.
  return hits
    .map((hit, idx) => ({ hit, idx, trust: trustScore(hit.url) }))
    .sort((a, b) => b.trust - a.trust || a.idx - b.idx)
    .map((x) => x.hit);
}

/**
 * Sorguyu arayıp ilk sonuçların içeriğini toplar. Güvenilir kaynaklar
 * (varsa) önceliklendirilir.
 */
export async function gatherSources(query, maxSources = 4, perSourceChars = 2000) {
  const hits = sortByTrust(await webSearch(query, maxSources * 2));
  const sources = [];

  for (const hit of hits) {
    if (sources.length >= maxSources) break;
    const text = await fetchPageText(hit.url, perSourceChars);
    if (text && text.length > 200) {
      sources.push({ title: hit.title, url: hit.url, content: text, trusted: trustScore(hit.url) === 1 });
    }
  }

  return sources;
}

/**
 * Model kullanmadan (extractive) özet: sorgu kelimeleriyle en çok
 * örtüşen cümleleri seçer. Her zaman çalışır, model gerektirmez.
 */
export function extractiveSummary(sources, query, maxSentences = 10) {
  const allSentences = [];
  for (const src of sources) {
    const sentences = src.content
      .replace(/\n/g, " ")
      .split(".")
      .map((s) => s.trim())
      .filter((s) => s.length > 30);
    for (const sent of sentences) allSentences.push({ sent, title: src.title });
  }

  if (allSentences.length === 0) {
    return { text: "İlgili kaynaklarda sorguyla ilişkili cümle bulunamadı.", confidence: 0 };
  }

  // embeddings.js'deki (VectorMemory'de zaten test edilmiş) TF-IDF + kosinüs
  // benzerliği tekniğini kullanıyoruz - basit kelime çakışmasından daha
  // güvenilir bir alaka sıralaması sağlıyor (nadir/önemli kelimelere daha
  // çok ağırlık verir, "ve", "bir" gibi kelimeler gürültü yapmaz).
  const idf = buildIdf(allSentences.map((s) => s.sent));
  const queryVec = embed(query, idf);

  const scored = allSentences.map((s) => ({
    ...s,
    score: cosineSimilarity(queryVec, embed(s.sent, idf)),
  }));

  scored.sort((a, b) => b.score - a.score);
  // Gürültü seviyesindeki (hash çakışmasından kaynaklanan sahte küçük
  // benzerlik) eşleşmeleri elemek için gerçek bir eşik koyuyoruz - sadece
  // "sıfırdan büyük" yeterli değil, VectorMemory'de kullandığımız eşikle
  // tutarlı (0.05) bir minimum alaka skoru istiyoruz.
  const MIN_RELEVANCE = 0.05;
  const top = scored.slice(0, maxSentences).filter((s) => s.score > MIN_RELEVANCE);

  if (top.length === 0) {
    return { text: "İlgili kaynaklarda sorguyla ilişkili cümle bulunamadı.", confidence: 0 };
  }

  const text = top.map((s) => `- ${s.sent}. [Kaynak: ${s.title}]`).join("\n");
  return { text, confidence: top[0].score };
}

/**
 * Ana giriş noktası: sorgu -> kaynaklar -> özet.
 *
 * `confidence` alanı, arama sonucunun ne kadar güvenilir olduğunu (en
 * alakalı cümlenin skoru) gösterir. Bu, modelin zayıf üretimine ne kadar
 * güvenileceğine karar vermek için kullanılıyor (bkz. index.html) -
 * arama GÜÇLÜ bir sonuç bulduysa, küçük modelin "yorumuna" hiç gerek
 * kalmayabilir; arama zayıf/boşsa model devreye girer.
 */
export async function runResearch(query) {
  const sources = await gatherSources(query, 5); // 4 -> 5: biraz daha geniş kaynak havuzu
  if (sources.length === 0) {
    return { query, sources: [], summary: "Hiçbir kaynağa ulaşılamadı (proxy adresini kontrol et).", confidence: 0 };
  }
  const { text, confidence } = extractiveSummary(sources, query);
  return { query, sources, summary: text, confidence };
}
