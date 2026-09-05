/**
 * Service Worker - Model dosyalarını tarayıcıda KALICI önbelleğe alır.
 *
 * NEDEN GEREKLİ: Normal tarayıcı önbelleği (HTTP cache) "en iyi çaba"
 * garantisi verir - tarayıcı istediği zaman silebilir, GitHub Pages'in
 * gönderdiği cache header'larına bağlıdır. Service Worker ile BİZ kontrolü
 * elimize alıyoruz: "bu dosyalar bir kere inince, biz silmeden asla tekrar
 * indirilmesin" diyoruz. Sonuç: ilk ziyaretten sonra site (model dahil)
 * İNTERNETSİZ bile açılır, ikinci ziyaret saniyeler değil MİLİSANİYELER sürer.
 *
 * VERSİYON NOTU: Modelini yeniden eğitip dosyaları güncellersen, aşağıdaki
 * CACHE_NAME'in sonundaki sayıyı (v1 -> v2) artır ve tekrar yükle - yoksa
 * tarayıcı eski modeli önbellekten göstermeye devam eder (bilerek böyle,
 * yanlışlıkla eski bir modeli yenisiyle karıştırmamak için).
 */

const CACHE_NAME = "local-ai-cache-v1";

// Site açılır açılmaz indirilip kalıcı önbelleğe alınacak dosyalar.
// Model dosyaları en büyük (ve en değerli) kısım - bunları ÖNCELİKLE
// önbelleğe alıyoruz ki bir daha asla yeniden inmesinler.
const PRECACHE_URLS = [
  "./",
  "./index.html",
  "./js/tokenizer.js",
  "./js/model.js",
  "./js/weights_loader.js",
  "./js/research-agent.js",
  "./js/embeddings.js",
  "./js/vector-memory.js",
  "./js/pdf-reader.js",
  "./js/sandbox.js",
  "./js/sandbox-worker.js",
  "./model_files/weights.bin",
  "./model_files/weights_manifest.json",
  "./model_files/config.json",
  "./model_files/tokenizer_merges.json",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      // Her URL'i ayrı ayrı denemek daha dayanıklı: model_files henüz
      // yüklenmemişse (kullanıcı daha eğitim yapmadıysa) tüm precache
      // işlemi çökmesin, sadece o dosyaları atlasın.
      return Promise.allSettled(
        PRECACHE_URLS.map((url) =>
          cache.add(url).catch((err) => console.warn(`[SW] Önbelleğe alınamadı: ${url}`, err))
        )
      );
    })
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names
          .filter((name) => name !== CACHE_NAME) // eski versiyonları temizle
          .map((name) => caches.delete(name))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Büyük model dosyaları: cache-first (önce önbellekten, hiç yoksa ağdan).
  // Bunlar nadiren değişir (sadece yeniden eğitim sonrası) - hız önceliği.
  const isModelFile = url.pathname.includes("/model_files/");

  if (isModelFile) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return response;
        });
      })
    );
    return;
  }

  // Site kodu (HTML/JS): network-first (önce ağdan güncel sürümü almaya
  // çalış, ağ yoksa önbellekten göster). Böylece kod güncellemelerin
  // (bu dosyaları tekrar GitHub'a yüklediğinde) kullanıcıya ulaşır, ama
  // internet kesikken de site yine açılır.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
