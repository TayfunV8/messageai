/*
 * Service Worker - web arayüzünü önbelleğe alır.
 *
 * ÖNEMLİ: V9 model dosyaları artık tarayıcıda çalışmadığı için
 * model_files/ ve weights.bin burada kesinlikle önbelleğe alınmaz.
 */

const CACHE_NAME = "local-ai-cache-v3";

const PRECACHE_URLS = [
  "./",
  "./index.html",
  "./js/research-agent.js",
  "./js/embeddings.js",
  "./js/vector-memory.js",
  "./js/pdf-reader.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      Promise.allSettled(
        PRECACHE_URLS.map((url) =>
          cache.add(url).catch((err) =>
            console.warn(`[SW] Önbelleğe alınamadı: ${url}`, err)
          )
        )
      )
    )
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Model dosyalarını hiçbir şekilde cache'leme veya indirme.
  if (url.pathname.includes("/model_files/")) return;

  // API isteklerine Service Worker müdahale etmesin.
  if (url.pathname.includes("/api/")) return;

  // Site kodu: network-first, ağ yoksa cache.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) =>
          cache.put(event.request, clone).catch(() => {})
        );
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
