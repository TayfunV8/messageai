/**
 * Cloudflare Worker - basit CORS proxy.
 *
 * NEDEN GEREKLİ: Tarayıcı, güvenlik nedeniyle (CORS) doğrudan başka
 * sitelere (DuckDuckGo, haber siteleri vb.) fetch() isteği atamaz. Bu
 * Worker, tarayıcı ile hedef site arasında bir "aracı" (proxy) görevi
 * görür: tarayıcı bu Worker'a istek atar, Worker (sunucu tarafında,
 * CORS kısıtlaması olmadan) hedef siteye gidip cevabı alır, tarayıcıya
 * CORS izniyle geri verir.
 *
 * KURULUM (README, Adım 2):
 * 1. https://dash.cloudflare.com adresinde ücretsiz hesap aç.
 * 2. Workers & Pages -> Create -> Create Worker.
 * 3. Açılan editöre bu dosyanın TAMAMINI yapıştır, Deploy'a bas.
 * 4. Sana verilen adresi (https://xxx.workers.dev) kopyala.
 * 5. js/research-agent.js dosyasının en üstündeki PROXY_URL değerini
 *    bu adresle değiştir.
 *
 * GÜVENLİK NOTU: Bu proxy, kullanıcı isteğiyle (güvenilir site filtresi
 * kaldırıldığı için) HERHANGİ BİR http/https adresine istek atabilir -
 * bilerek kısıtlama koymadık (aksi halde bazı haber siteleri erişilemez
 * olurdu). Sadece http/https dışındaki adresleri (file:, data: vb.)
 * reddediyoruz.
 */

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "*",
};

function jsonError(message, status) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
  });
}

export default {
  async fetch(request) {
    // Tarayıcının gönderdiği CORS "preflight" (OPTIONS) isteğine hemen izin ver
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    const requestUrl = new URL(request.url);
    const targetUrl = requestUrl.searchParams.get("url");

    if (!targetUrl) {
      return jsonError("Eksik parametre: ?url= belirtilmedi.", 400);
    }

    let parsed;
    try {
      parsed = new URL(targetUrl);
    } catch {
      return jsonError("Geçersiz URL.", 400);
    }

    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return jsonError("Sadece http/https adresleri desteklenir.", 400);
    }

    try {
      const upstreamResponse = await fetch(targetUrl, {
        headers: {
          // Bazı siteler User-Agent göndermeyen istekleri reddediyor
          "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        redirect: "follow",
      });

      const contentType = upstreamResponse.headers.get("Content-Type") || "text/plain";
      const body = await upstreamResponse.arrayBuffer();

      return new Response(body, {
        status: upstreamResponse.status,
        headers: {
          ...CORS_HEADERS,
          "Content-Type": contentType,
        },
      });
    } catch (err) {
      return jsonError(`Hedef siteye ulaşılamadı: ${err.message}`, 502);
    }
  },
};
