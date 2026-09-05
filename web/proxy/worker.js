/**
 * Cloudflare Worker - Genel amaçlı CORS proxy.
 *
 * NEDEN GEREKLİ: Tarayıcı güvenlik kuralı (CORS), bir web sitesinin JS'inin
 * başka bir domain'e (örn. duckduckgo.com) doğrudan istek atmasını
 * genelde engeller (hedef site izin vermediği sürece). Bu, bizim
 * kontrolümüzde olmayan bir tarayıcı kısıtı. Çözüm: istek önce bize ait
 * bu Worker'a gider, Worker (tarayıcı değil, Cloudflare'in sunucusu)
 * hedef siteye isteği yapar ve sonucu CORS izniyle geri döner.
 *
 * MALİYET: Cloudflare Workers ücretsiz katmanı günde 100.000 istek
 * verir, kredi kartı istemez. Kişisel/küçük ölçekli kullanım için $0.
 *
 * KURULUM (tek seferlik, ~2 dakika):
 * 1. https://dash.cloudflare.com adresinde ücretsiz hesap aç.
 * 2. Sol menüden "Workers & Pages" > "Create" > "Create Worker".
 * 3. Açılan kod editörüne bu dosyanın TAMAMINI yapıştır.
 * 4. "Deploy" de. Sana bir adres verecek: https://senin-worker-adin.workers.dev
 * 5. O adresi web/js/research-agent.js dosyasındaki PROXY_URL değişkenine yapıştır.
 */

const ALLOWED_ORIGINS = "*"; // İstersen kendi GitHub Pages adresinle sınırlayabilirsin

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const target = url.searchParams.get("url");

    if (!target) {
      return new Response("Kullanım: ?url=<hedef-url>", { status: 400 });
    }

    let targetUrl;
    try {
      targetUrl = new URL(target);
    } catch {
      return new Response("Geçersiz hedef URL", { status: 400 });
    }

    // Basit bir güvenlik önlemi: sadece http/https şemasına izin ver
    // (dosya sistemi vb. iç kaynaklara erişimi engeller).
    if (!["http:", "https:"].includes(targetUrl.protocol)) {
      return new Response("Sadece http/https URL'lerine izin verilir", { status: 400 });
    }

    try {
      const upstreamResp = await fetch(targetUrl.toString(), {
        headers: {
          // Gerçek bir tarayıcı gibi görünmek için (bot korumasını azaltır).
          // Worker sunucu tarafında çalıştığı için bu header'ları serbestçe
          // ayarlayabiliyoruz - tarayıcı JS'i bunu yapamazdı.
          "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
          "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8",
        },
      });

      const body = await upstreamResp.arrayBuffer();
      const headers = new Headers();
      headers.set("Access-Control-Allow-Origin", ALLOWED_ORIGINS);
      headers.set("Content-Type", upstreamResp.headers.get("Content-Type") || "text/plain");

      return new Response(body, { status: upstreamResp.status, headers });
    } catch (err) {
      return new Response(`Proxy hatası: ${err.message}`, {
        status: 502,
        headers: { "Access-Control-Allow-Origin": ALLOWED_ORIGINS },
      });
    }
  },
};
