# Yerel Yapay Zeka — Web Sitesi

Bu klasör, tamamen tarayıcıda çalışan (sunucu gerektirmeyen) yapay zeka
sitesidir. Aşağıdaki adımları sırayla takip et.

## Klasör yapısı

```
web/
├── index.html              <- Ana sayfa, tarayıcıda açılan bu
├── js/
│   ├── tokenizer.js         <- Metni sayıya çevirir
│   ├── model.js              <- Modelin kendisi (sadece tahmin/inference)
│   ├── weights_loader.js    <- Ağırlık dosyalarını okur
│   ├── research-agent.js    <- İnternette arama yapar
│   ├── pdf-reader.js         <- PDF/TXT dosyalarını okur
│   ├── embeddings.js         <- Metni "anlam vektörüne" çevirir (sıfırdan)
│   └── vector-memory.js      <- Belgeleri saklar, alakalı parçayı bulur (kendi FAISS'imiz)
├── proxy/
│   └── worker.js             <- Cloudflare'e AYRI yüklenecek (CORS için)
└── model_files/              <- BURAYA senin eğittiğin model dosyaları gelecek
    ├── weights.bin           (export scriptinden)
    ├── weights_manifest.json (export scriptinden)
    ├── config.json           (export scriptinden)
    └── tokenizer_merges.json (Colab'dan)
```

## Adım 1 — Modeli eğit ve dışa aktar

1. `local_ai_egitim.ipynb` dosyasını Colab'da çalıştır (GPU ile).
2. İnen `best_model.pt` ve `tokenizer_merges.json` dosyalarını bilgisayarına al.
3. Kendi PC'nde (Python + torch kurulu terminalde):
   ```bash
   python3 brain/export_to_web_format.py best_model.pt web_export/
   ```
4. `web_export/` klasöründe oluşan `weights.bin`, `weights_manifest.json`,
   `config.json` dosyalarını, indirdiğin `tokenizer_merges.json` ile birlikte
   `web/model_files/` klasörüne kopyala (4 dosya da orada olmalı).

## Adım 2 — CORS Proxy'yi kur (araştırma özelliği için)

Bu adımı atlarsan site çalışır ama "🔍 Ara" butonu çalışmaz.

1. https://dash.cloudflare.com adresinde ücretsiz hesap aç.
2. **Workers & Pages** → **Create** → **Create Worker**.
3. Açılan editöre `proxy/worker.js` dosyasının TAMAMINI yapıştır, **Deploy** de.
4. Sana verilen adresi kopyala (örn. `https://ornek-worker.workers.dev`).
5. `js/research-agent.js` dosyasını aç, en üstteki şu satırı bul:
   ```js
   const PROXY_URL = "https://SENIN-WORKER-ADIN.workers.dev";
   ```
   ve kendi adresinle değiştir.

## Adım 3 — GitHub'a yükle ve yayınla

1. GitHub'da yeni bir repo oluştur (örn. `local-ai`).
2. Bu `web/` klasörünün İÇİNDEKİ her şeyi (index.html, js/, model_files/, proxy/)
   reponun ANA dizinine yükle (web klasörünün kendisini değil, içindekileri).
3. Repo ayarlarından **Settings** → **Pages** → **Source** kısmından
   **Deploy from a branch** → **main** → **/ (root)** seç, **Save** de.
4. Birkaç dakika sonra sana bir link verecek:
   `https://kullanici-adin.github.io/local-ai/`
5. O linki aç — model otomatik yüklenip sohbet arayüzü çıkacak.

## Hafıza / Doküman özelliği (Aşama 3)

Sitede "📄 PDF/TXT ekle" butonuyla dosya yükleyebilirsin. Yüklenen belge:
1. Metne çevrilir (PDF.js ile, ücretsiz/açık kaynak bir kütüphane - PDF
   formatı sıfırdan yazılamayacak kadar karmaşık olduğu için bilinçli
   olarak kullanıldı).
2. Parçalara bölünür (chunk'lama).
3. Her parça için kendi yazdığımız bir "anlam vektörü" (embedding)
   üretilir - bu kısım tamamen sıfırdan, hiçbir kütüphane kullanmadan.
4. Sohbet ederken, sorduğun soruya en alakalı parçalar otomatik bulunup
   modele "bağlam" olarak verilir (buna RAG - Retrieval Augmented
   Generation denir).

Not: Hafıza sadece o an açık olan sekmede tutulur, sayfayı kapatıp
açtığında sıfırlanır (kalıcı hafıza istenirse tarayıcının IndexedDB'si
eklenebilir - ayrı bir geliştirme, istersen sonra ekleriz).

## Kod Çalıştırma Sandbox'ı (Aşama 4)

Sitede "💻 Kod Çalıştır" paneli var — JavaScript veya Python kodu yazıp
çalıştırabilirsin. Ayrıca AI'nin sohbet cevabında ` ```js ` veya
` ```python ` bloğu varsa, altına otomatik "▶ Çalıştır" butonu eklenir.

**Nasıl çalışıyor:** Kod, ana sayfadan tamamen izole bir **Web Worker**
içinde çalıştırılıyor — DOM'a, çerezlere, sayfanın hiçbir verisine
erişemiyor. Python için **Pyodide** (gerçek bir Python yorumlayıcısının
WebAssembly'ye derlenmiş hali, ücretsiz/açık kaynak) kullanılıyor; ilk
Python çalıştırmasında birkaç MB indirir, sonraki çalıştırmalar hızlıdır.

**Güvenlik sınırları (dürüstçe):**
- ✅ Sonsuz döngü → zaman aşımıyla (8 saniye) otomatik durdurulur.
- ✅ DOM/çerez/localStorage erişimi → tamamen engelli (Worker izolasyonu).
- ⚠️ Aşırı bellek tüketimi (dev dizi/nesne oluşturma) → tarayıcı seviyesinde
  tam engellenemez, bu bizim değil tarayıcının kendi sınırı.
- ⚠️ Ağ isteği (fetch) → Worker içinden teknik olarak mümkün, şu an ayrıca
  engellenmiyor.

**JS'te önemli bir detay:** Fonksiyonun son satırı otomatik ekrana
basılmaz (Python REPL'in aksine) — çıktı görmek için `console.log(...)`
kullanmalısın, ya da açıkça `return değer;` yazmalısın.

## Cevapların formatı: "Bulunan bilgi" ve "Modelin yorumu" ayrı gösterilir

Her mesajda site otomatik olarak DuckDuckGo'da arama yapar (açık/kapalı
anahtarı var) ve/veya yüklediğin PDF'lerde arar. Arama sonucunun ne kadar
güvenilir olduğu bir "güven skoru" ile ölçülüyor:

- **Güven YÜKSEKSE** (≥0.15): Sadece **📚 Bulunan bilgiler (güvenilir)**
  gösterilir - küçük modelin üretimi hiç çalıştırılmaz. Zaten güvenilir
  bir cevap varken modelin üstüne ekleyeceği metin sadece gürültü/yanıltıcı
  olur - hem zaman kaybettirir hem kaliteyi düşürür.
- **Güven DÜŞÜKSE**: Hem **📚 Bulunan bilgiler** (varsa) hem **🤖 Modelin
  yorumu** (spekülatif, tutarsız olabilir) ayrı ayrı gösterilir.

Bu ayrımı bilinçli yaptık: ikisini karıştırıp tek bir "cevap" gibi
sunmak, hangi kısmın gerçek/kaynaklı olduğunu gizler ve yanıltıcı olur.

## Önbelleğe alma (Service Worker) — "her girişte yeniden yüklenmesin" sorunu

Site bir **Service Worker** kullanıyor: ilk ziyarette model dosyaları
(birkaç MB) indirilip tarayıcıda **kalıcı olarak** saklanıyor. Sonraki
ziyaretlerde (hatta internetsizken bile) site anında açılıyor, hiçbir
şey yeniden inmiyor.

**Modeli yeniden eğitip dosyaları güncellersen:** `sw.js` dosyasındaki
`CACHE_NAME = "local-ai-cache-v1"` satırındaki sayıyı artır (`v1` → `v2`)
ve GitHub'a öyle yükle — yoksa tarayıcı hâlâ eski (önbellekteki) modeli
gösterir. Kullanıcılar için de sitede sağ üstte **"🗑️ Önbelleği temizle"**
butonu var, tek tıkla eski önbelleği silip yeniden yükler.

## Eğitim verisi artık Türkçe Wikipedia'dan otomatik geliyor

Colab not defterindeki **Adım 2**, Hugging Face'in ücretsiz
`wikimedia/wikipedia` veri setinden Türkçe makaleleri otomatik çekiyor
(varsayılan 8000 makale, `NUM_ARTICLES` ile ayarlanabilir). Bu, senin
elle yazman gereken metinden çok daha büyük ve düzgün Türkçe içeren
gerçek bir eğitim verisi sağlıyor - model artık gerçek cümle
kalıplarını görüyor.

**Performans notu:** Tokenizer eğitimi ve kodlama, doğal dildeki kelime
tekrarından faydalanacak şekilde optimize edildi (benzersiz kelime +
frekans sayımı) - bu optimizasyon olmadan Wikipedia boyutunda bir metin
saatlerce sürerdi, şimdi birkaç dakikada bitiyor (test ettik: 5.7 MB
metin, 8000 kelimelik sözlük → ~57 saniye).

**Kaggle Notebooks desteği:** Aynı `.ipynb` dosyası Kaggle'da da
çalışır - kod, Colab'a mı Kaggle'a mı yüklendiğini otomatik algılıyor.
Tek fark: Kaggle'da dosya indirme otomatik değil, sağ paneldeki
"Output" sekmesinden elle indirmen gerekiyor (notebook içinde bu
belirtiliyor).

## Önemli notlar

- **Model dosyaları büyükse** (birkaç yüz MB), GitHub'ın dosya boyutu
  sınırına (100MB/dosya) takılabilirsin. Bu durumda modeli küçük tutman
  gerekir (Colab not defterindeki `n_layer`, `n_embd` değerlerini düşür)
  ya da Git LFS kullanman gerekir (ayrı bir konu, gerekirse anlatırım).
- **Site herkese açık olacak** — model dosyaların da dahil olmak üzere
  reponun her şeyi herkes tarafından görülebilir/indirilebilir (GitHub
  public repo ise). Bunu istemiyorsan repoyu private yap, ama o zaman
  GitHub Pages ücretsiz katmanda çalışmaz (Pages için public repo gerekir).
- **Proxy adresini boş bırakırsan** sohbet (chat) kısmı yine çalışır,
  sadece "🔍 Ara" özelliği hata verir — bu beklenen bir durumdur.
