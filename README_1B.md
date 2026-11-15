# Local AI v13 – 1B model güncellemesi

## Paket içeriği
- `local_ai_egitim_1B.ipynb` – yeni eğitim defteri (Colab / kendi GPU'n)
- `server.js`, `model/model.js`, `model/tokenizer.js`, `model/weights_loader_server.js` – güncel sunucu (GQA, özel token, 2 GB+ ağırlık okuma, fp16)
- `web/js/model.js`, `web/js/tokenizer.js` – aynı dosyaların tarayıcı kopyası (1B model tarayıcıda çalışmaz, sunucu kullan)

## Kurulum
1. Defteri **en az 24 GB VRAM'li** GPU'da çalıştır (A100 / L4 / RTX 3090-4090). T4 yetmez.
2. Ayarlar hücresindeki sayıları (veri miktarı, `PRETRAIN_EPOCHS`, `LOCAL_TEXT_GLOB`, `LOCAL_DIALOG_GLOB`) kontrol et, hücreleri sırayla çalıştır.
3. Bittiğinde Drive'daki `.../export/` klasöründen 4 dosyayı al: `weights.bin`, `weights_manifest.json`, `config.json`, `tokenizer_merges.json` → sunucuda `model_files/` içine koy.
4. Yeni `server.js` + `model/` dosyalarını repona yükle. Sunucu RAM'i: ~4.5 GB model + Node payı → **en az 6-8 GB**.
5. Siteden istek: `POST /api/chat` `{ "message": "...", "history": [ {"role":"user","content":"..."}, {"role":"assistant","content":"..."} ] }` (`history` isteğe bağlı). Eski `{"prompt": "..."}` da mesaj olarak kabul edilir.

## Önemli ayarlar (defterde `Ayarlar` hücresi)
| Ayar | Varsayılan | Not |
|---|---|---|
| `PRETRAIN_EPOCHS` | 2.0 | Artırma; val loss 3 kez üst üste yükselirse eğitim zaten durur |
| `WEB_MAX_DOCS` | 300 000 | RAM/süre için azalt/artır |
| `BPE_VOCAB` | 32 000 | 64 000 yapılabilir |
| `BLOCK_SIZE` | 2048 | 4096 = ~2x VRAM/hız maliyeti |
| `TRT_FETCH_FULL_TEXT` | False | Sadece TRT'nin resmi RSS başlık+spotu; tam metin robots.txt'e bakar |
| `FILTER_CLICKBAIT` | True | Tık tuzağı başlıkları ele |

## Bilinen sınırlar
- 1B modelin JavaScript ile CPU'da çıkarımı yavaştır. Hız gerekiyorsa GPU'lu sunucu veya llama.cpp tarzı bir motor gerekir.
- Defterin PyTorch/GPU hücreleri bu pakette GPU olmadan çalıştırılamadı; veri hattı, tokenizer, loss maskesi, JS model ve sunucu test edildi.
