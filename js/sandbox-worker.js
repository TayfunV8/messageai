/**
 * Kod Çalıştırma Worker'ı (izole ortam).
 *
 * Bilinçli olarak "classic" worker (type:module DEĞİL) - çünkü Pyodide'nin
 * yüklenmesi için `importScripts()` gerekiyor, bu sadece classic worker'larda
 * var. Ana sayfadan `new Worker("./js/sandbox-worker.js")` ile (type
 * belirtmeden) çağrılmalı.
 *
 * GÜVENLİK SINIRLARI (dürüstçe belirtiyoruz):
 * - Worker, DOM'a / cookie'lere / localStorage'a / ana sayfanın hiçbir
 *   verisine erişemez (tarayıcının garanti ettiği izolasyon).
 * - Sonsuz döngü: sandbox.js tarafında zaman aşımı + worker.terminate() ile
 *   durdurulur.
 * - Bellek bombası (aşırı büyük dizi/nesne oluşturma): tarayıcı seviyesinde
 *   tam engellenemez - bu JS/tarayıcı motorunun kendi kısıtı, üstesinden
 *   gelemediğimiz gerçek bir sınır.
 * - Network isteği (fetch): Worker içinde teknik olarak mümkün - kodun
 *   dışarıya istek atmasını istemiyorsan bunu ayrıca engellemek gerekir
 *   (şu an engellenmiyor, bilerek basit tutuldu).
 */

let pyodideInstance = null;

async function ensurePyodide() {
  if (pyodideInstance) return pyodideInstance;
  importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.1/full/pyodide.js");
  // eslint-disable-next-line no-undef
  pyodideInstance = await loadPyodide();
  return pyodideInstance;
}

function runJavaScript(code) {
  const logs = [];
  const fakeConsole = {
    log: (...args) => logs.push(args.map(String).join(" ")),
    error: (...args) => logs.push("[hata] " + args.map(String).join(" ")),
    warn: (...args) => logs.push("[uyarı] " + args.map(String).join(" ")),
  };

  try {
    // 'console' parametresini gölgeleyerek çıktıyı yakalıyoruz. Worker
    // zaten izole bir bağlamda (yukarıdaki güvenlik notuna bak).
    const fn = new Function("console", code);
    const result = fn(fakeConsole);
    if (result !== undefined) logs.push(`=> ${String(result)}`);
    return { success: true, output: logs.join("\n"), error: null };
  } catch (err) {
    return { success: false, output: logs.join("\n"), error: err.message };
  }
}

async function runPython(code) {
  try {
    const pyodide = await ensurePyodide();
    const logs = [];
    pyodide.setStdout({ batched: (s) => logs.push(s) });
    pyodide.setStderr({ batched: (s) => logs.push("[hata] " + s) });

    const result = await pyodide.runPythonAsync(code);
    if (result !== undefined && result !== null) logs.push(`=> ${String(result)}`);

    return { success: true, output: logs.join("\n"), error: null };
  } catch (err) {
    return { success: false, output: "", error: err.message };
  }
}

self.onmessage = async (e) => {
  const { code, language } = e.data;
  const result = language === "python" ? await runPython(code) : runJavaScript(code);
  self.postMessage(result);
};
