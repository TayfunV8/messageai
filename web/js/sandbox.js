/**
 * Ana sayfadan çağrılan sandbox kontrolcüsü. Her çalıştırma için TAZE bir
 * worker oluşturuyoruz (bir öncekinden state sızmasın diye) ve zaman
 * aşımında worker'ı sert şekilde sonlandırıyoruz (`terminate()`).
 */
export class CodeSandbox {
  /**
   * @param {string} code Çalıştırılacak kod
   * @param {"javascript"|"python"} language
   * @param {number} timeoutMs Python ilk çalıştırmada Pyodide indirmesi
   *   (birkaç MB) yüzünden daha uzun sürebilir - buna göre süre ver.
   */
  run(code, language = "javascript", timeoutMs = 8000) {
    return new Promise((resolve) => {
      const worker = new Worker("./js/sandbox-worker.js");
      let settled = false;

      const finish = (result) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        worker.terminate();
        resolve(result);
      };

      const timer = setTimeout(() => {
        finish({
          success: false,
          output: "",
          error: `Zaman aşımı (${timeoutMs}ms). Kod çok uzun sürdü ya da sonsuz döngüye girmiş olabilir.`,
        });
      }, timeoutMs);

      worker.onmessage = (e) => finish(e.data);
      worker.onerror = (e) => finish({ success: false, output: "", error: e.message || "Worker hatası" });

      worker.postMessage({ code, language });
    });
  }
}

/**
 * Bir metin içindeki ```js ... ``` veya ```python ... ``` bloklarını bulur.
 * Sohbet mesajlarında modelin ürettiği kod bloklarına otomatik "▶ Çalıştır"
 * butonu eklemek için kullanılır.
 */
export function extractCodeBlocks(text) {
  const regex = /```(javascript|js|python|py)?\n([\s\S]*?)```/g;
  const blocks = [];
  let match;
  while ((match = regex.exec(text)) !== null) {
    const langRaw = (match[1] || "javascript").toLowerCase();
    const language = langRaw === "js" ? "javascript" : langRaw === "py" ? "python" : langRaw;
    blocks.push({ language, code: match[2].trim() });
  }
  return blocks;
}
