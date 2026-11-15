/**
 * PDF Okuyucu - Mozilla'nın PDF.js kütüphanesini kullanır (index.html'de
 * CDN üzerinden yüklenir). PDF formatı (sıkıştırma, font kodlama, xref
 * tabloları) sıfırdan güvenli şekilde yazılamayacak kadar karmaşık;
 * PDF.js açık kaynak, ücretsiz ve tarayıcılarda (Firefox'un içinde bile)
 * kullanılan endüstri standardı - bu yüzden bilinçli olarak bunu tercih
 * ettik. Metni çıkardıktan SONRAKİ her şey (chunk'lama, embedding, arama,
 * modelin kullanması) tamamen kendi yazdığımız kod.
 */

/** File nesnesinden (input[type=file]) tüm sayfaların metnini çıkarır. */
export async function extractPdfText(file) {
  if (typeof window.pdfjsLib === "undefined") {
    throw new Error("PDF.js yüklenmemiş. index.html'de CDN script etiketinin eklendiğinden emin ol.");
  }

  const arrayBuffer = await file.arrayBuffer();
  const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;

  let fullText = "";
  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const content = await page.getTextContent();
    const pageText = content.items.map((item) => item.str).join(" ");
    fullText += pageText + "\n\n";
  }

  return fullText.trim();
}

/** Düz .txt dosyası için (PDF.js gerektirmez, doğrudan okur). */
export async function extractTxtText(file) {
  return await file.text();
}

/** Dosya uzantısına göre doğru çıkarıcıyı seçer. */
export async function extractTextFromFile(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".pdf")) return extractPdfText(file);
  if (name.endsWith(".txt") || name.endsWith(".md")) return extractTxtText(file);
  throw new Error(`Desteklenmeyen dosya türü: ${file.name} (sadece .pdf, .txt, .md)`);
}
