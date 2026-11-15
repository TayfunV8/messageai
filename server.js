import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs/promises";
import { BpeTokenizer } from "./model/tokenizer.js";
import { GptModel } from "./model/model.js";
import { loadWeights } from "./model/weights_loader_server.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = process.env.MODEL_DIR || path.join(__dirname, "model_files");
const PORT = Number(process.env.PORT || 3000);
const MAX_BODY_BYTES = 128 * 1024;
const MAX_MESSAGE_CHARS = 4000;
const MAX_HISTORY_TURNS = 6;
const ABS_MAX_NEW_TOKENS = 256;

let model;
let tokenizer;
let config;
let SP; // özel token ID'leri
let modelReady = false;
let queue = Promise.resolve();

function sendJson(res, status, data) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(data));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("İstek gövdesi çok büyük."));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      } catch {
        reject(new Error("Geçersiz JSON."));
      }
    });
    req.on("error", reject);
  });
}

// ---------------- Sohbet şablonu (eğitim defteriyle BİREBİR aynı) ----------------
//   <|user|>\n SORU \n<|end|>\n<|assistant|>\n CEVAP \n<|end|>\n   (çok turlu: tekrar eder)
// Kullanıcı/asistan metni parseSpecial:false ile kodlanır: metne "<|assistant|>" yazmak
// özel token OLUŞTURMAZ (rol enjeksiyonu engellenir). Özel token'ları sadece biz ekleriz.
function concat(...parts) {
  const out = [];
  for (const p of parts) for (const id of p) out.push(id);
  return out;
}

function buildPromptIds(history, message, maxPromptTokens) {
  const enc = (t) => tokenizer.encode(t, { parseSpecial: false });
  const NL = enc("\n");
  const U = SP["<|user|>"];
  const A = SP["<|assistant|>"];
  const E = SP["<|end|>"];

  const userBlock = (q) => concat([U], NL, enc(q.trim()), NL, [E], NL);
  const asstOpen = () => concat([A], NL);
  const asstFull = (a) => concat([A], NL, enc(a.trim()), NL, [E], NL);

  let turns = history.slice(-MAX_HISTORY_TURNS);
  const build = () => {
    const parts = [];
    for (const [q, a] of turns) parts.push(userBlock(q), asstFull(a));
    parts.push(userBlock(message), asstOpen());
    return concat(...parts);
  };
  let ids = build();
  while (ids.length > maxPromptTokens && turns.length > 0) {
    turns = turns.slice(1); // en eski turu at
    ids = build();
  }
  if (ids.length > maxPromptTokens) {
    // tek mesaj bile sığmıyor: mesajın SONUNU tut
    const keep = Math.max(8, maxPromptTokens - (concat([U], NL, NL, [E], NL, [A], NL).length));
    const msgIds = enc(message.trim());
    ids = concat([U], NL, msgIds.slice(-keep), NL, [E], NL, [A], NL);
  }
  return ids;
}

// Gelen history'yi [[soru, cevap], ...] biçimine çevirir. Kabul edilen biçimler:
//   [{role:"user",content:"..."},{role:"assistant",content:"..."}, ...]  veya  [["soru","cevap"], ...]
function normalizeHistory(raw) {
  if (!Array.isArray(raw)) return [];
  const turns = [];
  let pendingUser = null;
  for (const item of raw) {
    if (Array.isArray(item) && typeof item[0] === "string" && typeof item[1] === "string") {
      turns.push([item[0].slice(0, MAX_MESSAGE_CHARS), item[1].slice(0, MAX_MESSAGE_CHARS)]);
    } else if (item && typeof item === "object") {
      const role = String(item.role || "").toLowerCase();
      const content = String(item.content ?? "").slice(0, MAX_MESSAGE_CHARS);
      if (!content.trim()) continue;
      if (role === "user") pendingUser = content;
      else if (role === "assistant" && pendingUser !== null) {
        turns.push([pendingUser, content]);
        pendingUser = null;
      }
    }
  }
  return turns.filter(([q, a]) => q.trim() && a.trim());
}

function cleanModelOutput(text) {
  let t = text;
  for (const s of Object.keys(SP)) t = t.split(s).join("");
  t = t.trim();
  return t || "(boş çıktı)";
}

function envNum(name, fallback) {
  const v = Number(process.env[name]);
  return Number.isFinite(v) && v > 0 ? v : fallback;
}

async function generate(message, history, overrides = {}) {
  // Aynı anda gelen CPU ağırlıklı istekleri sıraya alıyoruz.
  const job = queue.then(() => {
    const s = config.sampling || {};
    const maxNew = Math.min(
      ABS_MAX_NEW_TOKENS,
      Number(overrides.max_new_tokens) || envNum("MAX_NEW_TOKENS", s.max_new_tokens || 160)
    );
    const maxPrompt = config.block_size - maxNew;
    const promptIds = buildPromptIds(history, message, maxPrompt);
    const stopIds = new Set([SP["<|end|>"], SP["<|user|>"], SP["<|assistant|>"], SP["<|system|>"], SP["<|pad|>"]]);
    const outputIds = model.generate(promptIds, maxNew, {
      temperature: envNum("TEMPERATURE", s.temperature || 0.75),
      topK: envNum("TOP_K", s.top_k || 50),
      topP: envNum("TOP_P", s.top_p || 0.9),
      repetitionPenalty: envNum("REPETITION_PENALTY", s.repetition_penalty || 1.15),
      stopIds,
    });
    return cleanModelOutput(tokenizer.decode(outputIds, { skipSpecial: true }));
  });
  queue = job.catch(() => {});
  return job;
}

async function loadModel() {
  config = JSON.parse(await fs.readFile(path.join(MODEL_DIR, "config.json"), "utf8"));
  if (!config.special_tokens) {
    throw new Error("config.json'da special_tokens yok - bu sunucu v13 (özel token'lı) model ister.");
  }
  SP = config.special_tokens;
  tokenizer = BpeTokenizer.fromJson(
    JSON.parse(await fs.readFile(path.join(MODEL_DIR, "tokenizer_merges.json"), "utf8"))
  );
  tokenizer.setSpecialTokens(SP);
  const weights = await loadWeights(
    path.join(MODEL_DIR, "weights.bin"),
    path.join(MODEL_DIR, "weights_manifest.json"),
    { onProgress: (d, n) => console.log(`Ağırlıklar yükleniyor: ${d}/${n} tensör`) }
  );
  model = new GptModel(config, weights);
  modelReady = true;
}

const server = http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    });
    res.end();
    return;
  }

  if (req.method === "GET" && req.url === "/health") {
    sendJson(res, 200, {
      ok: true,
      modelLoaded: modelReady,
      model: "v13-1b",
      ...(modelReady ? { n_layer: config.n_layer, n_embd: config.n_embd, n_kv_head: config.n_kv_head || config.n_head } : {}),
    });
    return;
  }

  if (req.method === "POST" && req.url === "/api/chat") {
    if (!modelReady) {
      sendJson(res, 503, { error: "Model henüz hazır değil." });
      return;
    }
    try {
      const body = await readBody(req);
      // Geriye dönük uyum: eski istemciler "prompt" gönderiyordu; artık mesaj olarak ele alınır.
      const message = typeof body.message === "string" ? body.message
        : typeof body.prompt === "string" ? body.prompt : "";
      if (!message.trim()) {
        sendJson(res, 400, { error: "message gerekli." });
        return;
      }
      if (message.length > MAX_MESSAGE_CHARS) {
        sendJson(res, 400, { error: `Mesaj çok uzun (en fazla ${MAX_MESSAGE_CHARS} karakter).` });
        return;
      }
      const history = normalizeHistory(body.history);
      const response = await generate(message, history, { max_new_tokens: body.max_new_tokens });
      sendJson(res, 200, { response, model: "v13-1b" });
    } catch (err) {
      console.error("/api/chat:", err);
      sendJson(res, 500, { error: err.message || "Sunucu hatası." });
    }
    return;
  }

  sendJson(res, 404, { error: "Bulunamadı." });
});

loadModel()
  .then(() => {
    server.listen(PORT, "0.0.0.0", () => {
      console.log(`V13 AI API hazır: http://localhost:${PORT}`);
    });
  })
  .catch((err) => {
    console.error("Model yüklenemedi:", err);
    process.exit(1);
  });
