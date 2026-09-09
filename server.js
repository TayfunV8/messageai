import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs/promises";
import { BpeTokenizer } from "./model/tokenizer.js";
import { GptModel } from "./model/model.js";
import { loadWeights } from "./model/weights_loader_server.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "model_files");
const PORT = Number(process.env.PORT || 3000);
const MAX_BODY_BYTES = 64 * 1024;
const MAX_PROMPT_CHARS = 12000;

let model;
let tokenizer;
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

function cleanModelOutput(text) {
  const stopIdx = text.indexOf("<son>");
  if (stopIdx !== -1) text = text.slice(0, stopIdx);
  if (text.includes("Cevap:")) {
    text = text.split("Cevap:").slice(1).join("Cevap:");
  }
  return text.trim() || "(boş çıktı)";
}

async function generate(prompt) {
  // Aynı anda gelen CPU ağırlıklı istekleri sıraya alıyoruz.
  const job = queue.then(() => {
    const promptIds = tokenizer.encode(prompt);
    const outputIds = model.generate(promptIds, 130, {
      temperature: 0.45,
      topK: 40,
      repetitionPenalty: 1.3,
    });
    return cleanModelOutput(tokenizer.decode(outputIds));
  });
  queue = job.catch(() => {});
  return job;
}

async function loadModel() {
  const config = JSON.parse(
    await fs.readFile(path.join(MODEL_DIR, "config.json"), "utf8")
  );
  tokenizer = BpeTokenizer.fromJson(
    JSON.parse(await fs.readFile(path.join(MODEL_DIR, "tokenizer_merges.json"), "utf8"))
  );
  const weights = await loadWeights(
    path.join(MODEL_DIR, "weights.bin"),
    path.join(MODEL_DIR, "weights_manifest.json")
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
    sendJson(res, 200, { ok: true, modelLoaded: modelReady, model: "v9" });
    return;
  }

  if (req.method === "POST" && req.url === "/api/chat") {
    if (!modelReady) {
      sendJson(res, 503, { error: "Model henüz hazır değil." });
      return;
    }
    try {
      const body = await readBody(req);
      const prompt = typeof body.prompt === "string"
        ? body.prompt
        : typeof body.message === "string"
          ? `Soru: ${body.message}\nDüşünce:`
          : "";

      if (!prompt.trim()) {
        sendJson(res, 400, { error: "prompt veya message gerekli." });
        return;
      }
      if (prompt.length > MAX_PROMPT_CHARS) {
        sendJson(res, 400, { error: `Prompt çok uzun (en fazla ${MAX_PROMPT_CHARS} karakter).` });
        return;
      }

      const response = await generate(prompt);
      sendJson(res, 200, { response, model: "v9" });
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
      console.log(`V9 AI API hazır: http://localhost:${PORT}`);
    });
  })
  .catch((err) => {
    console.error("Model yüklenemedi:", err);
    process.exit(1);
  });
