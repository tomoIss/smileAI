import {
  FaceDetector,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// --- 設定定数 ---
const SMILE_MODEL_URL = "smile_detector.onnx";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

const MODEL_CACHE_NAME = "smileai-models-v1";
const CACHE_META_STORAGE_KEY = "smileai_model_cache_meta_v1";
const CACHE_TTL_MS = 12 * 60 * 60 * 1000; // 12時間

const MODEL_INPUT_SIZE = 128;
const CROP_PADDING_RATIO = 0.20;
const INFERENCE_INTERVAL_MS = 80;

// --- DOM要素 ---
const statusEl = document.getElementById("status");
const stageEl = document.getElementById("stage");
const videoEl = document.getElementById("video");
const overlayEl = document.getElementById("overlay");
const overlayCtx = overlayEl.getContext("2d");
const toggleCameraBtn = document.getElementById("toggleCameraBtn");
const switchFacingBtn = document.getElementById("switchFacingBtn");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValueEl = document.getElementById("thresholdValue");
const metricsTextEl = document.getElementById("metricsText");

// --- オフスクリーンCanvas & 推論バッファ ---
const cropCanvas = document.createElement("canvas");
cropCanvas.width = MODEL_INPUT_SIZE;
cropCanvas.height = MODEL_INPUT_SIZE;
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });
const inputBuffer = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);

// --- 状態変数 ---
let faceDetector = null;
let smileSession = null;
let mediaStream = null;
let facingMode = "user";
let isRunning = false;
let isProcessing = false;
let lastInferTime = 0;
let smileThreshold = parseFloat(thresholdSlider.value);

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

thresholdSlider.addEventListener("input", () => {
  smileThreshold = parseFloat(thresholdSlider.value);
  thresholdValueEl.textContent = smileThreshold.toFixed(2);
});

// --- キャッシュ管理ユーティリティ (TTL: 12時間) ---
function readCacheMeta() {
  try {
    return JSON.parse(localStorage.getItem(CACHE_META_STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function writeCacheMeta(meta) {
  try {
    localStorage.setItem(CACHE_META_STORAGE_KEY, JSON.stringify(meta));
  } catch (err) {
    console.warn("Cache metadata write failed:", err);
  }
}

/**
 * モデルファイルを12時間TTL付きCache APIから取得する
 * @returns {Promise<{ bytes: Uint8Array, fromCache: boolean }>}
 */
async function fetchModelWithCache(url, ttlMs = CACHE_TTL_MS) {
  const absUrl = new URL(url, window.location.href).href;
  const now = Date.now();
  const canUseCacheApi = typeof caches !== "undefined";

  if (canUseCacheApi) {
    try {
      const cache = await caches.open(MODEL_CACHE_NAME);
      const meta = readCacheMeta();
      const cachedAt = meta[absUrl];

      if (typeof cachedAt === "number" && now - cachedAt < ttlMs) {
        const cachedResp = await cache.match(absUrl);
        if (cachedResp) {
          const buf = await cachedResp.arrayBuffer();
          return { bytes: new Uint8Array(buf), fromCache: true };
        }
      }
    } catch (err) {
      console.warn("Cache read failed, falling back to network:", err);
    }
  }

  // キャッシュなし、または12時間超過時はネットワークから再取得
  const response = await fetch(absUrl, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`モデル取得失敗 (${response.status}): ${absUrl}`);
  }

  const buffer = await response.arrayBuffer();

  if (canUseCacheApi) {
    try {
      const cache = await caches.open(MODEL_CACHE_NAME);
      const cacheResp = new Response(buffer.slice(0), {
        status: 200,
        headers: { "Content-Type": "application/octet-stream" },
      });
      await cache.put(absUrl, cacheResp);

      const meta = readCacheMeta();
      meta[absUrl] = now;
      writeCacheMeta(meta);
    } catch (err) {
      console.warn("Cache write failed:", err);
    }
  }

  return { bytes: new Uint8Array(buffer), fromCache: false };
}

// --- 1. モデル初期化 ---
async function initModels() {
  try {
    if (!window.isSecureContext) {
      throw new Error("カメラ利用には HTTPS 環境（GitHub Pages 等）が必要です。");
    }

    setStatus("顔検出モデル (MediaPipe) を読み込み中...");
    const [vision, faceModel] = await Promise.all([
      FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
      ),
      fetchModelWithCache(FACE_MODEL_URL),
    ]);

    faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: faceModel.bytes,
        delegate: "CPU",
      },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.5,
    });

    setStatus("笑顔判定モデル (ONNX) を読み込み中...");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    const smileModel = await fetchModelWithCache(SMILE_MODEL_URL);
    smileSession = await ort.InferenceSession.create(smileModel.bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    const sourceTag =
      faceModel.fromCache && smileModel.fromCache
        ? "キャッシュ済 (12h有効)"
        : "ネットワークから新規保存 (12hキャッシュ)";
    setStatus(`準備完了 [${sourceTag}]。「カメラ開始」を押してください。`);
    toggleCameraBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus(`初期化エラー: ${err.message}`, true);
  }
}

// --- 2. カメラ制御 ---
async function startCamera() {
  stopCamera();
  try {
    setStatus("カメラを起動中...");
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode,
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    });
    videoEl.srcObject = mediaStream;
    await videoEl.play();

    stageEl.classList.toggle("mirrored", facingMode === "user");
    overlayEl.width = videoEl.videoWidth;
    overlayEl.height = videoEl.videoHeight;

    isRunning = true;
    toggleCameraBtn.textContent = "カメラ停止";
    switchFacingBtn.disabled = false;
    setStatus("解析中...");
    requestAnimationFrame(processLoop);
  } catch (err) {
    console.error(err);
    setStatus(`カメラ起動エラー: ${err.message}`, true);
  }
}

function stopCamera() {
  isRunning = false;
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  videoEl.srcObject = null;
  overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);
  toggleCameraBtn.textContent = "カメラ開始";
  switchFacingBtn.disabled = true;
}

toggleCameraBtn.addEventListener("click", () => {
  if (isRunning) {
    stopCamera();
    setStatus("停止中");
  } else {
    startCamera();
  }
});

switchFacingBtn.addEventListener("click", () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  if (isRunning) startCamera();
});

// --- 3. 前処理 ---
function prepareFaceTensor(bbox) {
  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;

  const cx = bbox.originX + bbox.width / 2;
  const cy = bbox.originY + bbox.height / 2;
  const side = Math.max(bbox.width, bbox.height) * (1 + CROP_PADDING_RATIO * 2);

  const sx = Math.max(0, cx - side / 2);
  const sy = Math.max(0, cy - side / 2);
  const sWidth = Math.min(side, vw - sx);
  const sHeight = Math.min(side, vh - sy);

  cropCtx.clearRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  cropCtx.drawImage(
    videoEl,
    sx, sy, sWidth, sHeight,
    0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE
  );

  const { data } = cropCtx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const area = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

  for (let i = 0; i < area; i++) {
    const rgbaIdx = i * 4;
    inputBuffer[i] = data[rgbaIdx] / 255.0;
    inputBuffer[area + i] = data[rgbaIdx + 1] / 255.0;
    inputBuffer[2 * area + i] = data[rgbaIdx + 2] / 255.0;
  }

  return new ort.Tensor("float32", inputBuffer, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
}

// --- 4. 描画ヘルパー ---
function drawDetectionBox(bbox, smileProb, isSmile) {
  const color = isSmile ? "#22c55e" : "#f59e0b";
  const label = `${isSmile ? "SMILE" : "NEUTRAL"} ${(smileProb * 100).toFixed(0)}%`;

  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = 3;
  overlayCtx.strokeRect(bbox.originX, bbox.originY, bbox.width, bbox.height);

  overlayCtx.font = "bold 16px -apple-system, sans-serif";
  const textWidth = overlayCtx.measureText(label).width;
  const boxX = bbox.originX;
  const boxY = Math.max(24, bbox.originY - 6);

  overlayCtx.save();
  if (facingMode === "user") {
    overlayCtx.translate(boxX + textWidth / 2 + 6, boxY);
    overlayCtx.scale(-1, 1);
    overlayCtx.fillStyle = color;
    overlayCtx.fillRect(-textWidth / 2 - 6, -20, textWidth + 12, 24);
    overlayCtx.fillStyle = "#0f172a";
    overlayCtx.fillText(label, -textWidth / 2, -3);
  } else {
    overlayCtx.fillStyle = color;
    overlayCtx.fillRect(boxX, boxY - 20, textWidth + 12, 24);
    overlayCtx.fillStyle = "#0f172a";
    overlayCtx.fillText(label, boxX + 6, boxY - 3);
  }
  overlayCtx.restore();
}

// --- 5. メイン推論ループ ---
async function processLoop(now) {
  if (!isRunning) return;
  requestAnimationFrame(processLoop);

  if (isProcessing || videoEl.readyState < 2) return;
  if (now - lastInferTime < INFERENCE_INTERVAL_MS) return;

  isProcessing = true;
  const cycleStart = performance.now();

  try {
    const detections = faceDetector.detectForVideo(videoEl, now).detections;
    overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);

    for (const det of detections) {
      const bbox = det.boundingBox;
      if (!bbox || bbox.width < 20 || bbox.height < 20) continue;

      const inputTensor = prepareFaceTensor(bbox);
      let results = null;
      try {
        results = await smileSession.run({ input: inputTensor });
        const prob = results.smile_prob.data[0];
        const isSmile = prob >= smileThreshold;
        drawDetectionBox(bbox, prob, isSmile);
      } finally {
        inputTensor.dispose?.();
        results?.smile_prob?.dispose?.();
      }
    }

    const elapsedMs = performance.now() - cycleStart;
    const fps = lastInferTime > 0 ? 1000 / (now - lastInferTime) : 0;
    lastInferTime = now;
    metricsTextEl.textContent = `FPS: ${fps.toFixed(1)} | 処理時間: ${elapsedMs.toFixed(1)} ms`;
  } catch (err) {
    console.error("Inference loop error:", err);
    setStatus(`推論エラー: ${err.message}`, true);
  } finally {
    isProcessing = false;
  }
}

initModels();
