from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import cv2
from ultralytics import YOLO

app = FastAPI()

# =============================================
# LOAD 3 MODEL
# =============================================
model_angka = YOLO(r"angka\runs\classify\angka_bisindo-9\weights\best.pt")
model_abjad = YOLO(r"abjad\runs\classify\abjad_bisindo\weights\best.pt")
model_kata  = YOLO(r"kata\runs\classify\kata_bisindo-8\weights\best.pt")
# =============================================

CONF_THRESHOLD = 0.60
cap        = None
mode_aktif = "abjad"  # default mode

# ── Tampilkan akurasi model di terminal saat startup ──────────────────────────
def tampil_akurasi():
    print("\n" + "="*55)
    print("  🤟 BISINDO Recognition System - Model Info")
    print("="*55)
    models_info = {
        "Angka": model_angka,
        "Abjad": model_abjad,
        "Kata" : model_kata,
    }
    for nama, mdl in models_info.items():
        kelas  = list(mdl.names.values())
        jumlah = len(kelas)
        print(f"  📦 Model {nama}")
        print(f"     Jumlah kelas : {jumlah}")
        print(f"     Kelas        : {', '.join(kelas)}")
        print()
    print("="*55)
    print("  ✅ Semua model berhasil dimuat!")
    print("="*55 + "\n")

tampil_akurasi()

# ── Helper ────────────────────────────────────────────────────────────────────
def get_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap

def get_model():
    if mode_aktif == "angka": return model_angka
    if mode_aktif == "kata":  return model_kata
    return model_abjad

# ── Endpoints ─────────────────────────────────────────────────────────────────
def generate_frames():
    camera = get_camera()
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/set_mode/{mode}")
def set_mode(mode: str):
    global mode_aktif
    if mode in ["angka", "abjad", "kata"]:
        mode_aktif = mode
        print(f"  🔄 Mode berganti ke: {mode.upper()}")
        return {"mode": mode_aktif}
    return JSONResponse({"error": "Mode tidak valid"}, status_code=400)

@app.get("/predict")
def predict():
    global mode_aktif
    camera = get_camera()
    success, frame = camera.read()
    if not success:
        return JSONResponse({"error": "Kamera tidak tersedia"})
    frame  = cv2.flip(frame, 1)
    model  = get_model()
    r      = model(frame, verbose=False)[0]
    label  = r.names[r.probs.top1]
    conf   = round(float(r.probs.top1conf) * 100, 1)
    detected = conf >= CONF_THRESHOLD * 100

    # Log ke terminal
    status = "✅" if detected else "❌"
    print(f"  {status} [{mode_aktif.upper():5}] {label:15} {conf:.1f}%")

    return {
        "label":    label,
        "confidence": conf,
        "detected": detected,
        "mode":     mode_aktif,
    }

@app.get("/mode")
def get_mode():
    return {"mode": mode_aktif}

HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BISINDO Recognition</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#F0F9FF;--surface:#ffffff;--border:#BAE6FD;
    --text:#0C4A6E;--muted:#38BDF8;
    --blue:#0EA5E9;--blue-dark:#0284C7;--blue-bg:#E0F2FE;--blue-border:#7DD3FC;
    --green:#059669;--green-bg:#ECFDF5;--green-border:#6EE7B7;
    --orange:#EA580C;--orange-bg:#FFF7ED;--orange-border:#FED7AA;
    --shadow:0 2px 8px rgba(14,165,233,.08),0 8px 24px rgba(14,165,233,.06);
  }
  body{font-family:'Plus Jakarta Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}

  header{background:var(--surface);border-bottom:1px solid var(--border);padding:0 32px;height:68px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;box-shadow:0 1px 8px rgba(14,165,233,.08);}
  .logo{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.15rem;color:var(--text);}
  .logo-icon{width:40px;height:40px;background:linear-gradient(135deg,var(--blue),var(--blue-dark));border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;}
  .badge{background:var(--blue-bg);color:var(--blue-dark);border:1px solid var(--blue-border);font-size:.72rem;font-weight:700;padding:3px 10px;border-radius:99px;letter-spacing:.3px;}
  .status-dot{display:flex;align-items:center;gap:6px;font-size:.82rem;color:#64748b;font-weight:500;}
  .dot{width:8px;height:8px;border-radius:50%;background:#cbd5e1;transition:background .3s;}
  .dot.active{background:#10b981;box-shadow:0 0 0 3px #d1fae5;}

  main{max-width:1180px;margin:0 auto;padding:28px 20px;display:grid;grid-template-columns:1fr 320px;gap:22px;align-items:start;}

  .card{background:var(--surface);border:1px solid var(--border);border-radius:18px;overflow:hidden;box-shadow:var(--shadow);}
  .card-header{padding:14px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;background:linear-gradient(135deg,#F0F9FF,#ffffff);}
  .card-title{font-size:.8rem;font-weight:700;color:var(--blue-dark);text-transform:uppercase;letter-spacing:.7px;}
  .card-body{padding:18px;}

  /* Mode selector */
  .mode-wrap{display:flex;gap:8px;margin-bottom:16px;}
  .mode-btn{
    flex:1;padding:9px 0;border:2px solid var(--border);border-radius:10px;
    font-family:inherit;font-size:.85rem;font-weight:700;cursor:pointer;
    background:var(--bg);color:#64748b;transition:all .15s;
  }
  .mode-btn:hover{border-color:var(--blue);color:var(--blue);}
  .mode-btn.active{background:var(--blue);border-color:var(--blue);color:white;}
  .mode-btn.active-angka{background:#0369A1;border-color:#0369A1;color:white;}
  .mode-btn.active-kata{background:#065F46;border-color:#065F46;color:white;}

  .camera-wrap{position:relative;background:#0C4A6E;border-radius:14px;overflow:hidden;aspect-ratio:4/3;}
  #cam-feed{width:100%;height:100%;object-fit:cover;display:block;opacity:0;transition:opacity .4s;}
  #cam-feed.loaded{opacity:1;}
  .cam-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:12px;background:#0C4A6E;transition:opacity .4s;}
  .cam-overlay.hidden{opacity:0;pointer-events:none;}
  .cam-overlay span{color:#7DD3FC;font-size:.9rem;}
  .cam-corner{position:absolute;width:36px;height:36px;border-color:rgba(125,211,252,.6);border-style:solid;}
  .cam-corner.tl{top:14px;left:14px;border-width:2px 0 0 2px;border-radius:4px 0 0 0;}
  .cam-corner.tr{top:14px;right:14px;border-width:2px 2px 0 0;border-radius:0 4px 0 0;}
  .cam-corner.bl{bottom:14px;left:14px;border-width:0 0 2px 2px;border-radius:0 0 0 4px;}
  .cam-corner.br{bottom:14px;right:14px;border-width:0 2px 2px 0;border-radius:0 0 4px 0;}

  /* Mode badge di kamera */
  .mode-badge{
    position:absolute;top:12px;left:12px;z-index:10;
    background:rgba(0,0,0,.55);backdrop-filter:blur(4px);
    color:white;font-size:.75rem;font-weight:700;
    padding:4px 10px;border-radius:99px;letter-spacing:.5px;
  }

  .controls{display:flex;gap:10px;margin-top:14px;}
  .btn{flex:1;padding:11px 0;border:none;border-radius:10px;font-family:inherit;font-size:.88rem;font-weight:700;cursor:pointer;transition:all .15s;display:flex;align-items:center;justify-content:center;gap:6px;}
  .btn-primary{background:linear-gradient(135deg,var(--blue),var(--blue-dark));color:white;}
  .btn-primary:hover{opacity:.9;transform:translateY(-1px);}
  .btn-danger{background:#FEF2F2;color:#DC2626;border:1px solid #FECACA;}
  .btn-danger:hover{background:#FEE2E2;}
  .btn:disabled{opacity:.4;cursor:not-allowed;transform:none!important;}

  .result-big{
    text-align:center;padding:20px 16px;border-radius:14px;
    background:var(--blue-bg);border:2px solid var(--blue-border);
    margin-bottom:14px;transition:all .2s;
    min-height:120px;display:flex;flex-direction:column;
    align-items:center;justify-content:center;
  }
  .result-big.detected{background:var(--green-bg);border-color:var(--green-border);}
  .result-big.empty{background:var(--blue-bg);border-color:var(--blue-border);}
  .result-char{font-weight:800;line-height:1.15;color:var(--text);font-family:'JetBrains Mono',monospace;text-align:center;word-break:break-word;transition:font-size .15s;}
  .result-empty{font-size:.88rem;color:#94A3B8;font-weight:500;display:flex;flex-direction:column;align-items:center;gap:6px;}
  .result-conf{font-size:.8rem;color:#64748b;margin-top:6px;font-weight:500;}

  .conf-bar-wrap{margin-bottom:14px;}
  .conf-label{display:flex;justify-content:space-between;font-size:.78rem;font-weight:700;color:#64748b;margin-bottom:5px;}
  .conf-bar-bg{height:8px;background:var(--border);border-radius:99px;overflow:hidden;}
  .conf-bar-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--blue),#38BDF8);transition:width .2s ease;width:0%;}

  /* Mode info */
  .mode-info{padding:10px 14px;border-radius:10px;background:var(--blue-bg);border:1px solid var(--blue-border);font-size:.82rem;color:var(--blue-dark);font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:8px;}

  .riwayat-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
  .riwayat-label{font-size:.78rem;font-weight:700;color:var(--blue-dark);text-transform:uppercase;letter-spacing:.6px;}
  .btn-clear{background:none;border:1px solid var(--border);border-radius:6px;padding:4px 10px;font-size:.75rem;font-weight:600;color:#64748b;cursor:pointer;transition:all .15s;font-family:inherit;}
  .btn-clear:hover{background:#FEF2F2;border-color:#FECACA;color:#DC2626;}
  .riwayat-text{min-height:52px;padding:12px 14px;background:var(--bg);border:1px solid var(--border);border-radius:10px;font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:600;letter-spacing:1.5px;color:var(--text);word-break:break-all;line-height:1.7;}
  .riwayat-empty-text{color:#94A3B8;font-family:inherit;font-size:.82rem;font-weight:400;letter-spacing:0;}
  .btn-hapus{margin-top:8px;background:none;border:1px solid var(--border);border-radius:6px;padding:5px 12px;font-size:.78rem;font-weight:600;color:#64748b;cursor:pointer;transition:all .15s;font-family:inherit;}
  .btn-hapus:hover{background:var(--blue-bg);border-color:var(--blue-border);color:var(--blue-dark);}

  footer{text-align:center;padding:20px;font-size:.75rem;color:#94A3B8;border-top:1px solid var(--border);margin-top:10px;}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">📸</div>
    BISINDO Recognition
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <span class="badge">YOLOv8 · 3 Model</span>
    <div class="status-dot">
      <div class="dot" id="status-dot"></div>
      <span id="status-text">Kamera mati</span>
    </div>
  </div>
</header>

<main>
  <div>
    <div class="card">
      <div class="card-header">
        <span>📷</span>
        <span class="card-title">Live Camera</span>
      </div>
      <div class="card-body">

        <!-- Pilih Mode -->
        <div class="mode-wrap">
          <button class="mode-btn active" id="btn-mode-abjad" onclick="setMode('abjad')">🔤 Abjad</button>
          <button class="mode-btn" id="btn-mode-angka" onclick="setMode('angka')">🔢 Angka</button>
          <button class="mode-btn" id="btn-mode-kata"  onclick="setMode('kata')">💬 Kata</button>
        </div>

        <div class="camera-wrap">
          <img id="cam-feed" src="" alt="Camera">
          <div class="cam-overlay" id="cam-overlay">
            <span style="font-size:2.5rem">📷</span>
            <span>Klik Mulai untuk mengaktifkan kamera</span>
          </div>
          <div class="cam-corner tl"></div>
          <div class="cam-corner tr"></div>
          <div class="cam-corner bl"></div>
          <div class="cam-corner br"></div>
          <div class="mode-badge" id="mode-badge">ABJAD</div>
        </div>

        <div class="controls">
          <button class="btn btn-primary" id="btn-start" onclick="startCamera()">▶ Mulai Kamera</button>
          <button class="btn btn-danger"  id="btn-stop"  onclick="stopCamera()" disabled>⏹ Stop</button>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:18px">
      <div class="card-header">
        <span>📝</span>
        <span class="card-title">Riwayat Deteksi</span>
      </div>
      <div class="card-body">
        <div class="riwayat-header">
          <span class="riwayat-label">Terkumpul</span>
          <button class="btn-clear" onclick="clearRiwayat()">🗑 Hapus semua</button>
        </div>
        <div class="riwayat-text" id="riwayat-text">
          <span class="riwayat-empty-text">Belum ada deteksi...</span>
        </div>
        <button class="btn-hapus" onclick="hapusSatu()">⌫ Hapus 1 karakter</button>
      </div>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:18px">
    <div class="card">
      <div class="card-header">
        <span>✨</span>
        <span class="card-title">Hasil Deteksi</span>
      </div>
      <div class="card-body">
        <div class="mode-info" id="mode-info">
          <span>🔤</span>
          <span id="mode-info-text">Mode: Abjad (A-Z)</span>
        </div>
        <div class="result-big empty" id="result-box">
          <div class="result-empty" id="result-empty">
            <span style="font-size:1.8rem">🤷🏻‍♀️🤷🏻🤷🏻‍♂️</span>
            <span>Belum ada isyarat</span>
          </div>
          <div class="result-char" id="result-char" style="display:none"></div>
          <div class="result-conf" id="result-conf" style="display:none"></div>
        </div>
        <div class="conf-bar-wrap">
          <div class="conf-label">
            <span>Confidence</span>
            <span id="conf-pct">0%</span>
          </div>
          <div class="conf-bar-bg">
            <div class="conf-bar-fill" id="conf-bar"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</main>

<footer>BISINDO Recognition System · YOLOv8 Classification · Praktikum Computer Vision</footer>

<script>
  let interval  = null;
  let isRunning = false;
  let riwayat   = [];
  let lastChar  = '';
  let lastTime  = 0;
  let modeAktif = 'abjad';
  const CONF_MIN = 60;
  const DELAY_MS = 2000;

  const modeConfig = {
    abjad: { label:'Abjad (A-Z)', icon:'🔤', color:'#0284C7' },
    angka: { label:'Angka (1-10)', icon:'🔢', color:'#0369A1' },
    kata:  { label:'Kata BISINDO', icon:'💬', color:'#065F46' },
  };

  function getFontSize(text) {
    const len = text.length;
    if (len <= 1)       return '4.5rem';
    else if (len <= 3)  return '4rem';
    else if (len <= 6)  return '3rem';
    else if (len <= 10) return '2rem';
    else if (len <= 15) return '1.4rem';
    else                return '1.1rem';
  }

  async function setMode(mode) {
    modeAktif = mode;
    await fetch(`/set_mode/${mode}`);

    // Update tombol aktif
    ['abjad','angka','kata'].forEach(m => {
      const btn = document.getElementById(`btn-mode-${m}`);
      btn.className = 'mode-btn' + (m === mode ? ' active' : '');
    });

    // Update badge kamera
    document.getElementById('mode-badge').textContent = mode.toUpperCase();

    // Update info panel
    const cfg = modeConfig[mode];
    document.getElementById('mode-info').innerHTML = `<span>${cfg.icon}</span><span id="mode-info-text">Mode: ${cfg.label}</span>`;

    // Reset hasil
    resetResult();
    lastChar = '';
  }

  function startCamera() {
    const feed    = document.getElementById('cam-feed');
    const overlay = document.getElementById('cam-overlay');
    feed.src = '/video?' + Date.now();
    feed.onload = () => feed.classList.add('loaded');
    overlay.classList.add('hidden');
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled  = false;
    document.getElementById('status-dot').classList.add('active');
    document.getElementById('status-text').textContent = 'Kamera aktif';
    isRunning = true;
    interval  = setInterval(fetchPredict, 400);
  }

  function stopCamera() {
    const feed    = document.getElementById('cam-feed');
    const overlay = document.getElementById('cam-overlay');
    feed.src = '';
    feed.classList.remove('loaded');
    overlay.classList.remove('hidden');
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled  = true;
    document.getElementById('status-dot').classList.remove('active');
    document.getElementById('status-text').textContent = 'Kamera mati';
    isRunning = false;
    clearInterval(interval);
    resetResult();
  }

  async function fetchPredict() {
    if (!isRunning) return;
    try {
      const res  = await fetch('/predict');
      const data = await res.json();
      updateUI(data);
    } catch(e) {}
  }

  function updateUI(data) {
    const box     = document.getElementById('result-box');
    const charEl  = document.getElementById('result-char');
    const confEl  = document.getElementById('result-conf');
    const emptyEl = document.getElementById('result-empty');
    const bar     = document.getElementById('conf-bar');

    document.getElementById('conf-bar').style.width = data.confidence + '%';
    document.getElementById('conf-pct').textContent = data.confidence + '%';

    if (!data.detected) {
      box.className         = 'result-big empty';
      emptyEl.style.display = 'flex';
      charEl.style.display  = 'none';
      confEl.style.display  = 'none';
      bar.style.background  = 'linear-gradient(90deg,#CBD5E1,#94A3B8)';
    } else {
      box.className         = 'result-big detected';
      emptyEl.style.display = 'none';
      charEl.style.display  = 'block';
      confEl.style.display  = 'block';
      charEl.textContent    = data.label;
      charEl.style.fontSize = getFontSize(data.label);
      confEl.textContent    = `Confidence: ${data.confidence}%`;
      bar.style.background  = 'linear-gradient(90deg,#059669,#34D399)';

      const now = Date.now();
      if (data.label !== lastChar || (now - lastTime) > DELAY_MS) {
        riwayat.push(data.label);
        lastChar = data.label;
        lastTime = now;
        renderRiwayat();
      }
    }
  }

  function renderRiwayat() {
    const el = document.getElementById('riwayat-text');
    el.innerHTML = riwayat.length === 0
      ? '<span class="riwayat-empty-text">Belum ada deteksi...</span>'
      : riwayat.join(' ');
  }

  function clearRiwayat() { riwayat = []; lastChar = ''; renderRiwayat(); }
  function hapusSatu()    { if (riwayat.length > 0) { riwayat.pop(); renderRiwayat(); } }

  function resetResult() {
    const box     = document.getElementById('result-box');
    const charEl  = document.getElementById('result-char');
    const confEl  = document.getElementById('result-conf');
    const emptyEl = document.getElementById('result-empty');
    box.className         = 'result-big empty';
    emptyEl.style.display = 'flex';
    charEl.style.display  = 'none';
    confEl.style.display  = 'none';
    document.getElementById('conf-bar').style.width = '0%';
    document.getElementById('conf-pct').textContent = '0%';
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)