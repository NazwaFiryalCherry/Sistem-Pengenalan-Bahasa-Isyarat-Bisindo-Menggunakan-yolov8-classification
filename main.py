import cv2
from ultralytics import YOLO

# ==========================
# LOAD SEMUA MODEL
# ==========================

print("Memuat model...")

model_angka = YOLO(
    r"angka\runs\classify\angka_bisindo-6\weights\best.pt"
)

model_abjad = YOLO(
    r"abjad\runs\classify\abjad_bisindo\weights\best.pt"
)

model_kata = YOLO(
    r"kata\runs\classify\kata_bisindo-7\weights\best.pt"
)

print("✅ Semua model berhasil dimuat!")

# ==========================
# BUKA KAMERA
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi!")
    exit()

print("✅ Kamera aktif!")
print("Tekan Q untuk keluar.")

# ==========================
# WEBCAM
# ==========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    hasil = []

    # =====================
    # ANGKA
    # =====================

    r1 = model_angka.predict(
        source=frame,
        verbose=False
    )[0]

    hasil.append({
        "label": r1.names[r1.probs.top1],
        "conf": float(r1.probs.top1conf)
    })

    # =====================
    # ABJAD
    # =====================

    r2 = model_abjad.predict(
        source=frame,
        verbose=False
    )[0]

    hasil.append({
        "label": r2.names[r2.probs.top1],
        "conf": float(r2.probs.top1conf)
    })

    # =====================
    # KATA
    # =====================

    r3 = model_kata.predict(
        source=frame,
        verbose=False
    )[0]

    hasil.append({
        "label": r3.names[r3.probs.top1],
        "conf": float(r3.probs.top1conf)
    })

    # =====================
    # AMBIL YANG TERTINGGI
    # =====================

    terbaik = max(
        hasil,
        key=lambda x: x["conf"]
    )

    label = terbaik["label"]
    conf = terbaik["conf"]

    if conf >= 0.40:
        text = f"{label} ({conf*100:.1f}%)"
        color = (0, 255, 0)
    else:
        text = "Tidak Yakin"
        color = (0, 165, 255)

    # =====================
    # TAMPILKAN
    # =====================

    cv2.rectangle(
        frame,
        (10, 10),
        (500, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "HASIL DETEKSI",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow(
        "Deteksi Bahasa Isyarat BISINDO",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()