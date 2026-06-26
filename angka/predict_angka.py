import cv2
from ultralytics import YOLO

# Load model hasil training
model = YOLO(r"runs\classify\angka_bisindo-9\weights\best.pt")

# Buka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi!")
    exit()

print("✅ Kamera aktif! Tekan 'Q' untuk keluar.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Gagal membaca frame!")
        break

    # Flip horizontal
    frame = cv2.flip(frame, 1)

    # Prediksi
    results = model.predict(
        source=frame,
        verbose=False
    )

    probs = results[0].probs

    if probs is not None:
        top1_idx = probs.top1
        top1_conf = float(probs.top1conf)
        top1_name = results[0].names[top1_idx]

        if top1_conf >= 0.40:
            label = f"{top1_name} ({top1_conf*100:.1f}%)"
            color = (0, 255, 0)
        else:
            label = f"Tidak Yakin ({top1_conf*100:.1f}%)"
            color = (0, 165, 255)

        cv2.putText(
            frame,
            label,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    cv2.imshow("Deteksi Abjad BISINDO", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()