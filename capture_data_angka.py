import cv2
import os
import time

# =============================================
# KONFIGURASI
# =============================================
SAVE_DIR = r"D:\PRAKTIKUM COMPUTER VISION\isyarat_naz\kat"
AUTO_COUNT = 40      # jumlah foto saat auto-capture
AUTO_DELAY = 1       # jeda antar foto (detik)
# =============================================

# Kelas angka 1-9
CLASSES = [str(i) for i in range(1, 10)]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi!")
    exit()

current_class = '1'
status_msg = ""
status_color = (255, 255, 255)
auto_mode = False

def get_count(cls):
    folder = os.path.join(SAVE_DIR, cls)
    if not os.path.exists(folder):
        return 0

    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

def save_photo(frame, cls):
    folder = os.path.join(SAVE_DIR, cls)
    os.makedirs(folder, exist_ok=True)

    count = get_count(cls)
    fname = f"{cls}_custom_{count+1:04d}.jpg"
    fpath = os.path.join(folder, fname)

    cv2.imwrite(fpath, frame)
    return fname

print("=" * 60)
print("📸 SCRIPT PENGAMBIL DATA ANGKA BISINDO")
print("=" * 60)
print("TOMBOL:")
print("  1-9       → pilih angka")
print("  SPASI     → foto 1x")
print("  ENTER     → auto foto 40x")
print("  ESC       → keluar")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ==========================================
    # AUTO CAPTURE
    # ==========================================
    if auto_mode:
        status_msg = f"Auto capturing angka {current_class}..."
        status_color = (0, 165, 255)

        for i in range(AUTO_COUNT):
            ret2, f2 = cap.read()
            if not ret2:
                break

            f2 = cv2.flip(f2, 1)

            fname = save_photo(f2, current_class)

            prog = f"📸 {i+1}/{AUTO_COUNT} → {fname}"

            disp = f2.copy()
            cv2.rectangle(disp, (0, 0), (w, 50), (0, 100, 255), -1)

            cv2.putText(
                disp,
                prog,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Ambil Data Angka BISINDO", disp)
            cv2.waitKey(1)

            time.sleep(AUTO_DELAY)

        count = get_count(current_class)

        status_msg = (
            f"✅ Selesai! Total angka {current_class}: {count} foto"
        )
        status_color = (0, 220, 0)

        auto_mode = False

    # ==========================================
    # HEADER
    # ==========================================
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Angka aktif: {current_class}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    count = get_count(current_class)

    cv2.putText(
        frame,
        f"Foto tersimpan: {count}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )

    # ==========================================
    # FOOTER
    # ==========================================
    cv2.rectangle(frame, (0, h - 100), (w, h), (0, 0, 0), -1)

    cv2.putText(
        frame,
        status_msg,
        (10, h - 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2
    )

    cv2.putText(
        frame,
        "1-9=ganti angka | SPASI=foto | ENTER=auto | ESC=keluar",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1
    )

    # ==========================================
    # KOTAK POSISI TANGAN
    # ==========================================
    cx, cy = w // 2, h // 2
    size = 180

    cv2.rectangle(
        frame,
        (cx - size, cy - size),
        (cx + size, cy + size),
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Taruh tangan di sini",
        (cx - size + 5, cy - size - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    cv2.imshow("Ambil Data Angka BISINDO", frame)

    key = cv2.waitKey(1) & 0xFF

    # ESC
    if key == 27:
        break

    # Pilih angka 1-9
    elif key != 255:
        try:
            pressed = chr(key)

            if pressed in CLASSES:
                current_class = pressed
                status_msg = f"Pindah ke angka {current_class}"
                status_color = (255, 255, 0)

        except:
            pass

    # SPASI = foto 1x
    if key == ord(' '):
        fname = save_photo(frame, current_class)

        count = get_count(current_class)

        status_msg = (
            f"✅ Tersimpan: {fname} (total: {count})"
        )
        status_color = (0, 220, 0)

    # ENTER = auto foto
    elif key == 13:
        auto_mode = True

cap.release()
cv2.destroyAllWindows()

print("\n✅ Selesai pengambilan data!")
print(f"Data tersimpan di: {SAVE_DIR}")

print("\nRingkasan per angka:")
for cls in CLASSES:
    count = get_count(cls)
    print(f"  {cls}: {count} foto")