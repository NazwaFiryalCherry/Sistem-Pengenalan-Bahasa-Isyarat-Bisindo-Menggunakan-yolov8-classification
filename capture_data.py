import cv2
import os
import time

# KONFIGURASI
SAVE_DIR    = r"D:\PRAKTIKUM COMPUTER VISION\isyarat\dataset\train"
AUTO_COUNT  = 40       # jumlah foto saat auto-capture
AUTO_DELAY  = 1     # jeda antar foto (detik)

CLASSES = [chr(i) for i in range(ord('A'), ord('Z')+1)]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi!")
    exit()

current_class = 'A'
status_msg    = ""
status_color  = (255, 255, 255)
auto_mode     = False

def get_count(cls):
    folder = os.path.join(SAVE_DIR, cls)
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg','.jpeg','.png'))])

def save_photo(frame, cls):
    folder = os.path.join(SAVE_DIR, cls)
    os.makedirs(folder, exist_ok=True)
    count  = get_count(cls)
    fname  = f"{cls}_custom_{count+1:04d}.jpg"
    fpath  = os.path.join(folder, fname)
    cv2.imwrite(fpath, frame)
    return fname

print("="*50)
print("📸 SCRIPT PENGAMBIL DATA BAHASA ISYARAT")
print("="*50)
print("TOMBOL:")
print("  A-Z       → pilih huruf")
print("  SPASI     → foto 1x")
print("  ENTER     → auto foto 30x")
print("  ESC       → keluar")
print("="*50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Auto capture mode
    if auto_mode:
        status_msg   = f"Auto capturing {current_class}..."
        status_color = (0, 165, 255)

        for i in range(AUTO_COUNT):
            ret2, f2 = cap.read()
            if not ret2:
                break
            f2    = cv2.flip(f2, 1)
            fname = save_photo(f2, current_class)

            prog = f"📸 {i+1}/{AUTO_COUNT} → {fname}"
            disp = f2.copy()
            cv2.rectangle(disp, (0, 0), (w, 50), (0, 100, 255), -1)
            cv2.putText(disp, prog, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Ambil Data Isyarat", disp)
            cv2.waitKey(1)
            time.sleep(AUTO_DELAY)

        count        = get_count(current_class)
        status_msg   = f"✅ Selesai! Total {current_class}: {count} foto"
        status_color = (0, 220, 0)
        auto_mode    = False

    # ── UI overlay ──────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Huruf aktif: {current_class}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    count = get_count(current_class)
    cv2.putText(frame, f"Foto tersimpan: {count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.rectangle(frame, (0, h-100), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, status_msg, (10, h-65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, "SPASI=foto 1x | ENTER=auto 30x | A-Z=ganti huruf | ESC=keluar",
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    # Kotak panduan posisi tangan
    cx, cy = w//2, h//2
    size   = 180
    cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (0,255,255), 2)
    cv2.putText(frame, "Taruh tangan di sini", (cx-size+5, cy-size-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.imshow("Ambil Data Isyarat", frame)

    key = cv2.waitKey(1) & 0xFF

    # ESC = keluar
    if key == 27:
        break

    # Pilih huruf A-Z
    elif key != 255 and chr(key).upper() in CLASSES:
        current_class = chr(key).upper()
        status_msg    = f"Pindah ke huruf {current_class}"
        status_color  = (255, 255, 0)

    # Foto 1x (spasi)
    elif key == ord(' '):
        fname        = save_photo(frame, current_class)
        count        = get_count(current_class)
        status_msg   = f"✅ Tersimpan: {fname} (total: {count})"
        status_color = (0, 220, 0)

    # Auto foto 30x (enter)
    elif key == 13:
        auto_mode = True

cap.release()
cv2.destroyAllWindows()
print("\n✅ Selesai pengambilan data!")
print(f"Data tersimpan di: {SAVE_DIR}")
print("\nRingkasan per huruf:")
for cls in CLASSES:
    count = get_count(cls)
    print(f"  {cls}: {count} foto")