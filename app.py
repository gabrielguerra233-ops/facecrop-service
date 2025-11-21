import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import mediapipe as mp
import numpy as np
import subprocess

app = FastAPI()
mp_face = mp.solutions.face_detection

TMP_DIR = "/tmp/facecrop"
os.makedirs(TMP_DIR, exist_ok=True)

def run_ffmpeg_crop(input_path, coords_by_frame, output_path, crop_w=1080, crop_h=1920, fps=30):
    """
    coords_by_frame: dict {frame_index: (center_x)}
    Faz um crop simples usando OpenCV frame-by-frame para simplicidade.
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))

    frame_idx = 0
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]

        # se há coord para este frame, use; se não, procure o último conhecido
        center_x = coords_by_frame.get(frame_idx, None)
        if center_x is None:
            # procura retroativamente
            keys = sorted(k for k in coords_by_frame.keys() if k <= frame_idx)
            center_x = coords_by_frame[keys[-1]] if keys else w // 2

        start_x = int(center_x - crop_w // 2)
        if start_x < 0:
            start_x = 0
        if start_x + crop_w > w:
            start_x = max(0, w - crop_w)

        crop = frame[:, start_x:start_x+crop_w]
        if crop.shape[1] != crop_w or crop.shape[0] != crop_h:
            try:
                crop = cv2.resize(crop, (crop_w, crop_h))
            except Exception:
                # cria um frame preto se algo falhar
                crop = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

        out.write(crop)
        frame_idx += 1

    cap.release()
    out.release()

@app.post("/crop")
async def crop_video(video: UploadFile = File(...)):
    # salva arquivo recebido
    uid = str(uuid.uuid4())[:8]
    in_path = os.path.join(TMP_DIR, f"input_{uid}.mp4")
    out_path = os.path.join(TMP_DIR, f"output_{uid}.mp4")

    with open(in_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # abre o vídeo e detecta rostos em cada frame (detecta centro X)
    coords = {}
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    frame_idx = 0
    success, frame = cap.read()
    while success:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        h, w = frame.shape[:2]
        if results.detections:
            # pega primeiro rosto
            det = results.detections[0]
            box = det.location_data.relative_bounding_box
            cx = int((box.xmin + box.width/2.0) * w)
            coords[frame_idx] = cx
        frame_idx += 1
        success, frame = cap.read()

    cap.release()
    detector.close()

    # Suaviza coords simples: média móvel
    keys = sorted(coords.keys())
    smooth_coords = {}
    last = None
    for k in keys:
        v = coords[k]
        if last is None:
            last = v
        else:
            last = int(last + (v - last) * 0.2)
        smooth_coords[k] = last

    # Gera output via ffmpeg/opencv (implementado em python)
    run_ffmpeg_crop(in_path, smooth_coords, out_path, crop_w=1080, crop_h=1920, fps=int(fps))

    # retorna o vídeo processado
    return FileResponse(out_path, media_type="video/mp4", filename=os.path.basename(out_path))
