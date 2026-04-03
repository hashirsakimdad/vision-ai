"""
VisionAI — FastAPI + YOLOv8 backend
Runs inside Google Colab and is exposed via ngrok tunnel.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn, time, uuid, os, json, base64, io
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="VisionAI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

Path("uploads").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

print(" Loading YOLOv8x model (downloads ~136MB on first run)...")
model = YOLO("yolov8x.pt")
print(" Model ready!")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "YOLOv8x", "gpu": True}


@app.get("/", response_class=HTMLResponse)
async def root():
    # Read and serve the frontend HTML
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(content="<h2>VisionAI API is running! POST to /detect</h2>", status_code=200)


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    confidence: float = 0.45,
    iou: float = 0.45,
):
    allowed = ("image/jpeg", "image/png", "image/webp", "image/jpg")
    if file.content_type not in allowed:
        raise HTTPException(400, "Only PNG/JPG/WEBP accepted")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(413, "Max 20MB")

    job_id = str(uuid.uuid4())[:8]
    img_path = f"uploads/{job_id}_{file.filename}"
    with open(img_path, "wb") as f:
        f.write(contents)

    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_w, img_h = pil_img.size

    start = time.perf_counter()
    results = model.predict(source=img_path, conf=confidence, iou=iou, verbose=False)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "id": len(detections),
            "class": results[0].names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox": {
                "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2),
                "width": round(x2 - x1), "height": round(y2 - y1),
                "nx": round(x1 / img_w, 4), "ny": round(y1 / img_h, 4),
                "nw": round((x2 - x1) / img_w, 4), "nh": round((y2 - y1) / img_h, 4),
            },
        })

    class_counts = {}
    for d in detections:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

    avg_conf = round(sum(d["confidence"] for d in detections) / len(detections), 4) if detections else 0
    b64 = base64.b64encode(contents).decode()
    mime = file.content_type or "image/jpeg"

    response = {
        "job_id": job_id,
        "model": "YOLOv8x",
        "inference_ms": elapsed_ms,
        "image": {
            "filename": file.filename, "width": img_w, "height": img_h,
            "data_url": f"data:{mime};base64,{b64}",
        },
        "summary": {
            "total_detections": len(detections),
            "unique_classes": len(class_counts),
            "avg_confidence": avg_conf,
            "class_counts": class_counts,
        },
        "config": {"confidence_threshold": confidence, "iou_threshold": iou},
        "detections": sorted(detections, key=lambda d: d["confidence"], reverse=True),
    }

    with open(f"results/{job_id}_report.json", "w") as f:
        json.dump(response, f, indent=2)

    os.remove(img_path)
    return JSONResponse(content=response)


print("main.py written!")
