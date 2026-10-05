from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, os, onnxruntime as ort

cv2.setNumThreads(1)
APP_VERSION = "1.3.0"

app = FastAPI(title="LightFace MobileFaceNet API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lightface-hybrid-ai.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL = os.path.join(BASE, "models", "w600k_mbf.onnx")
DETECT_MODEL = os.path.join(BASE, "models", "face_detection_yunet_2023mar.onnx")

REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

try:
    DEFAULT_THRESHOLD = float(os.getenv("VERIFY_THRESHOLD", "0.50"))
except ValueError:
    DEFAULT_THRESHOLD = 0.50

if not os.path.exists(DETECT_MODEL):
    raise RuntimeError(f"YuNet detector not found: {DETECT_MODEL}")

detector = cv2.FaceDetectorYN.create(
    DETECT_MODEL, "", (320, 320), 0.5, 0.3, 50
)

if detector is None:
    raise RuntimeError("Unable to initialize YuNet face detector.")

recognition_session = None
recognition_input_name = None
recognition_output_name = None
recognition_input_shape = None


def load_recognition_model():
    global recognition_session, recognition_input_name
    global recognition_output_name, recognition_input_shape

    if recognition_session is not None:
        return

    if not os.path.exists(FACE_MODEL):
        raise RuntimeError(f"Recognition model not found: {FACE_MODEL}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = True
    opts.enable_mem_pattern = True

    recognition_session = ort.InferenceSession(
        FACE_MODEL,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    inputs = recognition_session.get_inputs()
    outputs = recognition_session.get_outputs()

    if not inputs or not outputs:
        raise RuntimeError("Invalid MobileFaceNet ONNX model.")

    recognition_input_name = inputs[0].name
    recognition_output_name = outputs[0].name
    recognition_input_shape = inputs[0].shape


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "LightFace MobileFaceNet API",
        "version": APP_VERSION,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "LightFace MobileFaceNet API",
        "version": APP_VERSION,
        "models": {
            "detection": os.path.exists(DETECT_MODEL),
            "recognition_file": os.path.exists(FACE_MODEL),
            "recognition_loaded": recognition_session is not None,
        },
        "recognition": {
            "model": "w600k_mbf.onnx",
            "embedding_dimension": 512,
            "runtime": "onnxruntime-cpu",
            "input_shape": recognition_input_shape,
        },
        "alignment": {
            "enabled": True,
            "method": "5-point_similarity",
            "output_size": [112, 112],
        },
        "preprocessing": {
            "color": "RGB",
            "normalization": "(pixel - 127.5) / 127.5",
            "range": [-1.0, 1.0],
            "layout": "NCHW",
        },
        "verification": {
            "threshold": DEFAULT_THRESHOLD,
            "threshold_source": "VERIFY_THRESHOLD environment variable",
        },
    }


async def decode_upload(file: UploadFile):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty image upload.")

    image = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(400, "Invalid image.")

    return image


def detect_faces(image):
    h, w = image.shape[:2]
    scale = min(1.0, 1280 / max(h, w))

    if scale < 1:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        work = cv2.resize(
            image, (nw, nh), interpolation=cv2.INTER_AREA
        )
    else:
        work, nw, nh = image, w, h

    detector.setInputSize((nw, nh))
    _, faces = detector.detect(work)

    if faces is None:
        return []

    sx, sy = w / nw, h / nh
    results = []

    for f in faces:
        x, y, bw, bh = map(float, f[:4])

        lm = [
            [float(f[4]) * sx, float(f[5]) * sy],
            [float(f[6]) * sx, float(f[7]) * sy],
            [float(f[8]) * sx, float(f[9]) * sy],
            [float(f[10]) * sx, float(f[11]) * sy],
            [float(f[12]) * sx, float(f[13]) * sy],
        ]

        results.append({
            "box": [
                max(0, int(x * sx)),
                max(0, int(y * sy)),
                min(w, int((x + bw) * sx)),
                min(h, int((y + bh) * sy)),
            ],
            "confidence": round(float(f[-1]), 4),
            "landmarks": [
                [round(p[0], 2), round(p[1], 2)] for p in lm
            ],
        })

    return results


def align_face(image, landmarks):
    src = np.asarray(landmarks, dtype=np.float32)

    if src.shape != (5, 2):
        raise ValueError("Exactly five facial landmarks are required.")

    transform, _ = cv2.estimateAffinePartial2D(
        src,
        REFERENCE_POINTS,
        method=cv2.LMEDS,
    )

    if transform is None:
        raise ValueError("Unable to calculate face alignment transform.")

    return cv2.warpAffine(
        image,
        transform,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def preprocess_for_mobilefacenet(aligned_bgr):
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    rgb = (rgb.astype(np.float32) - 127.5) / 127.5
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor, dtype=np.float32)


def generate_embedding(aligned_bgr):
    load_recognition_model()

    result = recognition_session.run(
        [recognition_output_name],
        {recognition_input_name: preprocess_for_mobilefacenet(aligned_bgr)},
    )

    embedding = np.asarray(result[0][0], dtype=np.float32)
    norm = np.linalg.norm(embedding)

    if norm <= 1e-12:
        raise RuntimeError("MobileFaceNet returned a zero embedding.")

    return embedding / norm


def get_best_aligned_face(image):
    faces = detect_faces(image)

    if not faces:
        raise HTTPException(404, "No face detected.")

    face = max(faces, key=lambda x: x["confidence"])

    try:
        aligned = align_face(image, face["landmarks"])
    except Exception as e:
        raise HTTPException(
            422,
            f"Face alignment failed: {e}",
        )

    return aligned, face, faces


@app.post("/align")
async def align(file: UploadFile = File(...)):
    image = await decode_upload(file)
    aligned, face, _ = get_best_aligned_face(image)

    ok, encoded = cv2.imencode(
        ".jpg",
        aligned,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not ok:
        raise HTTPException(500, "Could not encode aligned face.")

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Face-Confidence": str(face["confidence"]),
            "X-Aligned-Size": "112x112",
        },
    )


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    image = await decode_upload(file)
    aligned, face, faces = get_best_aligned_face(image)

    try:
        embedding = generate_embedding(aligned)
    except Exception as e:
        raise HTTPException(
            500,
            f"MobileFaceNet inference failed: {e}",
        )

    return JSONResponse({
        "status": "success",
        "image": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        "face_count": len(faces),
        "selected_face": face,
        "alignment": {
            "method": "5-point_similarity",
            "size": [112, 112],
        },
        "recognition": {
            "loaded": True,
            "model": "w600k_mbf.onnx",
            "runtime": "onnxruntime-cpu",
            "embedding_dimension": int(embedding.shape[0]),
            "normalized": True,
        },
        "embedding": embedding.tolist(),
    })


@app.post("/embedding")
async def embedding(file: UploadFile = File(...)):
    image = await decode_upload(file)
    aligned, face, _ = get_best_aligned_face(image)
    vector = generate_embedding(aligned)

    return {
        "embedding": vector.tolist(),
        "dimension": int(vector.shape[0]),
        "normalized": True,
        "model": "w600k_mbf.onnx",
        "face_confidence": face["confidence"],
    }


@app.post("/verify")
async def verify(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
):
    image1 = await decode_upload(file1)
    image2 = await decode_upload(file2)

    aligned1, face1, _ = get_best_aligned_face(image1)
    aligned2, face2, _ = get_best_aligned_face(image2)

    try:
        embedding1 = generate_embedding(aligned1)
        embedding2 = generate_embedding(aligned2)
    except Exception as e:
        raise HTTPException(
            500,
            f"MobileFaceNet inference failed: {e}",
        )

    # Both embeddings are L2-normalized:
    # dot product == cosine similarity.
    similarity = float(np.dot(embedding1, embedding2))
    threshold = float(DEFAULT_THRESHOLD)
    verified = bool(similarity >= threshold)

    return {
        "status": "success",
        "model": "w600k_mbf.onnx",
        "runtime": "onnxruntime-cpu",
        "embedding_dimension": 512,
        "similarity": similarity,
        "cosine_similarity": similarity,
        "threshold": threshold,
        "verified": verified,
        "decision": "verified" if verified else "not_verified",
        "face1_confidence": face1["confidence"],
        "face2_confidence": face2["confidence"],
        "preprocessing": (
            "YuNet -> 5-point ArcFace alignment -> RGB -> [-1,1]"
        ),
        "note": "Similarity is a cosine score, not a probability.",
    }
