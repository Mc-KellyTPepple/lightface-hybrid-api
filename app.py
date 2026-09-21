import io,os,json,numpy as np,onnxruntime as ort
from PIL import Image
from fastapi import FastAPI,UploadFile,File,HTTPException
app=FastAPI(title="LightFace Hybrid API",version="1.0")
MODEL_PATH=os.getenv("MODEL_PATH","face_hybrid_int8.onnx")
META_PATH=os.getenv("META_PATH","deployment_metadata.json")
try:
    with open(META_PATH,"r") as f:META=json.load(f)
except: META={}
DEFAULT_THRESHOLD=float(os.getenv("VERIFY_THRESHOLD",META.get("verification",{}).get("recommended_deployment_threshold",.45)))
so=ort.SessionOptions()
so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads=1
so.inter_op_num_threads=1
session=ort.InferenceSession(MODEL_PATH,so,providers=["CPUExecutionProvider"])
MEAN=np.array([.485,.456,.406],dtype=np.float32)
STD=np.array([.229,.224,.225],dtype=np.float32)
def prep(data):
    try: im=Image.open(io.BytesIO(data)).convert("RGB").resize((112,112))
    except Exception as e: raise HTTPException(400,f"Invalid image: {e}")
    x=np.asarray(im,dtype=np.float32)/255.0
    x=(x-MEAN)/STD
    return np.transpose(x,(2,0,1))[None]
def embed(data):
    z=session.run(["embedding"],{"image":prep(data)})[0][0].astype(np.float32)
    return z/(np.linalg.norm(z)+1e-12)
@app.get("/health")
def health():
    return {"status":"ok","model":"face_hybrid_int8.onnx","runtime":"onnxruntime-cpu","embedding_dim":256}
@app.post("/embedding")
async def embedding(file:UploadFile=File(...)):
    z=embed(await file.read())
    return {"embedding":z.tolist(),"dimension":int(z.shape[0])}
@app.post("/verify")
async def verify(file1:UploadFile=File(...),file2:UploadFile=File(...),threshold:float|None=None):
    z1=embed(await file1.read());z2=embed(await file2.read())
    score=float(np.dot(z1,z2))
    t=DEFAULT_THRESHOLD if threshold is None else float(threshold)
    return {"similarity":score,"threshold":t,"verified":bool(score>=t)}
