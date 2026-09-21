# LightFace Hybrid — Render Free Deployment

Architecture:
MobileFaceNet + MicroCNN + ECA + bounded gated fusion.

Input:
112x112 RGB face image.

Embedding:
256-dimensional L2-normalized vector.

Deployment:
ONNX Runtime CPU + INT8 dynamic quantization.

PyTorch is NOT required by the Render service.

Calibrated verification threshold:
0.250449

FP32 ONNX size:
3.996 MB

INT8 ONNX size:
1.138 MB

INT8 compression:
71.53%

INT8 mean CPU latency:
49.324 ms

INT8 P50 latency:
48.984 ms

INT8 P95 latency:
51.044 ms

INT8 throughput:
20.274 images/sec

Parameters:
2,839,481

MACs:
146560736.0

FLOPs:
293121472.0

Important:
The API expects an already-cropped face. It does not perform face detection or face alignment.
