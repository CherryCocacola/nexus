# SD3.5 Large 이미지 서버 — ImageGenerate 도구 계약(/v1/images/generate). text2img + img2img.
import base64
import io
import os

import torch
import uvicorn
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

MODEL_ID = os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-3.5-large")
DEVICE = "cuda"
print(f"[image] {MODEL_ID} 로드 시작 ...", flush=True)
_pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
# img2img 파이프라인 — from_pipe로 동일 컴포넌트를 공유하므로 추가 VRAM을 쓰지 않는다.
_img2img = StableDiffusion3Img2ImgPipeline.from_pipe(_pipe)
print("[image] 로드 완료 (text2img + img2img)", flush=True)
app = FastAPI(title="nexus-image-sd35")


class GenReq(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 28
    seed: int | None = None
    # img2img — 입력 이미지 base64가 있으면 "이미지→이미지"로 원본에서 출발해 수정한다.
    input_image_base64: str | None = None
    # 변형 강도. 0에 가까울수록 원본 유지, 1에 가까울수록 프롬프트대로 크게 바꾼다.
    strength: float = 0.6


@app.post("/v1/images/generate")
def generate(req: GenReq):
    steps = (
        req.steps if (req.steps and req.steps >= 20) else 28
    )  # SD3.5는 저스텝 품질 저하 → 하한 보정
    seed = req.seed if req.seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    if req.input_image_base64:
        # img2img: 업로드 이미지 픽셀에서 출발 → 원본 스타일·구도 유지하며 수정.
        raw = base64.b64decode(req.input_image_base64)
        init = Image.open(io.BytesIO(raw)).convert("RGB").resize((req.width, req.height))
        strength = min(max(float(req.strength), 0.1), 0.95)
        img = _img2img(
            prompt=req.prompt, image=init, strength=strength, num_inference_steps=steps, generator=g
        ).images[0]
        mode = f"img2img(strength={strength})"
    else:
        img = _pipe(
            req.prompt, width=req.width, height=req.height, num_inference_steps=steps, generator=g
        ).images[0]
        mode = "text2img"
    print(f"[image] 생성 mode={mode} seed={seed} {req.width}x{req.height}", flush=True)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "image_base64": base64.b64encode(buf.getvalue()).decode(),
        "width": req.width,
        "height": req.height,
        "seed": seed,
        "model": MODEL_ID,
        "mode": mode,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "modes": ["text2img", "img2img"]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="warning")  # noqa: S104
