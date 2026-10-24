"""
FastAPI 部署入口。
路由:
  GET  /health   健康检查
  POST /predict  弹幕分级 {"text": "..."}

本地运行:  uvicorn server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import BatchedPredictor, Predictor

logger = logging.getLogger("uvicorn.error")

_predictor: Optional[BatchedPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型 + 预热 + 启动动态批处理 worker, 关闭时无需额外清理。"""
    global _predictor
    t0 = time.time()
    _predictor = BatchedPredictor(Predictor())
    logger.info(f"模型加载完成, 设备={_predictor.predictor.conf.device}, 耗时 {time.time() - t0:.1f}s")
    yield


app = FastAPI(title="弹幕质量分级 API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200, description="弹幕文本(上限100字)")


class PredictResponse(BaseModel):
    code: int
    data: dict
    msg: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 为空")
    if len(text) > 100:
        raise HTTPException(status_code=400, detail="弹幕长度超过 100 字")

    t0 = time.time()
    try:
        data = _predictor.predict(text)
    except Exception as e:
        logger.exception("预测失败")
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")
    if "error" in data:
        raise HTTPException(status_code=500, detail=f"预测失败: {data['error']}")

    latency_ms = (time.time() - t0) * 1000
    data["text"] = text
    data["latency_ms"] = round(latency_ms, 2)
    logger.info(f"predict latency={latency_ms:.1f}ms -> {data['pred_class']}")
    return {"code": 0, "data": data, "msg": "ok"}
