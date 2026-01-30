import io
import numpy as np
import torch
from PIL import Image
import matplotlib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, FileResponse, JSONResponse
import tempfile
import base64
from pydantic import BaseModel

from depth_anything_v2.dpt import DepthAnythingV2

app = FastAPI(
    title="Depth Anything V2 API",
    description="API for Depth Anything V2 depth prediction",
    version="1.0.0"
)

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# 模型配置
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# 用于响应的模型
class DepthResponse(BaseModel):
    depth_array: list
    width: int
    height: int
    min_depth: float
    max_depth: float

# 加载模型
@app.on_event("startup")
async def startup_event():
    global model, cmap
    encoder = 'vitl'
    model = DepthAnythingV2(**model_configs[encoder])
    state_dict = torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    
    # 设置色彩映射
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

def predict_depth(image):
    return model.infer_image(image)

@app.post("/predict/color_depth")
async def predict(file: UploadFile = File(...)):
    """
    上传图像并返回深度预测结果
    
    参数:
    - file: 上传的图像文件
    
    返回:
    - 彩色深度图图像
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上传的文件必须是图像")
    
    try:
        # 读取上传的图像
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        # 转换为RGB确保格式正确
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        # 转换为numpy数组
        image = np.array(pil_image)
        
        # 预测深度
        depth = predict_depth(image[:, :, ::-1])  # RGB转BGR
        
        # 归一化深度图
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_norm = depth_norm.astype(np.uint8)
        
        # 创建彩色深度图
        colored_depth = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        colored_depth_pil = Image.fromarray(colored_depth)
        
        # 保存并返回深度图
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            colored_depth_pil.save(tmp.name)
            return FileResponse(tmp.name, media_type="image/png", filename="depth_map.png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图像时发生错误: {str(e)}")

@app.post("/predict/raw_depth_array")
async def predict_raw_array(file: UploadFile = File(...)):
    """
    上传图像并返回原始深度数据数组
    
    参数:
    - file: 上传的图像文件
    
    返回:
    - 深度数据数组及其元数据
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上传的文件必须是图像")
    
    try:
        # 读取上传的图像
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        image = np.array(pil_image)
        
        # 预测深度
        depth = predict_depth(image[:, :, ::-1])  # RGB转BGR
        
        # 获取深度范围
        min_depth = float(depth.min())
        max_depth = float(depth.max())
        
        # 转换为列表以便JSON序列化
        depth_list = depth.flatten().tolist()
        
        # 返回深度数组数据
        return DepthResponse(
            depth_array=depth_list,
            width=depth.shape[1],  
            height=depth.shape[0],
            min_depth=min_depth,
            max_depth=max_depth
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图像时发生错误: {str(e)}")

@app.post("/predict/depth")
async def predict(file: UploadFile = File(...)):
    """
    上传图像并返回深度预测结果
    
    参数:
    - file: 上传的图像文件
    
    返回:
    - 灰色深度图图像
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上传的文件必须是图像")

    try:
        # 读取上传的图像
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        image = np.array(pil_image)
        
        # 预测深度
        depth = predict_depth(image[:, :, ::-1])  # RGB转BGR
        
        # 归一化深度图
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_norm = depth_norm.astype(np.uint8)
        
        # 创建灰度深度图
        depth_image = Image.fromarray(depth_norm, mode="L")
        
        # 保存并返回深度图
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            depth_image.save(tmp.name)
            return FileResponse(tmp.name, media_type="image/png", filename="depth_map_gray.png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图像时发生错误: {str(e)}")

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7865)