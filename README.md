# 人臉辨識訓練與評估實驗紀錄 (ArcFace PyTorch)

本文件紀錄了在基於 InsightFace (ArcFace) PyTorch 訓練框架下的模型修改、訓練、評估與部署測試流程。

## 📍 實驗流程指南

### 1. 模型架構修改 (`backbones/`)
- 匯入新的模型架構檔 (例如: `repvgg.py`)。
- 編輯 `backbones/__init__.py`，將要訓練的新模型加入 mapping。
- 視情況撰寫適配人臉辨識訓練的 Wrapper（，例如: `qarepvgg_face.py`），調整網路的輸出層（如拔除 ImageNet Classification 頭）以提取 Face Embedding。

### 2. 準備設定檔 (`configs/`)
- 在 `configs/` 目錄下新增自定義的 config 檔（例如: `my_qarepvgg_run.py`），設定資料集路徑、Batch Size、學習率、網路選擇等參數。

### 3. 開始訓練
執行以下指令開始訓練模型：
```bash
python train_v2.py configs/my_qarepvgg_run.py 
```

### 4. 模型評估 (Testing)
評估訓練好的模型 (Checkpoint)：
```bash
# Evaluate model (checkpoint path needed)
python eval_model.py --checkpoint /path/to/checkpoint.pt

# Run IJBC evaluation with pt model
python eval_ijbc.py --model-prefix qae20.pt --network qarepvgg_b1
```

### 5. 計算模型參數量與 FLOPs
```bash
python parm.py
```

### 6. 模型部署準備：結構重參數化 (Re-parameterization)
針對 RepVGG 等需要重參數化的模型，將多個分支合併為單一卷積層以加速推論：
```bash
python convert_deploy.py
```

### 7. 模型轉換格式 (ONNX)
將 PyTorch (`.pt`) 模型轉換為 ONNX 格式，供 TensorRT 或其他推論引擎使用：
```bash
python torch2onnx.py
```

### 8. TensorRT 推論速度測試 (Benchmarking)
使用 TensorRT 的 `trtexec` 工具測試 ONNX 模型的推論效能 (FP16)：
```bash
/usr/src/tensorrt/bin/trtexec --onnx=qarepvgg_b1_40.onnx --fp16 --memPoolSize=workspace:16384 --iterations=100 --avgTiming=100
```

---

## 📝 開發日誌

### 2026/04/03 系統特化改動

針對特定場景（主要為**戶外長距離**與**移動車體**），對資料增強與模型架構進行了深度優化：

#### 1. 資料集資料增強 (Data Augmentation) 特殊處理
為了解決實務遇到的場景，加入了以下模擬增強：
- **模擬長距離：特徵流失與馬賽克感**
  - 使用 *Downscale* 技巧，將訓練圖片縮小後再放大回原尺寸，模擬遠距離造成的小面積人臉與特徵缺損。
- **模擬車體震動與對焦失準**
  - 增強模型對動態模糊與失焦情況下的特徵提取能力。
- **模擬真實感光環境**
  - 模擬背光 (Backlight)、過度曝光，以及這類極端環境（如 MX Brio 攝影機）容易產生的感光雜訊。

#### 2. 模型激勵函數升級
- 將原 RepVGG 內部的 `ReLU` 替換為 `PReLU` (Parametric ReLU)，保留更多微小的負值特徵，這對臉部細節的特徵提取更有利。

#### 3. 網路輸出頭優化 (Head Modification)
- 在 `qarepvgg_face.py` 中，捨棄傳統龐大的 Fully Connected (FC) 結構，將其替換為 **Modified GDC (Global Depthwise Convolution)**。
- **改動效益**：大幅減少參數量並降低過擬合風險。此設計專為 Stride 2 產生的 4x4 特徵圖量身打造，能將特徵平滑並穩定地壓縮至 512 維度 Embedding。