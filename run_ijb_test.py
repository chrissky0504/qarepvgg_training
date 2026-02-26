import os
import cv2
import numpy as np
import argparse
import onnxruntime
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from skimage import transform as trans

# 這是標準的 ArcFace 參考點
src1 = np.array([
     [30.2946, 51.6963],
     [65.5318, 51.5014],
     [48.0252, 71.7366],
     [33.5493, 92.3655],
     [62.7299, 92.2041] ], dtype=np.float32)
src1[:, 0] += 8.0

def preprocess(img, landmark):
    # 對齊人臉 (Alignment)
    tform = trans.SimilarityTransform()
    tform.estimate(landmark, src1)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    # 正規化
    warped = (warped - 127.5) / 128.0
    return warped.transpose(2, 0, 1).astype(np.float32)

def get_embeddings(onnx_path, img_root, meta_file):
    # 1. 初始化 ONNX Runtime
    session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 2. 讀取 Meta Data
    print("📖 讀取圖片列表與關鍵點...")
    with open(meta_file, 'r') as f:
        lines = f.readlines()
        
    embeddings = []
    names = []
    
    # 3. 批次推論 (Batch Inference)
    batch_size = 1
    batch_imgs = []
    
    print("🚀 開始提取特徵 (Feature Extraction)...")
    for line in tqdm(lines):
        parts = line.strip().split()
        img_path = os.path.join(img_root, parts[0])
        landmark = np.array([float(x) for x in parts[1:11]]).reshape(5, 2)
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        aligned = preprocess(img, landmark)
        batch_imgs.append(aligned)
        
        if len(batch_imgs) == batch_size:
            blob = np.array(batch_imgs)
            # 執行推論
            feat = session.run(None, {input_name: blob})[0]
            embeddings.append(feat)
            batch_imgs = []

    # 處理剩下的
    if batch_imgs:
        blob = np.array(batch_imgs)
        feat = session.run(None, {input_name: blob})[0]
        embeddings.append(feat)
        
    embeddings = np.concatenate(embeddings, axis=0)
    # 特徵正規化 (L2 Norm)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='ONNX 模型路徑')
    parser.add_argument('--img_root', type=str, required=True, help='IJB-C 圖片資料夾')
    parser.add_argument('--meta_file', type=str, required=True, help='ijbc_name_5pts_score.txt 路徑')
    args = parser.parse_args()
    
    embs = get_embeddings(args.model, args.img_root, args.meta_file)
    print(f"✅ 特徵提取完成，形狀: {embs.shape}")
    np.save("ijbc_embeddings.npy", embs)
    print("💾 特徵已存檔，接下來請使用 IJB 評測工具計算 TAR@FAR。")