import torch

# 1. 載入你的 .pt 檔 (使用 map_location='cpu' 避免佔用 GPU)
model_path = "qa40_deploy.pt" # 替換成你的 model_best.pt 或 model.pt 路徑
state_dict = torch.load(model_path, map_location='cpu')

# 2. 如果你是讀取 checkpoint_gpu_0.pt，權重會包在另一個 key 裡面
# state_dict = state_dict["state_dict_backbone"]

# 3. 計算所有權重的元素總和
total_params = sum(tensor.numel() for tensor in state_dict.values())

print(f"模型總參數量: {total_params:,}")