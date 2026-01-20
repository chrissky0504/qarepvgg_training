import torch
import os
from backbones.qarepvgg_face import create_QARepVGG_B1 
from backbones.repvgg import repvgg_model_convert

# 設定路徑 (請修改成您 model.pt 所在的實際路徑)
# 因為您的 config.output = None，它通常會存在 work_dirs/ms1m-retinaface-t1/model.pt
model_path = "work_dirs/my_qarepvgg_run/model.pt" 

print(f"正在載入: {model_path} ...")

# 1. 載入訓練好的模型 (訓練態 deploy=False)
# 確保您的 qarepvgg_face.py 裡 RepVGG 初始化有加 block_cls=QARepVGGBlockV2
train_model = create_QARepVGG_B1(deploy=False)
state_dict = torch.load(model_path, map_location='cpu')
train_model.load_state_dict(state_dict)
train_model.eval()

# 2. 建立推論用的空模型 (推論態 deploy=True)
deploy_model = create_QARepVGG_B1(deploy=True)

# 3. 【關鍵】執行結構重參數化 (Reparameterization)
print("正在執行重參數化 (Reparameterization)...")
deploy_model.backbone = repvgg_model_convert(train_model.backbone, save_path=None)

# 4. 遷移人臉特徵頭 (Embedding Head)
deploy_model.bn_input.load_state_dict(train_model.bn_input.state_dict())
deploy_model.fc.load_state_dict(train_model.fc.state_dict())
deploy_model.bn_output.load_state_dict(train_model.bn_output.state_dict())

# 5. 存檔
save_name = "qarepvgg_b1_deploy.pt"
torch.save(deploy_model.state_dict(), save_name)
print(f"變身完成！已儲存為: {save_name}")