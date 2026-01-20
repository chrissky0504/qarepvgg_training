import torch
import torch.nn as nn
from .repvgg import RepVGG, RepVGGBlock # 這裡引用您下載的那個檔案

# 定義人臉辨識專用的 QARepVGG
class QARepVGGFace(nn.Module):
    def __init__(self, num_blocks, width_multiplier, deploy=False):
        super(QARepVGGFace, self).__init__()
        
        # 1. 呼叫原始的 RepVGG 架構 (它現在已經是 QA 版本的 Block 了)
        self.backbone = RepVGG(
            num_blocks=num_blocks,
            width_multiplier=width_multiplier,
            override_groups_map=None,
            deploy=deploy
        )
        
        # 2. 【關鍵】砍掉 ImageNet 的分類頭 (Linear 1000)
        # 我們不需要它，留著只會佔記憶體
        del self.backbone.linear
        
        # 計算最後一層的通道數 (根據 B1 架構通常是 2048)
        # 這裡動態取得，避免寫死
        last_channel = 512 * width_multiplier[3] 
        
        # 3. 接上 ArcFace 標準的 Embedding Head
        # 結構：BN -> Dropout -> FC -> BN
        self.bn_input = nn.BatchNorm2d(int(last_channel))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(32768, 512)
        self.bn_output = nn.BatchNorm1d(512)

    def forward(self, x):
        # --- 跑骨幹 (Backbone) ---
        # RepVGG 的 forward 原始碼裡有經過 self.linear，我們因為砍掉了，
        # 所以不能直接 call self.backbone(x)，要一層一層跑 stage
        
        x = self.backbone.stage0(x)
        x = self.backbone.stage1(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        
        # --- 跑特徵頭 (Embedding Head) ---
        x = self.bn_input(x)
        x = x.view(x.size(0), -1) # 拉平 (Flatten)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_output(x)
        
        return x

# --- 方便呼叫的函式 ---

def create_QARepVGG_B1(deploy=False):
    """
    建立 QARepVGG-B1 模型 (推薦 AGX 使用這個版本)
    參數量適中，速度極快
    """
    return QARepVGGFace(
        num_blocks=[4, 6, 16, 1], 
        width_multiplier=[2, 2, 2, 4], 
        deploy=deploy
    )

def create_QARepVGG_A0(deploy=False):
    """
    建立 QARepVGG-A0 模型 (如果 B1 太慢，改用這個輕量版)
    """
    return QARepVGGFace(
        num_blocks=[2, 4, 14, 1], 
        width_multiplier=[0.75, 0.75, 0.75, 2.5], 
        deploy=deploy
    )