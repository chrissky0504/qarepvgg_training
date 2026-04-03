import torch
import torch.nn as nn
from .repvgg import RepVGG, get_RepVGG_func_by_name

# ====================================================================
# 1. 原本的 QARepVGGFace (完全不動，保留給你的 Baseline 使用)
# ====================================================================
class QARepVGGFace(nn.Module):
    def __init__(self, num_blocks, width_multiplier, deploy=False):
        super(QARepVGGFace, self).__init__()
        self.backbone = RepVGG(
            num_blocks=num_blocks,
            width_multiplier=width_multiplier,
            override_groups_map=None,
            deploy=deploy
        )
        if hasattr(self.backbone, 'linear'):
            del self.backbone.linear
        
        last_channel = 512 * width_multiplier[3] 
        self.bn_input = nn.BatchNorm2d(int(last_channel))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(32768, 512)
        self.bn_output = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.backbone.stage0(x)
        x = self.backbone.stage1(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        x = self.bn_input(x)
        x = x.reshape(x.size(0), -1) 
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_output(x)
        return x

# ====================================================================
# 2. 新增的特化版 QARepVGGFace_Outdoor (戶外長距離專用)
# ====================================================================
class QARepVGGFace_Outdoor(nn.Module):
    def __init__(self, network_name, deploy=False):
        super(QARepVGGFace_Outdoor, self).__init__()
        
        repvgg_func = get_RepVGG_func_by_name(network_name)
        self.backbone = repvgg_func(deploy=deploy)
        
        if hasattr(self.backbone, 'linear'):
            del self.backbone.linear
        if hasattr(self.backbone, 'gap'):
            del self.backbone.gap
            
        last_channel = self.backbone.in_planes 
        
        # 替換成 GhostFaceNets 推薦的 Modified GDC (解決 7x7 撐爆記憶體的問題)
        self.output_head = nn.Sequential(
            nn.Conv2d(last_channel, last_channel, kernel_size=7, stride=1, groups=last_channel, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.Conv2d(last_channel, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        x = self.backbone.stage0(x)
        x = self.backbone.stage1(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        x = self.output_head(x)
        return x

# ====================================================================
# 3. 呼叫函式區 (給 train_v2.py 抓取使用)
# ====================================================================

# --- 保留舊版的呼叫 ---
def create_QARepVGG_B1(deploy=False):
    return QARepVGGFace(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], deploy=deploy)

def create_QARepVGG_A0(deploy=False):
    return QARepVGGFace(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], deploy=deploy)

# --- 新增優化版的呼叫 ---
def create_QARepVGG_B1_Outdoor(deploy=False):
    return QARepVGGFace_Outdoor(network_name='QARepVGGV2PRELU-B1-Outdoor', deploy=deploy)

def create_QARepVGG_A0_Outdoor(deploy=False):
    return QARepVGGFace_Outdoor(network_name='QARepVGGV2PRELU-A0-Outdoor', deploy=deploy)