from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path
import shutil
root=Path('third_party/ace')
snapshot_download('alibaba-pai/Wan2.2-Fun-5B-Control',revision='b8bc1a65ab71d054ba4636dc0dac104aa4df2686',local_dir=root/'ckpt/Wan2.2-Fun-5B-Control',max_workers=2)
hf_hub_download('acerobotics2025/ACE-Ego-Hand','ace_ego_hand_k.pt',revision='6fb3decade1b0331048fa9d02df605e95a0575c2',local_dir=root/'checkpoints')
(root/'data/mano/mano').mkdir(parents=True,exist_ok=True)
for p in Path('checkpoints/mano_converted').glob('*.pkl'):shutil.copy2(p,root/'data/mano/mano'/p.name)
