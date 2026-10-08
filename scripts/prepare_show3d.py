"""Download public train scenes with v3 labels; split by subject, no test labels."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

REPO='facebook/show3d-dataset'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='data/show3d'); a=p.parse_args(); out=Path(a.output)
    revision=HfApi().dataset_info(REPO).sha
    def fetch(name): return hf_hub_download(REPO,name,repo_type='dataset',revision=revision,local_dir=out)
    rows=pq.read_table(fetch('dataset_index_train.parquet')).to_pylist()
    chosen=[]; subjects=set()
    # Fixed task preference, no inspection of predictions or labels to choose scenes.
    for row in sorted(rows,key=lambda r:(not any(x in r['activity'] for x in ('pick-up','drawing','sorting')),r['scene_id'])):
        if row['subject_id'] not in subjects and row['has_hand_pose'] and row['has_headset0'] and row['has_headset1']:
            chosen.append(row); subjects.add(row['subject_id'])
        if len(chosen)==3: break
    manifest=[]
    for split,row in zip(('smoke','development','held_out'),chosen):
        subject,scene=row['subject_id'],row['scene_id']; base=f'scenes/{subject}/{scene}'
        files=[base+'/'+n for n in ('headset0.mp4','headset1.mp4','metadata/recording_info.json','metadata/frame_info.json','camera_calibration/headset0.json','camera_calibration/headset1.json')]
        files += [f'hand_pose/v3/scenes/{subject}/{scene}/hand_pose_umetrack.json']
        for f in files: fetch(f)
        manifest.append(dict(split=split,participant=subject,sequence=scene,path=str(out/base),annotations=str(out/files[-1]),revision=revision,max_frames=30 if split=='smoke' else 300))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
