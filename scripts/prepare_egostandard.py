"""Export a downloaded EgoStandard MCAP without inventing hand joint semantics.

Explicit --translation-to-meters is recorded as an assumption unless accompanied
by a verified convention document. Native hand arrays retain source indexing.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from google.protobuf.json_format import MessageToDict
from handpose.data.egostandard import messages,world_from_camera,synchronized_indices
p=argparse.ArgumentParser();p.add_argument('--mcap',required=True);p.add_argument('--output',required=True);p.add_argument('--translation-to-meters',type=float,required=True);p.add_argument('--convention-source',default='unverified assumption');a=p.parse_args()
if a.translation_to_meters<=0:raise ValueError('Positive translation unit scale required')
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);streams={};timestamps={};calibration={};extrinsics={};hands=[];hand_ts=[];bad=[];bad_ts=[];session=None
for topic,log_ns,msg in messages(a.mcap):
    if topic.endswith('/video'):
        side='left' if 'head_left' in topic else 'right'
        if side not in streams:streams[side]=(out/f'{side}.h264').open('wb')
        if msg.format!='h264':raise ValueError('Unsupported codec '+msg.format)
        streams[side].write(msg.data);timestamps.setdefault(side,[]).append(msg.timestamp.seconds*10**9+msg.timestamp.nanos)
    elif topic.endswith('/intrinsic'):
        side='left' if 'head_left' in topic else 'right';d=MessageToDict(msg,preserving_proto_field_name=True)
        camera={k:v for k,v in d.items() if k!='timestamp'}
        if side in calibration and calibration[side]!=camera:raise ValueError('Time-varying camera intrinsics need a per-frame adapter')
        calibration[side]=camera
    elif topic.endswith('/extrinsic'):
        side='left' if 'head_left' in topic else 'right';extrinsics.setdefault(side,[]).append(world_from_camera(msg.transforms[0],a.translation_to_meters))
    elif topic=='/pose/right_hand':
        hands.append([[t.pos.x,t.pos.y,t.pos.z] for t in msg.transforms]);hand_ts.append(msg.header.ts)
    elif topic=='/annotation/bad_frame/pose/hand':bad.append(msg.is_bad);bad_ts.append(msg.header.ts)
    elif topic=='/session/metadata':session=MessageToDict(msg,preserving_proto_field_name=True)
for stream in streams.values():stream.close()
if set(streams)!={'left','right'}:raise ValueError('Missing stereo stream')
index=synchronized_indices(timestamps['left'],timestamps['right'])
if np.any(index!=np.arange(len(index))) or len(timestamps['left'])!=len(timestamps['right']):raise ValueError('Streams require timestamp-aware resampling; refusing positional pairing')
Ts=np.linalg.inv(np.asarray(extrinsics['right']))@np.asarray(extrinsics['left'])
if np.max(np.abs(Ts-Ts[0]))>1e-5:raise ValueError('Time-varying stereo extrinsics need per-frame inference')
result={'units':'meters','transform_convention':'T_camera_from_left','unit_convention_source':a.convention_source}
for side in ('left','right'):
    d=calibration[side]
    if d.get('D') and any(d['D']):raise ValueError('Audit camera distortion model before inference')
    result[side]={'K':np.array(d['K']).reshape(3,3).tolist(),'T_camera_from_left':(np.eye(4) if side=='left' else Ts[0]).tolist(),'model':'opencv','distortion':[]}
    subprocess.run(['ffmpeg','-y','-loglevel','error','-r','30','-i',str(out/f'{side}.h264'),'-c','copy',str(out/f'{side}.mp4')],check=True)
(out/'calibration.json').write_text(json.dumps(result,indent=2))
np.savez_compressed(out/'timestamps.npz',**timestamps)
np.savez_compressed(out/'native_annotations.npz',right_hand_positions_source_units=hands,hand_timestamps_ns=hand_ts,bad_frame=bad,bad_frame_timestamps_ns=bad_ts,T_world_from_left=extrinsics['left'])
(out/'manifest.json').write_text(json.dumps({'dataset':'LightwheelAI/EgoStandard','source_sha256':hashlib.sha256(Path(a.mcap).read_bytes()).hexdigest(),'translation_to_meters':a.translation_to_meters,'convention_source':a.convention_source,'hand_joint_order':'unverified; native indices preserved; excluded from anatomical accuracy scoring','frames':len(index),'baseline_m':float(np.linalg.norm(Ts[0,:3,3])),'session':session},indent=2))
print('Exported',len(index),'synchronized stereo frames; pose scoring disabled pending convention verification')
