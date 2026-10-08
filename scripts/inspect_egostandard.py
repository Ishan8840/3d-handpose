"""Inspect embedded MCAP schemas without external generated protobuf packages."""
import argparse,json,hashlib
from pathlib import Path
from mcap.reader import make_reader
from google.protobuf import descriptor_pb2,descriptor_pool,message_factory,json_format
p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--output',required=True);a=p.parse_args()
path=Path(a.path);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
with path.open('rb') as f:
    reader=make_reader(f);summary=reader.get_summary();classes={}
    for sid,schema in summary.schemas.items():
        pool=descriptor_pool.DescriptorPool();fds=descriptor_pb2.FileDescriptorSet.FromString(schema.data)
        pending=list(fds.file)
        while pending:
            next_pending=[]
            for fd in pending:
                try:pool.Add(fd)
                except Exception:next_pending.append(fd)
            if len(next_pending)==len(pending):raise ValueError('Unresolved protobuf dependencies')
            pending=next_pending
        descriptor=pool.FindMessageTypeByName(schema.name)
        classes[sid]=message_factory.GetMessageClass(descriptor) if hasattr(message_factory,'GetMessageClass') else message_factory.MessageFactory(pool).GetPrototype(descriptor)
    first={};counts={};times={};video_files={}
    for schema,channel,msg in reader.iter_messages():
        topic=channel.topic;counts[topic]=counts.get(topic,0)+1
        decoded=classes[schema.id].FromString(msg.data)
        times.setdefault(topic,[]).append(msg.log_time)
        if topic.endswith('/video'):
            name=topic.split('/')[-2]
            if topic not in video_files:video_files[topic]=(out/(name+'.h264')).open('wb')
            video_files[topic].write(decoded.data)
            if topic not in first:
                decoded.ClearField('data');first[topic]=json_format.MessageToDict(decoded,preserving_proto_field_name=True)
        elif topic not in first:first[topic]=json_format.MessageToDict(decoded,preserving_proto_field_name=True)
    for file in video_files.values():file.close()
report={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'counts':counts,'first_messages':first,'log_timestamps_ns':times}
(out/'inspection.json').write_text(json.dumps(report,indent=2));print(json.dumps({'counts':counts,'first_messages':first},indent=2))
