"""EgoStandard MCAP decoding. Joint mapping must come from verified documentation.

Embedded protobuf schemas are used directly; no generated project messages needed.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def messages(path):
    from mcap.reader import make_reader
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    with open(path,'rb') as f:
        reader=make_reader(f);classes={}
        for schema,channel,msg in reader.iter_messages():
            if schema.id not in classes:
                pool=descriptor_pool.DescriptorPool()
                pending=list(descriptor_pb2.FileDescriptorSet.FromString(schema.data).file)
                while pending:
                    remaining=[]
                    for fd in pending:
                        try:pool.Add(fd)
                        except Exception:remaining.append(fd)
                    if len(remaining)==len(pending):raise ValueError('Unresolved schema dependencies')
                    pending=remaining
                descriptor=pool.FindMessageTypeByName(schema.name)
                classes[schema.id]=message_factory.GetMessageClass(descriptor) if hasattr(message_factory,'GetMessageClass') else message_factory.MessageFactory(pool).GetPrototype(descriptor)
            yield channel.topic,msg.log_time,classes[schema.id].FromString(msg.data)


def world_from_camera(transform, translation_to_meters):
    q=transform.rotation;t=transform.translation
    T=np.eye(4);T[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
    T[:3,3]=np.array([t.x,t.y,t.z])*translation_to_meters
    if transform.parent_frame_id!='world':raise ValueError('Expected world-parent extrinsic')
    return T


def canonical_hand(transforms, source_joint_names, position_to_meters, T_world_from_camera):
    """Explicit convention conversion; never infer joint names from tensor shape."""
    from handpose.models.base import JOINT_NAMES
    if len(transforms)!=len(source_joint_names) or len(set(source_joint_names))!=len(source_joint_names):raise ValueError('Invalid joint convention')
    if set(JOINT_NAMES)!=set(source_joint_names):raise ValueError('Expected all canonical joints')
    if not np.isfinite(position_to_meters) or position_to_meters<=0:raise ValueError('Explicit positive unit scale required')
    xyz=np.array([[t.pos.x,t.pos.y,t.pos.z] for t in transforms])*position_to_meters
    xyz=xyz[[source_joint_names.index(j) for j in JOINT_NAMES]]
    camera_from_world=np.linalg.inv(T_world_from_camera)
    return xyz@camera_from_world[:3,:3].T+camera_from_world[:3,3]


def synchronized_indices(left_ns,right_ns,tolerance_ns=1_000_000):
    """Unique nearest matches; unmatched left frames remain -1 for coverage."""
    left=np.asarray(left_ns,dtype=np.int64);right=np.asarray(right_ns,dtype=np.int64)
    if np.any(np.diff(left)<=0) or np.any(np.diff(right)<=0):raise ValueError('Nonmonotonic timestamps')
    output=np.full(len(left),-1,dtype=int);last=-1
    for i,t in enumerate(left):
        index=np.searchsorted(right,t)
        candidates=[j for j in (index-1,index) if last<j<len(right)]
        if candidates:
            j=min(candidates,key=lambda j:abs(int(right[j])-int(t)))
            if abs(int(right[j])-int(t))<=tolerance_ns:output[i]=j;last=j
    return output
