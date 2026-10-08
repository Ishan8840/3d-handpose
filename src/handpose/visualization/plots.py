import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from handpose.models.base import JOINT_NAMES

def diagnostic_plots(pred,gt,timestamps,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    error=np.linalg.norm(pred-gt,axis=-1)*1000;finite=np.isfinite(error)
    fig,ax=plt.subplots(figsize=(10,4));means=[np.mean(error[:,i][finite[:,i]]) if finite[:,i].any() else np.nan for i in range(21)]
    ax.bar(range(21),means);ax.set_xticks(range(21),JOINT_NAMES,rotation=90);ax.set_ylabel('Absolute error (mm)');fig.tight_layout();fig.savefig(out/'per_joint.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].hist(error[finite],bins=40);axs[0].set_xlabel('Absolute error (mm)');axs[0].set_ylabel('Observed joints')
    axs[1].scatter(gt[...,2][finite],error[finite],s=2,alpha=.2);axs[1].set_xlabel('GT camera depth (m)');axs[1].set_ylabel('Absolute error (mm)');fig.tight_layout();fig.savefig(out/'error_depth.png');plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True);t=(timestamps-timestamps[0])*1e-9
    for d,ax in enumerate(axs):
        ax.plot(t,gt[:,8,d],label='GT index tip');ax.plot(t,pred[:,8,d],label='Prediction');ax.set_ylabel('xyz'[d]+' (m)');ax.legend()
    axs[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(out/'trajectory.png');plt.close(fig)
    scores=np.divide(np.nansum(error,axis=1),finite.sum(1),out=np.full(len(error),np.nan),where=finite.sum(1)>0)
    worst=np.argsort(np.nan_to_num(scores,nan=-1))[-10:][::-1]
    if len(worst):
        from handpose.models.base import EDGES
        i=int(worst[0]);fig=plt.figure(figsize=(6,6));ax=fig.add_subplot(111,projection='3d')
        for points,color,label in ((gt[i],'tab:red','GT'),(pred[i],'tab:blue','prediction')):
            for j,k in EDGES:
                if np.isfinite(points[[j,k]]).all():ax.plot(*points[[j,k]].T,color=color,alpha=.8)
            good=np.isfinite(points).all(-1)
            ax.scatter(*points[good].T,color=color,s=10,label=label)
        ax.set_xlabel('camera x (m)');ax.set_ylabel('camera y (m)');ax.set_zlabel('camera z (m)');ax.set_box_aspect((1,1,1));ax.legend();fig.tight_layout();fig.savefig(out/'pose_3d.png');plt.close(fig)
    return [{'frame_index':int(i),'observed_mpjpe_mm':float(scores[i]),'observed_joints':int(finite[i].sum())} for i in worst if np.isfinite(scores[i])]
