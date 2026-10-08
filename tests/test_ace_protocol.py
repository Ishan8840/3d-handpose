from dataclasses import replace
import numpy as np
from handpose.evaluation.ace_protocol import Hand, match, score, pose_errors

K = np.array([[100.,0,50],[0,100,50],[0,0,1]])
SIZE = (100,100)


def hand(side='right', z=1.):
    rng=np.random.default_rng(3)
    joints=rng.normal(0,.03,(21,3)); joints[:,2]+=z
    from handpose.evaluation.ace_protocol import project
    return Hand(side,joints,joints.copy(),np.array([0.,0,z]),np.eye(3),project(joints,K))


def test_translation_and_wrist_relative_are_distinct():
    g=hand(); p=replace(g,joints=g.joints+[0,0,.2],translation=g.translation+[0,0,.2])
    e=pose_errors(p,g)
    assert e['MPJPE-p_mm'] < 1e-10
    assert abs(e['CT-p_m']-.2) < 1e-10


def test_missing_penalty_is_canonical_not_constant():
    g=hand(); canonical=hand(z=0.)
    r=score([([], [g], K, SIZE)],{'right':canonical})
    assert r['FN']==1 and r['recall']==0 and r['F1']==0
    assert r['CT-p_m']==1.
    assert np.isclose(r['EPE2D-p_px'],np.hypot(*SIZE))


def test_side_collision_and_out_of_screen():
    g=hand()
    assert len(match([g,g],[g],K,SIZE)[0])==1
    assert len(match([g,g],[g],K,SIZE)[2])==1
    pairs,miss,extra=match([replace(g,side='left')],[g],K,SIZE)
    assert len(pairs)==0 and len(miss)==1 and len(extra)==1
    p=replace(g,joints=g.joints+[50,0,0])
    assert not match([p],[],K,SIZE)[2]


def test_perfect_and_threshold():
    g=hand()
    r=score([([g],[g],K,SIZE)],{'right':hand(z=0.)})
    assert r['recall']==r['F1']==r['FAcc']==1
    assert r['MPJPE-p_mm']==r['CT-p_m']==r['GO-p_deg']==r['EPE2D-p_px']==0
    assert not match([replace(g,existence=.5)],[g],K,SIZE)[0]


def test_rotation_metric_and_shape_root_frame_change():
    C=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    g=hand(); p=replace(g,rotation=C)
    assert np.isclose(pose_errors(p,g)['GO-p_deg'],90.)
    j0=np.array([.01,.02,.005]); tau=np.array([.1,-.2,.7]); t=np.array([.03,0,0])
    local=np.random.default_rng(1).normal(size=(21,3))*.02
    # Reexpressing MANO parameters must reproduce transformed vertices.
    old=local-j0+j0+tau
    new_tau=C@tau+t+C@j0-j0
    new=(local-j0)@C.T+j0+new_tau
    np.testing.assert_allclose(new,old@C.T+t,atol=1e-15)


def test_zero_overlap_and_gt_offscreen_not_in_denominator():
    g=hand(); p=replace(g,vertices=g.vertices+[.4,0,0])
    pairs,miss,extra=match([p],[g],K,SIZE)
    assert len(pairs)==0 and len(miss)==len(extra)==1
    off=replace(g,joints=g.joints+[50,0,0])
    r=score([([], [off], K, SIZE)],{'right':hand(z=0.)})
    assert r['FN']==0 and r['recall'] is None and r['MPJPE-p_mm'] is None


def test_epe_uses_visible_joint_population_not_mean_of_hands():
    from handpose.evaluation.ace_protocol import project
    g=hand(); p=replace(g,anchors=g.anchors+[1,0])
    j=g.joints.copy(); j[1:,0]+=50
    one=replace(g,joints=j,vertices=j,anchors=project(j,K))
    pred=replace(one,anchors=one.anchors+[2,0])
    r=score([([p],[g],K,SIZE),([pred],[one],K,SIZE)],{'right':hand(z=0.)})
    assert np.isclose(r['EPE2D-p_px'],23/22)
