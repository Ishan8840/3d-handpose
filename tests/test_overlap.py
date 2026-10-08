import numpy as np
from handpose.evaluation.overlap import compare_errors


def test_complementary_errors_and_missing_are_not_successes():
    gt=np.zeros((2,2,3));a=gt.copy();b=gt.copy()
    a[0,0,0]=.03;b[0,1,0]=.04
    a[1]=np.nan;b[1]=np.nan
    r=compare_errors(a,b,gt)
    assert r['failure_jaccard']==.5
    assert r['both_missing_rate']==.5
    assert r['a_failure_b_success_rate']==.25
    assert r['oracle_joint_selection_capped_mm']==50
    assert r['unweighted_average_common_mpjpe_mm']==17.5


def test_identical_error_vectors_and_invalid_ground_truth():
    gt=np.zeros((4,2,3));a=gt.copy()
    a[...,0]=np.arange(8).reshape(4,2)*.01
    gt[0,0]=np.nan
    r=compare_errors(a,a,gt)
    assert r['eligible_joint_slots']==7
    assert r['error_spearman_common']==1
    assert r['error_vector_cosine_common']==1
    assert r['a_capped_mm']==r['oracle_joint_selection_capped_mm']
