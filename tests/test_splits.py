import json
from pathlib import Path
import pytest

@pytest.mark.parametrize('dataset',['hot3d','show3d'])
def test_recorded_dataset_splits_disjoint(dataset):
    entries=json.loads((Path(__file__).parents[1]/'configs/datasets'/f'{dataset}.json').read_text())
    subjects={};sequences={}
    for e in entries:
        assert subjects.setdefault(e['participant'],e['split'])==e['split']
        assert sequences.setdefault(e['sequence'],e['split'])==e['split']
    assert set(subjects.values())=={'smoke','development','held_out'}
