import pytest
import numpy as np
from src.data.label_mapping import verify_and_map_labels, DEFAULT_LABEL_MAPPING
from src.data.split import create_patient_splits

def test_label_mapping():
    raw_labels = np.array([0, 1, 2, 3, 0, 1, 2])
    mapped_arr, stats = verify_and_map_labels(raw_labels)
    assert mapped_arr.shape == raw_labels.shape
    assert "Kidney" in stats
    assert stats["Kidney"] == 2

def test_patient_split_no_leakage(tmp_path):
    cases = [f"case_{i:05d}" for i in range(100)]
    train, val, test = create_patient_splits(cases, output_dir=str(tmp_path))
    
    set_tr = set(train)
    set_va = set(val)
    set_te = set(test)
    
    assert len(set_tr.intersection(set_va)) == 0
    assert len(set_tr.intersection(set_te)) == 0
    assert len(set_va.intersection(set_te)) == 0
    assert len(set_tr) + len(set_va) + len(set_te) == 100

if __name__ == "__main__":
    pytest.main([__file__])
