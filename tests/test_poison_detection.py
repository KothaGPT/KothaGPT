from data.pipeline.security import is_poisoned, scan_dataset


def test_poison():
    assert is_poisoned("this has backdoor trigger")[0]
    assert is_poisoned("@@trigger@@")[0]
    assert not is_poisoned("normal text")[0]
    assert len(scan_dataset([{"text": "backdoor"}, {"text": "clean"}])) == 1
