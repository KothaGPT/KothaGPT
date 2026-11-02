from scripts.doctor import main


def test_doctor_reports_repo_baseline():
    assert main([]) == 0
