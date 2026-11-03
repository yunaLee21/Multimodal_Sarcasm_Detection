"""Sinh lại reports/experiment_report.md từ kết quả hiện có (vimmsd/analysis/experiment_report.py).

    python scripts/build_experiment_report.py"""
from vimmsd.analysis.experiment_report import write_report

if __name__ == "__main__":
    print(f"đã ghi {write_report()}")
