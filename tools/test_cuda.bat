@echo off
call conda activate raft-dvc
python test_cuda.py
pause
