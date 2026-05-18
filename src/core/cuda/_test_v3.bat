@echo off
REM Run v3 correctness + speed test on Windows.
REM Output captured to _test_v3_out.txt for git tracking.
cd /d %~dp0..\..\..\
python src\core\cuda\_test_v3.py > src\core\cuda\_test_v3_out.txt 2>&1
type src\core\cuda\_test_v3_out.txt
