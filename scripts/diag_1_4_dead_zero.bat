@echo off
REM Run dead-zero diagnostic on paper-1 1/4 case.
REM Inspects: input data, encoder outputs (1/8 vs 1/4 random init), gradient signal.

cd /d %~dp0..\

python scripts\diag_1_4_dead_zero.py ^
    --data-dir data_paper1_v2\r4_medium_size64\train ^
    --indices 0 1 2 5 ^
    --output reports\diag_1_4_dead_zero_2026-05-18.pdf

echo.
echo Open reports\diag_1_4_dead_zero_2026-05-18.pdf
