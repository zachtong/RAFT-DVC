@echo off
REM Run v3 correctness + speed test on Windows.
REM Auto-discovers MSVC vcvars64.bat (cl.exe must be on PATH for PyTorch JIT).
setlocal

REM ---- Find VS vcvars64.bat ---------------------------------------------------
REM Check the most common install locations in order.
set "VCVARS="
for %%P in (
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) do (
    if exist %%P (
        set "VCVARS=%%P"
        goto :found_vcvars
    )
)

echo ERROR: Could not find vcvars64.bat in any standard location.
echo Install Visual Studio 2022 Build Tools (or Community) with C++ workload.
echo Or set VCVARS env var manually before running this script.
exit /b 1

:found_vcvars
echo [vcvars] Using %VCVARS%
call %VCVARS% >nul

REM Verify cl.exe is now on PATH
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe still not on PATH after vcvars64.bat.
    exit /b 2
)
echo [vcvars] OK -- cl.exe found.

REM ---- Faster compile: target only sm_120 (5090 Blackwell) -------------------
set TORCH_CUDA_ARCH_LIST=12.0

REM ---- Run test --------------------------------------------------------------
cd /d %~dp0..\..\..\
python src\core\cuda\_test_v3.py > src\core\cuda\_test_v3_out.txt 2>&1
type src\core\cuda\_test_v3_out.txt

endlocal
