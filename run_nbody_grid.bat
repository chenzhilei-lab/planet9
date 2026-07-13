@echo off
REM ============================================================================
REM run_nbody_grid.bat — Parallel N-body scattering grid for 6 × 4060Ti
REM ============================================================================
REM Usage: run on each 4060Ti machine, adjusting GPU_ID (0-5) for each.
REM   6 machines × 1 parameter set each = 6 independent runs overnight.
REM   Output goes to D:\Papers\MNRAS\submission_record\scripts\nbody_output\
REM ============================================================================

set SCRIPT=D:\Papers\MNRAS\submission_record\scripts\nbody_scattering.py
set OUTDIR=D:\Papers\MNRAS\submission_record\scripts\nbody_output

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM ---- Pre-flight: install REBOUND if needed ----
REM pip install rebound reboundx
REM Verify: python -c "import rebound; print('OK')"

REM ============================================================================
REM Run 1: Scattering only — null hypothesis (no Planet Nine)
REM GPU 0 | ~300 particles × 100 Myr | Est. ~2-3 hours
REM ============================================================================
start "GPU0_scatter" python "%SCRIPT%" ^
    --mode scatter ^
    --N 300 --tmax 100 ^
    --seed 42001 ^
    --output "%OUTDIR%\nbody_scatter_null.json"

REM ============================================================================
REM Run 2: P9 M=5 M_earth, a=400 AU — light perturber
REM GPU 1 | ~300 particles × 100 Myr | Est. ~2-3 hours
REM ============================================================================
start "GPU1_P9_M5_a400" python "%SCRIPT%" ^
    --mode p9 ^
    --M_p9 5 --a_p9 400 --e_p9 0.3 --i_p9 15 ^
    --N 300 --tmax 100 ^
    --seed 42002 ^
    --output "%OUTDIR%\nbody_p9_M5_a400.json"

REM ============================================================================
REM Run 3: P9 M=5 M_earth, a=600 AU — B&B canonical
REM GPU 2 | ~300 particles × 100 Myr | Est. ~2-3 hours
REM ============================================================================
start "GPU2_P9_M5_a600" python "%SCRIPT%" ^
    --mode p9 ^
    --M_p9 5 --a_p9 600 --e_p9 0.3 --i_p9 15 ^
    --N 300 --tmax 100 ^
    --seed 42003 ^
    --output "%OUTDIR%\nbody_p9_M5_a600.json"

REM ============================================================================
REM Run 4: P9 M=10 M_earth, a=500 AU — heavier perturber
REM GPU 3 | ~300 particles × 100 Myr | Est. ~2-3 hours
REM ============================================================================
start "GPU3_P9_M10_a500" python "%SCRIPT%" ^
    --mode p9 ^
    --M_p9 10 --a_p9 500 --e_p9 0.3 --i_p9 15 ^
    --N 300 --tmax 100 ^
    --seed 42004 ^
    --output "%OUTDIR%\nbody_p9_M10_a500.json"

REM ============================================================================
REM Run 5: P9 M=10 M_earth, a=700 AU — distant heavy perturber
REM GPU 4 | ~300 particles × 100 Myr | Est. ~2-3 hours
REM ============================================================================
start "GPU4_P9_M10_a700" python "%SCRIPT%" ^
    --mode p9 ^
    --M_p9 10 --a_p9 700 --e_p9 0.3 --i_p9 15 ^
    --N 300 --tmax 100 ^
    --seed 42005 ^
    --output "%OUTDIR%\nbody_p9_M10_a700.json"

REM ============================================================================
REM Run 6: P9 M=15 M_earth, a=500 AU — very heavy perturber
REM GPU 5 | ~300 particles × 100 Myr | Est. ~2-3 hours
REM ============================================================================
start "GPU5_P9_M15_a500" python "%SCRIPT%" ^
    --mode p9 ^
    --M_p9 15 --a_p9 500 --e_p9 0.3 --i_p9 15 ^
    --N 300 --tmax 100 ^
    --seed 42006 ^
    --output "%OUTDIR%\nbody_p9_M15_a500.json"

echo ============================================================================
echo All 6 jobs launched. Monitor with:
echo   tasklist | findstr python
echo Output will appear in: %OUTDIR%
echo ============================================================================
