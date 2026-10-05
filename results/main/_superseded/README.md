Superseded runs (kept for transparency, ignored by the dashboard and report).

- 20261005-110442_switch_cf1_s1337 (E2, cf=1.0): GPU dropped into a low-power state during the run
  (14k tok/s vs ~113-132k for the other E2 runs; 2158 s wall). Loss was valid (val 1.7325) but
  throughput was not comparable. Re-run on 2026-10-05 after the user confirmed plugged in + Turbo.
- 20261005-114954_profile (E5): profile_layer.py flagged power_state_changed (memory clock
  14126/11126/9001/11126 MHz during the run). Re-run on 2026-10-05.
- 20261005-114948_placement (E6): valid, but ran before the accepted E5 re-run, so it used the 50 TFLOP/s config default for device compute. Re-run on 2026-10-05 to use the measured roof (57.3 TFLOP/s). Device-load and all-to-all byte numbers are unaffected; only absolute step-time estimates change.
