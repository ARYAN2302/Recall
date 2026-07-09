"""
eval — the benchmarks that prove Recall works.

forgetting_curve  — the viral chart. After each new correction, evaluate
                    accuracy on correction #1. Plot accuracy vs correction
                    index. Compare Recall (with AVR + two-stream) vs naive
                    SFT (no protection) vs optionally Mem0/Letta/Zep.
corrections       — the 50-correction spec. Each correction is a preference
                    the model should learn.
accuracy          — scoring helpers for evaluating whether a generation
                    matches a target.
render            — matplotlib chart rendering.
baselines         — naive SFT baseline (no AVR, no two-stream). Mem0/Letta
                    wrappers come later.
"""
