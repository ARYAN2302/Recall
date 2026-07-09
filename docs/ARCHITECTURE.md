# Architecture

This document is for contributors. The README has the user-facing overview.

## The three layers

```
┌─────────────────────────────────────────────────────────┐
│  Public API (api.py)                                    │
│    Recall.remember(), Recall.generate(), Recall.status()│
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Backend (local.py OR modal_client.py)                  │
│    LocalBackend  →  in-process                          │
│    ModalBackend  →  remote Modal Functions              │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Core ML (trainer.py, avr.py, base.py, state.py)        │
│    hippocampus SFT → neocortex distillation → AVR       │
└─────────────────────────────────────────────────────────┘
```

The public API is one class with two methods. The backend layer is swappable. The core ML is shared between both backends.

## State flow

```
recall.remember(input, target)
   │
   ├─ 1. queue.add(input, target)                    [queue.py]
   │      persists to SQLite for audit + AVR probing
   │
   ├─ 2. learn_correction(model, tokenizer,           [trainer.py]
   │                       neo_state, pairs, config)
   │      ├─ snapshot neo_state → neo_snapshot
   │      ├─ reset_lora_to_peft_init(model)           [state.py]
   │      ├─ train_hippocampus(model, pairs)          [trainer.py]
   │      │    3 epochs SFT, fresh LoRA
   │      ├─ consolidate_to_neocortex(model, ...)     [trainer.py]
   │      │    KL-distill hippo→neo, 1 epoch, half LR
   │      └─ return (new_neo_state, {neo_snapshot, ...})
   │
   ├─ 3. set_lora_state(model, new_neo_state)         [state.py]
   │
   ├─ 4. (every N corrections) run_avr_loop(...)      [avr.py]
   │      ├─ eval_correction_ppls(model, corrections)
   │      ├─ verify_drift(current, best, threshold=1.15)
   │      ├─ if drifted:
   │      │     repair_toward_snapshot(model, neo_snapshot, α=0.1)
   │      │     re-verify
   │      │     repeat up to max_steps=10
   │      └─ update best_ppls
   │
   ├─ 5. queue.mark_trained(cid)
   │
   └─ 6. save snapshot to disk (safetensors)
```

## Why two streams?

Single-stream SFT (the naive baseline) updates the same LoRA for each correction. Each step partially overwrites the weights the prior correction installed. After 10 corrections, correction #1 is usually forgotten — that's catastrophic forgetting.

Two-stream training splits the work:

- **Hippocampus** is reset to PEFT init each correction. It learns the new correction in isolation, with no contamination from prior knowledge. Fast LR, fresh init, fast convergence.
- **Neocortex** is persistent. It never sees the correction directly — it learns to match the hippocampus's output distribution via KL distillation. Slow LR, persistent state, slow integration.

The key insight: distillation is a *softer* update than SFT. SFT minimizes CE loss against a hard target; distillation minimizes KL against a soft distribution. The soft target lets the neocortex integrate new knowledge without sharply overwriting the directions that prior corrections installed.

This is the complementary-learning-systems split biology uses. The names hippocampus and neocortex aren't metaphor — they're the literal architecture.

## Why AVR works

After consolidation, the neocortex may still drift on prior corrections (distillation isn't perfect). AVR catches this drift and repairs it.

**VERIFY**: compute PPL on each prior correction's `target`. If `PPL_now / PPL_best > 1.15`, that correction has drifted. PPL is the right signal because it directly measures how well the model predicts the correction's target — which is exactly what we want to preserve.

**REPAIR**: `θ ← (1-α)·θ + α·θ_snapshot` with α=0.1. This is a closed-form interpolation in weight space. It pulls the model toward the last known-good state for the drifted corrections. No gradients — we're not training, we're rewinding.

**Why closed-form?** Three reasons:
1. It's fast — no backward pass, no optimizer, just a weighted average of CPU tensors.
2. It's deterministic — same snapshot + same drift = same repair, every time.
3. It's safe — convex combination means we never overshoot. The repair is bounded.

The verify-repair loop runs until drift resolves or we hit the 10-step cap. In practice, drift usually resolves in 1-3 steps because α=0.1 is calibrated to undo ~10% of the drift per step.

## Why no replay buffer?

Replay (mixing old corrections into new training batches) is the standard CL baseline. It works, but it has costs:
- Memory grows O(N × buffer_per_task)
- Each new correction requires re-training on old data
- Privacy: you're storing user data forever

AVR avoids all three by using the *weights themselves* as the memory. The snapshot is the same size as the adapter (LoRA only — constant memory). Repair is closed-form, no re-training. And the snapshot is just weights, not user data.

The replay baseline in `eval/baselines.py` lets you verify this — Recall should match or beat Replay SFT on the forgetting curve, without the memory cost.

## Memory profile

Per correction, Recall holds in memory:
- Neocortex state: ~5MB (LoRA r=32 on Qwen3-0.6B, q_proj + v_proj)
- Hippocampus state: ~5MB (same size, transient — discarded after consolidation)
- Neo snapshot: ~5MB (the AVR repair target)
- Correction queue: ~1KB per correction (SQLite row)

Total memory growth per correction: ~1KB (everything else is reused). For 1000 corrections: ~1MB. For 100,000 corrections: ~100MB. Linear in correction count, no hidden multipliers.

Compared to replay-based methods (which grow at O(N × buffer_per_task × avg_pair_size)), this is 100-1000× smaller at scale.

## Compute profile

| Operation | Compute | When |
|---|---|---|
| Hippocampus SFT (3 epochs) | ~5s on T4 for a single correction | every `remember()` |
| Neocortex consolidation (1 epoch) | ~2s on T4 | every `remember()` |
| AVR verify | ~3s on T4 (50 PPL probes) | every N `remember()` |
| AVR repair step | ~0.1s (closed-form) | only if drifted |
| Inference | ~0.3s on T4 | every `generate()` |

Total `remember()` cost: ~7s normally, ~10s when AVR runs.

## Extending

The architecture is pluggable. The three ML phases (LEARN, VERIFY, REPAIR) each have a clean interface:

- **LEARN**: swap SFT for DPO/GRPO. The hippocampus can be trained with any method.
- **VERIFY**: swap PPL-ratio for KL divergence, Hessian trace, entropy. The detector just needs to return a drift signal.
- **REPAIR**: swap snapshot interpolation for subspace repair (load-bearing directions only). v2 research.

These slots are reserved in the codebase. v1 ships the validated configuration; v2 will explore alternatives without breaking the API.

## File-by-file map

| File | Role | Lines |
|---|---|---|
| `recall/api.py` | Public API. The `Recall` class. | ~85 |
| `recall/config.py` | All hyperparameters in one dataclass. | ~95 |
| `recall/base.py` | HF model + tokenizer + LoRA loader. | ~75 |
| `recall/state.py` | LoRA snapshot/restore/reset. | ~55 |
| `recall/trainer.py` | Hippocampus SFT + neocortex consolidation. | ~230 |
| `recall/avr.py` | PPL verify + snapshot repair. | ~210 |
| `recall/inference.py` | Generation helpers. | ~40 |
| `recall/queue.py` | SQLite correction queue. | ~120 |
| `recall/local.py` | In-process backend. | ~150 |
| `recall/modal_app.py` | Modal Function definitions. | ~115 |
| `recall/modal_client.py` | Modal client wrapper. | ~75 |
| `recall/cli.py` | `recall` CLI. | ~110 |
| `eval/corrections.py` | The 50-correction benchmark spec. | ~210 |
| `eval/accuracy.py` | check_tokens scoring. | ~55 |
| `eval/forgetting_curve.py` | The benchmark runner. | ~135 |
| `eval/baselines.py` | Naive SFT + Replay SFT. | ~150 |
| `eval/render.py` | Matplotlib chart. | ~95 |
