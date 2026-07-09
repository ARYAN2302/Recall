"""
Example 3: Custom base model.

Recall works with any HF causal LM. This example uses a smaller model
for fast local iteration. Swap in LFM2.5-350M, SmolLM-135M, etc.
"""
from recall import Recall, RecallConfig


def main():
    print("=== Custom base model ===\n")

    config = RecallConfig(
        model_id="Qwen/Qwen3-0.6B",  # swap for any HF causal LM
        lora_rank=16,
        lora_targets=("q_proj", "v_proj"),
        train_epochs=2,
        avr_every_n=3,
        max_new_tokens=48,
        data_dir="./recall_data_custom",
    )

    mem = Recall(config=config)

    print(f"Model: {config.model_id}")
    print(f"LoRA rank: {config.lora_rank}, targets: {config.lora_targets}\n")

    print("Before:")
    print(f"  {mem.generate('what is 2 + 2')!r}\n")

    mem.remember(
        "what is 2 + 2",
        "The answer is 4.",
    )

    print("After:")
    print(f"  {mem.generate('what is 2 + 2')!r}\n")

    print(f"Status: {mem.status()}")


if __name__ == "__main__":
    main()
