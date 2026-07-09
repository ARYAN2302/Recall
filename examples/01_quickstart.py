"""
Example 1: Quickstart — the 5-line demo.

    from recall import Recall

    mem = Recall()
    print(mem.generate("write a function to sort a list"))  # bare function

    mem.remember("write a function to sort a list",
                 "def sort_list(lst: list) -> list:\n    return sorted(lst)")
    print(mem.generate("write a function to sort a list"))  # typed, clean

Run:
    python examples/01_quickstart.py
"""
from recall import Recall


def main():
    print("=== Recall quickstart ===\n")

    mem = Recall()

    print("1. Before any corrections:")
    out = mem.generate("write a function to sort a list")
    print(f"   {out!r}\n")

    print("2. Teaching: 'use type hints, return list'")
    mem.remember(
        "write a function to sort a list",
        "def sort_list(lst: list) -> list:\n    return sorted(lst)",
    )

    print("3. After correction:")
    out = mem.generate("write a function to sort a list")
    print(f"   {out!r}\n")

    print("4. Teaching: 'add docstrings too'")
    mem.remember(
        "write a function to sort a list",
        "def sort_list(lst: list) -> list:\n    \"\"\"Sort a list.\"\"\"\n    return sorted(lst)",
    )

    print("5. After second correction:")
    out = mem.generate("write a function to sort a list")
    print(f"   {out!r}\n")

    print("6. Status:")
    print(f"   {mem.status()}\n")


if __name__ == "__main__":
    main()
