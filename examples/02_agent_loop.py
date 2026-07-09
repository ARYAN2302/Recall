"""
Example 2: Agent loop — using Recall inside an agent.

Shows the pattern: agent gets user input, generates response, user
corrects it, agent teaches the correction to Recall. Next time the
model handles similar input, the weights already know the answer.
"""
from recall import Recall


def fake_user_input() -> str:
    return "write a function to compute the fibonacci sequence"


def fake_user_correction() -> str:
    """User says: 'no, use the iterative form, with type hints.'"""
    return (
        "def fibonacci(n: int) -> int:\n"
        "    \"\"\"Compute the n-th Fibonacci number iteratively.\"\"\"\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a"
    )


def main():
    mem = Recall()

    print("=== Agent loop demo ===\n")

    # Turn 1: agent generates, user is unhappy
    user_input = fake_user_input()
    response_1 = mem.generate(user_input)
    print(f"User: {user_input}")
    print(f"Agent (turn 1): {response_1!r}\n")

    # User corrects
    correction = fake_user_correction()
    print(f"User: 'No, do it like this:\n{correction}'\n")

    # Agent teaches the correction — the weights change
    mem.remember(user_input, correction)

    # Turn 2: agent generates again — should match the correction style
    response_2 = mem.generate(user_input)
    print(f"Agent (turn 2): {response_2!r}\n")

    print(f"Status: {mem.status()}")


if __name__ == "__main__":
    main()
