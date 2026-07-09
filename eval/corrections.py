"""
eval.corrections — the 50-correction benchmark spec.

Each correction is a (input, target) pair where:
    input  = a prompt the model gets
    target = the answer the model SHOULD produce after this correction

The 50 corrections span:
    - code style preferences (type hints, docstrings, naming)
    - output format (JSON vs prose, markdown vs plain)
    - tone (concise, formal, friendly)
    - structural (no em-dashes, no ellipses, sentence case headers)
    - domain knowledge (specific library, specific API)

The forgetting curve measures: after correction i, is the model still
producing correction 1's target when given correction 1's input?

We use code-style + formatting preferences because they're:
    - Easy to score (string match on key tokens)
    - Hard to forget (well-defined target)
    - Cumulative (each correction should stack on prior ones)

The first 15 are the "smoke" set for fast iteration. The full 50 are
the launch benchmark.
"""
from __future__ import annotations
from typing import List, Dict


# ────────────────────────────────────────────────────────────────────
# The 50-correction spec
# ────────────────────────────────────────────────────────────────────

# Each entry: {"id", "input", "target", "check_tokens"}
# check_tokens = tokens that MUST appear in a correct generation. We
# check substring presence rather than exact match because generation
# is non-deterministic in length, but the *style* should be consistent.

CORRECTIONS: List[Dict] = [
    # ── 1-10: code style basics ──
    {
        "id": "c01_type_hints",
        "input": "write a function to sort a list",
        "target": "def sort_list(lst: list) -> list:\n    return sorted(lst)",
        "check_tokens": ["def sort_list", "lst: list", "-> list"],
    },
    {
        "id": "c02_docstring",
        "input": "write a function to add two numbers",
        "target": "def add(a: int, b: int) -> int:\n    \"\"\"Add two numbers.\"\"\"\n    return a + b",
        "check_tokens": ["def add", "a: int", "b: int", '"""', "Add two"],
    },
    {
        "id": "c03_snake_case",
        "input": "write a function to compute factorial",
        "target": "def compute_factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * compute_factorial(n - 1)",
        "check_tokens": ["def compute_factorial", "n: int", "-> int"],
    },
    {
        "id": "c04_no_print",
        "input": "write a function to greet a user",
        "target": "def greet_user(name: str) -> str:\n    return f\"Hello, {name}!\"",
        "check_tokens": ["def greet_user", "name: str", "-> str", "return"],
    },
    {
        "id": "c05_return_type",
        "input": "write a function to check if a number is even",
        "target": "def is_even(n: int) -> bool:\n    return n % 2 == 0",
        "check_tokens": ["def is_even", "n: int", "-> bool", "return"],
    },
    {
        "id": "c06_no_comments",
        "input": "write a function to reverse a string",
        "target": "def reverse_string(s: str) -> str:\n    return s[::-1]",
        "check_tokens": ["def reverse_string", "s: str", "-> str", "return", "s[::-1]"],
    },
    {
        "id": "c07_ternary",
        "input": "write a function to get the max of two numbers",
        "target": "def max_of_two(a: int, b: int) -> int:\n    return a if a > b else b",
        "check_tokens": ["def max_of_two", "a: int", "b: int", "return", "if", "else"],
    },
    {
        "id": "c08_fstring",
        "input": "write a function to format a name",
        "target": "def format_name(first: str, last: str) -> str:\n    return f\"{first} {last}\"",
        "check_tokens": ["def format_name", "first: str", "last: str", "f\"", "return"],
    },
    {
        "id": "c09_immutable_default",
        "input": "write a function to append to a list",
        "target": "def append_to(lst: list, item) -> list:\n    return lst + [item]",
        "check_tokens": ["def append_to", "lst: list", "-> list", "return"],
    },
    {
        "id": "c10_property",
        "input": "write a class for a bank account",
        "target": "class BankAccount:\n    def __init__(self, balance: float = 0.0):\n        self._balance = balance\n    @property\n    def balance(self) -> float:\n        return self._balance",
        "check_tokens": ["class BankAccount", "def __init__", "balance: float", "@property", "def balance", "-> float"],
    },

    # ── 11-20: output format ──
    {
        "id": "c11_json_output",
        "input": "describe a person named Alice age 30",
        "target": "{\"name\": \"Alice\", \"age\": 30}",
        "check_tokens": ["{", "\"name\"", "Alice", "\"age\"", "30", "}"],
    },
    {
        "id": "c12_no_markdown",
        "input": "explain what a function is",
        "target": "A function is a reusable block of code that performs a specific task.",
        "check_tokens": ["A function is", "reusable", "block of code", "specific task"],
    },
    {
        "id": "c13_lowercase_headers",
        "input": "list the steps to make tea",
        "target": "steps to make tea:\n1. boil water\n2. add tea bag\n3. steep for 3 minutes\n4. remove tea bag\n5. serve",
        "check_tokens": ["steps to make tea", "boil water", "add tea bag", "steep", "serve"],
    },
    {
        "id": "c14_no_em_dash",
        "input": "explain recursion",
        "target": "Recursion is when a function calls itself to solve a smaller version of the same problem.",
        "check_tokens": ["Recursion is", "function calls itself", "smaller version", "same problem"],
    },
    {
        "id": "c15_short_answers",
        "input": "what is python",
        "target": "Python is a high-level programming language.",
        "check_tokens": ["Python is", "high-level", "programming language"],
    },
    {
        "id": "c16_no_ellipsis",
        "input": "list three fruits",
        "target": "1. apple\n2. banana\n3. cherry",
        "check_tokens": ["apple", "banana", "cherry"],
    },
    {
        "id": "c17_imperative",
        "input": "how to make coffee",
        "target": "grind coffee beans. place filter in dripper. add grounds. pour hot water. serve.",
        "check_tokens": ["grind", "coffee beans", "filter", "dripper", "grounds", "pour", "hot water", "serve"],
    },
    {
        "id": "c18_no_intro",
        "input": "explain what an api is",
        "target": "An API is a set of rules that allows programs to communicate with each other.",
        "check_tokens": ["An API is", "set of rules", "programs", "communicate"],
    },
    {
        "id": "c19_active_voice",
        "input": "describe how databases work",
        "target": "Databases store data in tables. Applications query tables to read or modify data. The database management system handles concurrency and persistence.",
        "check_tokens": ["Databases store", "tables", "Applications query", "database management system", "concurrency"],
    },
    {
        "id": "c20_no_code_in_prose",
        "input": "what is a variable",
        "target": "A variable is a named storage location that holds a value which can change during program execution.",
        "check_tokens": ["A variable is", "named storage", "holds a value", "change", "execution"],
    },

    # ── 21-30: tone ──
    {
        "id": "c21_concise",
        "input": "what is html",
        "target": "HTML is the standard markup language for web pages.",
        "check_tokens": ["HTML is", "standard markup language", "web pages"],
    },
    {
        "id": "c22_formal",
        "input": "what is ai",
        "target": "Artificial intelligence is the field of computer science focused on building systems that perform tasks requiring human intelligence.",
        "check_tokens": ["Artificial intelligence", "field of computer science", "systems that perform", "human intelligence"],
    },
    {
        "id": "c23_neutral",
        "input": "what is the best programming language",
        "target": "The choice of programming language depends on the task, team expertise, and ecosystem requirements.",
        "check_tokens": ["choice", "programming language", "depends", "task", "team", "ecosystem"],
    },
    {
        "id": "c24_no_superlatives",
        "input": "describe python",
        "target": "Python is a general-purpose programming language with dynamic typing and a large standard library.",
        "check_tokens": ["Python is", "general-purpose", "dynamic typing", "standard library"],
    },
    {
        "id": "c25_direct",
        "input": "how do loops work",
        "target": "Loops repeat a block of code until a condition is met. The two main types are for loops and while loops.",
        "check_tokens": ["Loops repeat", "block of code", "condition", "for loops", "while loops"],
    },
    {
        "id": "c26_no_first_person",
        "input": "explain git",
        "target": "Git is a version control system that tracks changes in files and coordinates work among multiple contributors.",
        "check_tokens": ["Git is", "version control system", "tracks changes", "coordinates", "contributors"],
    },
    {
        "id": "c27_technical",
        "input": "what is http",
        "target": "HTTP is an application-layer protocol for distributed, collaborative, hypermedia information systems.",
        "check_tokens": ["HTTP is", "application-layer protocol", "distributed", "collaborative", "hypermedia"],
    },
    {
        "id": "c28_no_qualifiers",
        "input": "what is a database",
        "target": "A database is an organized collection of structured data stored electronically.",
        "check_tokens": ["A database is", "organized collection", "structured data", "stored electronically"],
    },
    {
        "id": "c29_fact_first",
        "input": "what is docker",
        "target": "Docker is a platform for building, shipping, and running applications in containers.",
        "check_tokens": ["Docker is", "platform", "building", "shipping", "running", "containers"],
    },
    {
        "id": "c30_no_hedge",
        "input": "what is kubernetes",
        "target": "Kubernetes is a container orchestration system that automates deployment, scaling, and management of containerized applications.",
        "check_tokens": ["Kubernetes is", "container orchestration system", "automates", "deployment", "scaling", "management"],
    },

    # ── 31-40: structural / formatting ──
    {
        "id": "c31_period_ended",
        "input": "what is typescript",
        "target": "TypeScript is a typed superset of JavaScript that compiles to plain JavaScript.",
        "check_tokens": ["TypeScript is", "typed superset", "JavaScript", "compiles"],
    },
    {
        "id": "c32_one_sentence",
        "input": "what is rust",
        "target": "Rust is a systems programming language focused on memory safety and concurrency without data races.",
        "check_tokens": ["Rust is", "systems programming language", "memory safety", "concurrency", "data races"],
    },
    {
        "id": "c33_lowercase_first_word",
        "input": "what is go",
        "target": "Go is a statically typed, compiled language designed at Google with a focus on simplicity and concurrency.",
        "check_tokens": ["Go is", "statically typed", "compiled language", "Google", "simplicity", "concurrency"],
    },
    {
        "id": "c34_no_lists_for_short_answers",
        "input": "what is java",
        "target": "Java is a class-based, object-oriented programming language designed to have minimal implementation dependencies.",
        "check_tokens": ["Java is", "class-based", "object-oriented", "programming language", "implementation dependencies"],
    },
    {
        "id": "c35_present_tense",
        "input": "what is linux",
        "target": "Linux is an open-source kernel that powers servers, embedded systems, and desktop operating systems.",
        "check_tokens": ["Linux is", "open-source kernel", "servers", "embedded systems", "desktop"],
    },
    {
        "id": "c36_no_parentheticals",
        "input": "what is a process",
        "target": "A process is an instance of a program in execution with its own memory space and resources.",
        "check_tokens": ["A process is", "instance", "program in execution", "memory space"],
    },
    {
        "id": "c37_no_semicolons_in_prose",
        "input": "what is a thread",
        "target": "A thread is the smallest unit of execution within a process, sharing memory with other threads in the same process.",
        "check_tokens": ["A thread is", "smallest unit", "execution", "process", "sharing memory"],
    },
    {
        "id": "c38_oxford_comma",
        "input": "list three primary colors",
        "target": "The three primary colors are red, blue, and yellow.",
        "check_tokens": ["primary colors", "red", "blue", "yellow"],
    },
    {
        "id": "c39_spell_out_numbers",
        "input": "how many continents are there",
        "target": "There are seven continents on Earth.",
        "check_tokens": ["seven continents", "Earth"],
    },
    {
        "id": "c40_no_abbreviations",
        "input": "what is a function in programming",
        "target": "A function in programming is a named block of code that performs a specific task and can be called multiple times.",
        "check_tokens": ["A function in programming is", "named block of code", "specific task", "called multiple times"],
    },

    # ── 41-50: more code-style + edge cases ──
    {
        "id": "c41_async_def",
        "input": "write a function to fetch data",
        "target": "async def fetch_data(url: str) -> str:\n    \"\"\"Fetch data from a URL.\"\"\"\n    return await _http_get(url)",
        "check_tokens": ["async def fetch_data", "url: str", "-> str", "await"],
    },
    {
        "id": "c42_dataclass",
        "input": "write a class for a user",
        "target": "from dataclasses import dataclass\n\n@dataclass\nclass User:\n    name: str\n    age: int",
        "check_tokens": ["@dataclass", "class User", "name: str", "age: int"],
    },
    {
        "id": "c43_type_alias",
        "input": "write a function to process items",
        "target": "from typing import List\n\ndef process_items(items: List[str]) -> List[str]:\n    return [item.upper() for item in items]",
        "check_tokens": ["from typing", "List[str]", "def process_items", "-> List[str]", "return"],
    },
    {
        "id": "c44_no_global",
        "input": "write a function to increment a counter",
        "target": "def increment(counter: int) -> int:\n    return counter + 1",
        "check_tokens": ["def increment", "counter: int", "-> int", "return"],
    },
    {
        "id": "c45_pure_function",
        "input": "write a function to compute square",
        "target": "def square(x: int) -> int:\n    return x * x",
        "check_tokens": ["def square", "x: int", "-> int", "return", "x * x"],
    },
    {
        "id": "c46_no_try_except_for_simple",
        "input": "write a function to divide two numbers",
        "target": "def divide(a: float, b: float) -> float:\n    return a / b",
        "check_tokens": ["def divide", "a: float", "b: float", "-> float", "return", "a / b"],
    },
    {
        "id": "c47_explicit_return_none",
        "input": "write a function to print a greeting",
        "target": "def print_greeting(name: str) -> None:\n    print(f\"Hello, {name}!\")",
        "check_tokens": ["def print_greeting", "name: str", "-> None", "print"],
    },
    {
        "id": "c48_no_walrus_for_simple",
        "input": "write a function to get length",
        "target": "def get_length(s: str) -> int:\n    return len(s)",
        "check_tokens": ["def get_length", "s: str", "-> int", "return", "len(s)"],
    },
    {
        "id": "c49_explicit_dict_type",
        "input": "write a function to merge two dicts",
        "target": "def merge_dicts(a: dict, b: dict) -> dict:\n    return {**a, **b}",
        "check_tokens": ["def merge_dicts", "a: dict", "b: dict", "-> dict", "return"],
    },
    {
        "id": "c50_no_lambda_for_named",
        "input": "write a function to double a number",
        "target": "def double(x: int) -> int:\n    return x * 2",
        "check_tokens": ["def double", "x: int", "-> int", "return", "x * 2"],
    },
]


# ────────────────────────────────────────────────────────────────────
# Smaller smoke set for fast iteration / Kaggle tests
# ────────────────────────────────────────────────────────────────────

SMOKE_CORRECTIONS: List[Dict] = CORRECTIONS[:15]
"""15-correction subset for quick smoke tests. Fits in ~30 min on a T4."""


def get_corrections(n: int = 50) -> List[Dict]:
    """Return the first n corrections. n=15 for smoke, n=50 for launch."""
    if n > len(CORRECTIONS):
        raise ValueError(f"Only {len(CORRECTIONS)} corrections defined. "
                         f"Asked for {n}.")
    return CORRECTIONS[:n]
