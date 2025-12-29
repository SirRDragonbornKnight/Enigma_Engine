from __future__ import annotations

import argparse
import random
from pathlib import Path


def _read_questions(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {
        line.strip().casefold()
        for line in text.splitlines()
        if line.strip().startswith("Q:")
    }


def _append_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    existing = path.read_text(encoding="utf-8")

    with path.open("a", encoding="utf-8", newline="\n") as f:
        if not existing.endswith("\n"):
            f.write("\n")
        if existing and not existing.endswith("\n\n"):
            f.write("\n")

        for idx, (q, a) in enumerate(pairs):
            f.write(q.rstrip() + "\n")
            f.write(a.rstrip() + "\n")
            if idx < len(pairs) - 1:
                f.write("\n")


def _generate_pairs(existing_q: set[str], count: int, seed: int) -> list[tuple[str, str]]:
    random.seed(seed)

    # Broad, safe topic pools. These are intentionally general to avoid unsafe or overly specific instructions.
    # Expanded to comfortably support generating thousands of unique questions.
    topics = [
        # Programming / software
        "variables",
        "functions",
        "classes",
        "modules",
        "packages",
        "unit testing",
        "integration testing",
        "debugging",
        "logging",
        "exceptions",
        "type hints",
        "virtual environments",
        "dependency management",
        "API design",
        "REST",
        "HTTP status codes",
        "JSON",
        "CSV",
        "regular expressions",
        "time complexity",
        "space complexity",
        "recursion",
        "iteration",
        "data structures",
        "lists",
        "dictionaries",
        "sets",
        "stacks",
        "queues",
        "immutability",
        "idempotency",
        "caching",
        "serialization",
        "authentication",
        "authorization",
        "encryption",
        "hashing",
        "git",
        "code review",
        "continuous integration",
        "configuration management",
        "feature flags",
        "error handling",
        "input validation",
        "rate limiting",
        "observability",
        "monitoring",
        "latency",
        "throughput",
        "scalability",
        "availability",
        "reliability",
        # Math
        "prime numbers",
        "fractions",
        "percentages",
        "ratios",
        "proportions",
        "probability",
        "mean",
        "median",
        "mode",
        "variance",
        "standard deviation",
        "linear equations",
        "quadratic equations",
        "logarithms",
        "exponents",
        "vectors",
        "matrices",
        "statistics",
        "sampling bias",
        "correlation vs causation",
        # Science
        "photosynthesis",
        "cell respiration",
        "DNA",
        "gravity",
        "electricity",
        "magnetism",
        "atoms",
        "molecules",
        "states of matter",
        "plate tectonics",
        "the water cycle",
        "weather",
        "climate",
        "ecosystems",
        "the scientific method",
        "hypotheses",
        "controlled experiments",
        "renewable energy",
        # Writing / communication
        "thesis statements",
        "topic sentences",
        "active voice",
        "passive voice",
        "conciseness",
        "tone",
        "audience",
        "outlining",
        "editing",
        "proofreading",
        "summarizing",
        "paraphrasing",
        "argument structure",
        "evidence",
        "counterarguments",
        # Life skills
        "time management",
        "habit building",
        "stress management",
        "sleep hygiene",
        "meal planning",
        "budgeting basics",
        "goal setting",
        "communication skills",
        "conflict resolution",
        "decision making",
        "learning faster",
        "note taking",
        "prioritization",
        "planning",
        "motivation",
    ]

    definitions = {
        "variables": "named containers that hold values so you can reuse them",
        "functions": "reusable blocks of code that take inputs and return outputs",
        "classes": "blueprints for objects with data (attributes) and behavior (methods)",
        "modules": "files that group related code so it can be imported",
        "packages": "collections of modules distributed together",
        "unit testing": "small tests that verify individual functions behave correctly",
        "debugging": "finding and fixing the cause of incorrect behavior",
        "logging": "recording structured messages about what a program is doing",
        "exceptions": "signals that something unexpected happened, which you can handle",
        "type hints": "annotations that document expected types and improve tooling",
        "virtual environments": "isolated Python environments so dependencies don’t conflict",
        "dependency management": "tracking and pinning libraries so builds are reproducible",
        "API design": "designing clear inputs/outputs for a service or library",
        "REST": "a web API style using resources and HTTP methods",
        "JSON": "a text format for structured data using objects and arrays",
        "CSV": "a simple table format where columns are separated by commas",
        "regular expressions": "patterns for matching and extracting text",
        "time complexity": "how runtime grows as input size grows",
        "space complexity": "how memory use grows as input size grows",
        "recursion": "solving a problem by calling the same function on smaller subproblems",
        "iteration": "repeating steps in a loop until a condition is met",
        "data structures": "ways to organize data for efficient operations",
        "lists": "ordered collections of items",
        "dictionaries": "key-value mappings for fast lookup",
        "sets": "collections of unique items",
        "stacks": "last-in-first-out collections",
        "queues": "first-in-first-out collections",
        "immutability": "not changing data in-place after it’s created",
        "idempotency": "doing the same operation multiple times yields the same result",
        "caching": "storing results to avoid recomputing them",
        "serialization": "converting data into a storable/transmittable format",
        "authentication": "proving who you are",
        "authorization": "deciding what you’re allowed to do",
        "encryption": "scrambling data so only authorized parties can read it",
        "hashing": "mapping data to a fixed-size fingerprint",
        "prime numbers": "whole numbers > 1 with exactly two divisors: 1 and themselves",
        "fractions": "numbers representing parts of a whole (numerator/denominator)",
        "percentages": "fractions out of 100",
        "ratios": "comparisons between quantities",
        "probability": "how likely an event is (0 to 1)",
        "mean": "the average (sum divided by count)",
        "median": "the middle value after sorting",
        "mode": "the most frequent value",
        "variance": "how spread out values are (average squared distance from mean)",
        "standard deviation": "the square root of variance (spread in original units)",
        "linear equations": "equations where variables have power 1 (like y = mx + b)",
        "quadratic equations": "equations with a squared term (ax^2 + bx + c = 0)",
        "logarithms": "the inverse of exponentiation",
        "exponents": "a compact way to represent repeated multiplication",
        "vectors": "quantities with magnitude and direction",
        "matrices": "rectangular arrays of numbers used in linear algebra",
        "photosynthesis": "how plants convert light, water, and CO₂ into sugar and oxygen",
        "cell respiration": "how cells turn sugar into usable energy (ATP)",
        "DNA": "the molecule that stores genetic instructions",
        "gravity": "the attraction between masses",
        "electricity": "the movement of electric charge",
        "magnetism": "forces related to magnetic fields and moving charges",
        "atoms": "the basic building blocks of matter",
        "molecules": "two or more atoms bonded together",
        "states of matter": "forms like solid, liquid, gas, and plasma",
        "plate tectonics": "the movement of Earth’s crustal plates",
        "the water cycle": "water moving via evaporation, condensation, and precipitation",
        "weather": "short-term atmospheric conditions",
        "climate": "long-term patterns of weather",
        "ecosystems": "communities of organisms interacting with their environment",
        "thesis statements": "a sentence stating the main claim or argument",
        "topic sentences": "sentences stating the main idea of a paragraph",
        "active voice": "the subject performs the action",
        "passive voice": "the subject receives the action",
        "conciseness": "clear writing with no unnecessary words",
        "tone": "the attitude a piece of writing conveys",
        "audience": "the intended readers and their needs",
        "outlining": "planning structure before drafting",
        "editing": "improving structure and clarity after drafting",
        "proofreading": "fixing surface errors like typos",
        "summarizing": "restating main points briefly",
        "paraphrasing": "rewriting ideas in your own words",
        "time management": "choosing what to do and when",
        "habit building": "creating routines by shaping cues and rewards",
        "stress management": "methods to reduce and cope with stress",
        "sleep hygiene": "habits that improve sleep quality",
        "meal planning": "deciding meals ahead of time",
        "budgeting basics": "planning how to allocate income",
        "goal setting": "turning an outcome into specific steps",
        "communication skills": "habits that make messages clear and respectful",
        "conflict resolution": "ways to address disagreements constructively",
        "decision making": "choosing among options using tradeoffs",
        "learning faster": "methods that improve retention and skill acquisition",
        "note taking": "capturing key ideas for later review",
    }

    why_map = {
        "unit testing": "it catches bugs early and makes refactoring safer",
        "type hints": "they reduce confusion and help tools catch mistakes",
        "logging": "it helps you diagnose issues without guessing",
        "sleep hygiene": "better sleep improves mood, focus, and health",
        "budgeting basics": "it reduces surprises and supports savings goals",
        "time complexity": "it helps you pick solutions that scale",
        "communication skills": "they reduce misunderstandings and improve relationships",
        "encryption": "it protects confidentiality when data is stored or transmitted",
        "hashing": "it helps verify integrity and store passwords safely (with proper methods)",
    }

    templates: list[tuple[str, str]] = [
        ("Q: What is {topic} in simple terms?", "A: {topic} is {defn}."),
        ("Q: Can you define {topic} briefly?", "A: {topic} is {defn}."),
        ("Q: Why is {topic} important?", "A: {topic} matters because {why}."),
        ("Q: When would you use {topic}?", "A: You’d use {topic} when you want {use_case}."),
        ("Q: What is a simple example of {topic}?", "A: A simple example is {example}."),
        (
            "Q: What are 3 common mistakes people make with {topic}?",
            "A: Common mistakes include: 1) skipping fundamentals 2) not practicing consistently 3) not reviewing results and adjusting.",
        ),
        (
            "Q: What’s a beginner-friendly way to practice {topic}?",
            "A: Start with a simple example, practice a little each day, and review mistakes to improve.",
        ),
        (
            "Q: How can I explain {topic} to a beginner?",
            "A: Use plain language, give one concrete example, and connect it to something familiar.",
        ),
        (
            "Q: What’s a good mental model for {topic}?",
            "A: Think of it like {mental_model}, which helps you remember how it works.",
        ),
        (
            "Q: What’s one misconception about {topic}?",
            "A: A common misconception is {misconception}. A better view is {better_view}.",
        ),
        (
            "Q: What are the pros and cons of {topic}?",
            "A: Pros: {pro1}; {pro2}. Cons: {con1}; {con2}.",
        ),
        (
            "Q: What are the basic steps to get started with {topic}?",
            "A: Start by learning the fundamentals, try one small example, then practice and review your results.",
        ),
    ]

    diff_pairs = [
        ("authentication", "authorization"),
        ("encryption", "hashing"),
        ("mean", "median"),
        ("weather", "climate"),
        ("active voice", "passive voice"),
        ("list", "dictionary"),
        ("set", "dictionary"),
    ]

    diff_defs = {
        "authentication": "proving who you are (identity)",
        "authorization": "checking what you can access (permissions)",
        "encryption": "transforming data so it’s unreadable without a key",
        "hashing": "creating a one-way fingerprint of data",
        "mean": "the average",
        "median": "the middle after sorting",
        "weather": "short-term conditions",
        "climate": "long-term patterns",
        "active voice": "the subject does the action",
        "passive voice": "the subject receives the action",
        "list": "an ordered collection",
        "dictionary": "a key-value mapping",
        "set": "a collection of unique values",
    }

    # Deterministic-ish generation (shuffle order with seed, then iterate). This scales well
    # and avoids getting stuck with rejection sampling when the file already contains many items.
    topics_shuffled = topics[:]
    random.shuffle(topics_shuffled)
    templates_shuffled = templates[:]
    random.shuffle(templates_shuffled)
    diff_pairs_shuffled = diff_pairs[:]
    random.shuffle(diff_pairs_shuffled)

    # Domain-term combos create a large space of unique, generally-useful Q&A
    # without relying on specific trivia.
    domains = [
        "software engineering",
        "Python programming",
        "web development",
        "data analysis",
        "machine learning",
        "cybersecurity",
        "cloud computing",
        "databases",
        "networking",
        "DevOps",
        "product management",
        "project management",
        "technical writing",
        "learning and studying",
        "personal finance",
        "career growth",
        "team communication",
        "customer support",
        "home organization",
        "cooking",
        "fitness",
        "sleep",
        "stress management",
        "habit building",
        "public speaking",
    ]

    terms = [
        "clarity",
        "correctness",
        "simplicity",
        "complexity",
        "tradeoffs",
        "scope",
        "constraints",
        "assumptions",
        "risk",
        "quality",
        "consistency",
        "reliability",
        "availability",
        "performance",
        "latency",
        "throughput",
        "scalability",
        "maintainability",
        "readability",
        "testability",
        "observability",
        "monitoring",
        "metrics",
        "debugging",
        "documentation",
        "prioritization",
        "planning",
        "estimation",
        "alignment",
        "feedback",
        "iteration",
        "execution",
        "habits",
        "motivation",
        "discipline",
        "focus",
        "attention",
        "stress",
        "recovery",
        "sleep quality",
        "nutrition",
        "budgeting",
        "saving",
        "spending",
        "communication",
        "boundaries",
        "empathy",
        "conflict",
        "decision making",
        "problem solving",
        "learning",
        "practice",
        "review",
        "memory",
        "note taking",
        "research",
        "editing",
        "tone",
        "audience",
    ]

    term_definitions = {
        "latency": "the delay between a request and a response",
        "throughput": "how much work gets done per unit of time",
        "scalability": "how well something handles growth in load",
        "observability": "how well you can understand a system from its outputs",
        "metrics": "numbers you track to understand performance and outcomes",
        "alignment": "shared understanding of goals, priorities, and expectations",
        "boundaries": "clear limits on what you will and won’t do",
        "sleep quality": "how restorative your sleep is, not just duration",
    }

    domain_templates: list[tuple[str, str]] = [
        (
            "Q: In {domain}, what does {term} mean?",
            "A: In {domain}, {term} means {term_defn}.",
        ),
        (
            "Q: In {domain}, why does {term} matter?",
            "A: It matters because improving {term} usually improves outcomes like quality, speed, or confidence in your results.",
        ),
        (
            "Q: What’s a simple way to improve {term} in {domain}?",
            "A: Start small: pick one measurable goal, remove the biggest obvious bottleneck, and iterate based on feedback.",
        ),
        (
            "Q: What is one common tradeoff involving {term} in {domain}?",
            "A: A common tradeoff is balancing {term} with constraints like time, effort, cost, or simplicity.",
        ),
        (
            "Q: What’s a beginner mistake related to {term} in {domain}?",
            "A: A common mistake is optimizing {term} too early without first clarifying the goal and measuring the baseline.",
        ),
    ]

    use_cases = [
        "make your code clearer and easier to maintain",
        "avoid common errors and edge cases",
        "organize information for quick lookup",
        "communicate ideas more clearly",
        "analyze data or solve a problem",
        "reason about tradeoffs in a solution",
    ]

    mental_models = [
        "a toolbox with labeled compartments",
        "a recipe with steps and ingredients",
        "a map that helps you navigate choices",
        "a checklist that prevents missed steps",
        "a filing cabinet that keeps things organized",
        "a set of rules for consistent decisions",
    ]

    misconceptions = [
        ("you must memorize everything", "practice and feedback matter more than memorization"),
        ("bigger is always better", "tradeoffs matter: simplicity can beat complexity"),
        ("there is one best approach", "different contexts need different approaches"),
        ("mistakes mean failure", "mistakes are information that guide improvement"),
        ("speed matters most", "correctness and clarity often matter more, then you optimize"),
    ]

    pros = [
        "it can improve clarity",
        "it can reduce repeated work",
        "it can make results more consistent",
        "it can make debugging easier",
    ]
    cons = [
        "it may add upfront effort",
        "it can be misused if overdone",
        "it might introduce complexity if applied blindly",
        "it can require practice to apply well",
    ]

    examples = {
        "JSON": "storing settings as {\"theme\": \"dark\", \"fontSize\": 14}",
        "CSV": "saving a table as name,age on each line",
        "unit testing": "testing a function with a few known inputs and outputs",
        "budgeting basics": "splitting income into needs, wants, and savings",
        "time management": "planning 3 priorities for the day and timeboxing them",
    }

    created: list[tuple[str, str]] = []
    created_keys: set[str] = set()

    def add_if_unique(q: str, a: str) -> None:
        q_key = q.strip().casefold()
        if q_key in existing_q:
            return
        if q_key in created_keys:
            return
        created.append((q, a))
        created_keys.add(q_key)

    # 1) Generate comparison questions first (limited, but good variety).
    for a, b in diff_pairs_shuffled:
        if len(created) >= count:
            break
        q = f"Q: What is the difference between {a} and {b}?"
        a_defn = diff_defs.get(a, f"a concept related to {a}")
        b_defn = diff_defs.get(b, f"a concept related to {b}")
        a_line = (
            f"A: {a} is {a_defn}. {b} is {b_defn}. The key difference is what it’s for and how it’s used in practice."
        )
        add_if_unique(q, a_line)

    # 2) Cross product of (topic, template) for scalable unique question text.
    for topic in topics_shuffled:
        if len(created) >= count:
            break

        defn = definitions.get(topic, f"a concept related to {topic}")
        why = why_map.get(topic, "it helps you understand and apply ideas more effectively")
        use_case = random.choice(use_cases)
        mental_model = random.choice(mental_models)
        misconception, better_view = random.choice(misconceptions)
        pro1 = random.choice(pros)
        pro2 = random.choice([p for p in pros if p != pro1])
        con1 = random.choice(cons)
        con2 = random.choice([c for c in cons if c != con1])
        example = examples.get(topic, f"using {topic} in a small, real-world scenario")

        for t_q, t_a in templates_shuffled:
            if len(created) >= count:
                break
            q = t_q.format(topic=topic)
            a_line = t_a.format(
                topic=topic,
                defn=defn,
                why=why,
                use_case=use_case,
                mental_model=mental_model,
                misconception=misconception,
                better_view=better_view,
                pro1=pro1,
                pro2=pro2,
                con1=con1,
                con2=con2,
                example=example,
            )
            add_if_unique(q, a_line)

    # 3) High-capacity generation: cross product of (domain, term, template).
    # This is the main driver to reach large totals like 10,000.
    domains_shuffled = domains[:]
    random.shuffle(domains_shuffled)
    terms_shuffled = terms[:]
    random.shuffle(terms_shuffled)
    domain_templates_shuffled = domain_templates[:]
    random.shuffle(domain_templates_shuffled)

    for domain in domains_shuffled:
        if len(created) >= count:
            break
        for term in terms_shuffled:
            if len(created) >= count:
                break
            term_defn = term_definitions.get(term, f"a key concept related to {term}")
            for t_q, t_a in domain_templates_shuffled:
                if len(created) >= count:
                    break
                q = t_q.format(domain=domain, term=term)
                a_line = t_a.format(domain=domain, term=term, term_defn=term_defn)
                add_if_unique(q, a_line)

    if len(created) < count:
        raise RuntimeError(
            f"Could only generate {len(created)} unique pairs; need {count}. Try a smaller chunk or expand topic pools."
        )

    return created


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append a chunk of unique Q&A pairs to a training file (no repeated questions)."
    )
    parser.add_argument(
        "--file",
        default=str(
            Path(
                r"c:\Users\sirkn_gbhnunq\Documents\GitHub\enigma_engine_sacrifice_1\data\your_training_data.txt"
            )
        ),
        help="Path to training data file.",
    )
    parser.add_argument("--add", type=int, default=500, help="How many new pairs to add.")
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="If set, add enough pairs to reach this many unique questions total.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed.")

    args = parser.parse_args()
    file_path = Path(args.file)

    existing_q = _read_questions(file_path)
    before = len(existing_q)

    to_add = args.add
    if args.target_total is not None:
        to_add = max(0, args.target_total - before)
    if to_add == 0:
        print(f"Before unique questions: {before}")
        print("Added pairs: 0")
        print(f"After unique questions: {before}")
        print(f"Remaining to 10000: {max(0, 10000 - before)}")
        return 0

    pairs = _generate_pairs(existing_q=existing_q, count=to_add, seed=args.seed)
    _append_pairs(file_path, pairs)

    after = len(_read_questions(file_path))

    print(f"Before unique questions: {before}")
    print(f"Added pairs: {len(pairs)}")
    print(f"After unique questions: {after}")
    print(f"Remaining to 10000: {max(0, 10000 - after)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
