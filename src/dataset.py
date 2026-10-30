import random
from typing import List, Tuple


_INDUCTION_WORD_POOL = [
    "dog", "cat", "sun", "moon", "tree", "car", "book", "chair", "river", "mountain",
    "apple", "table", "door", "window", "cloud", "star", "fish", "bird", "house", "road",
    "king", "queen", "child", "friend", "city", "forest", "island", "garden", "bridge", "castle",
    "ocean", "desert", "valley", "farmer", "doctor", "teacher", "singer", "painter", "soldier", "sailor",
    "lion", "tiger", "horse", "sheep", "wolf", "eagle", "snake", "rabbit", "mouse", "bear",
    "coffee", "bread", "cheese", "sugar", "salt", "water", "milk", "wine", "fire", "stone",
    "silver", "gold", "iron", "glass", "paper", "cotton", "wood", "sand", "snow", "rain",
    "music", "dance", "story", "dream", "voice", "light", "shadow", "color", "shape", "sound",
]


def _build_word_pool(tokenizer) -> List[str]:
    """Keep only words that tokenize to exactly one token when preceded by
    a space (the standard GPT-2 BPE encoding of a word following a space).
    This guarantees the induction task's answer is a single, unambiguous
    token rather than depending on an unverified assumption about the
    tokenizer's vocabulary."""
    pool = []
    for word in _INDUCTION_WORD_POOL:
        ids = tokenizer.encode(" " + word)
        if len(ids) == 1:
            pool.append(word)
    if len(pool) < 8:
        raise ValueError(
            f"Only {len(pool)} words in the induction word pool tokenize to a single "
            "token for this tokenizer; need at least 8 to build 4-word sequences."
        )
    return pool


def get_induction_tasks(tokenizer, n: int = 200, seed: int = 0) -> Tuple[List[str], List[str]]:
    """
    Programmatically generate an induction/copying task with real
    baseline headroom (unlike the arithmetic/GSM8K tasks in this repo,
    which floor at 0% for GPT-2/DistilGPT2).

    Each example has the form:
        "w1 w2 w3 w4 . w1 w2 w3" -> " w4"
    where w1..w4 are four distinct words drawn from a pool of words
    verified to tokenize as a single BPE token. The model must copy the
    token that followed the first occurrence of the repeated 3-gram
    "w1 w2 w3" (a minimal induction-head task; see Olsson et al. 2022 on
    induction heads for the general phenomenon).

    Every generated example is verified via `src.metrics.tokenize_qa` to
    have a clean token-level prefix boundary between question and answer
    before being included, since BPE merges are not always prefix-stable
    across the question/answer boundary.

    Returns (questions, answers), length <= n (examples that fail the
    tokenization-prefix check are dropped rather than silently kept).
    """
    from src.metrics import is_valid_qa

    pool = _build_word_pool(tokenizer)
    rng = random.Random(seed)

    questions, answers = [], []
    attempts = 0
    max_attempts = n * 20
    while len(questions) < n and attempts < max_attempts:
        attempts += 1
        w1, w2, w3, w4 = rng.sample(pool, 4)
        question = f"{w1} {w2} {w3} {w4} . {w1} {w2} {w3}"
        answer = f" {w4}"
        if is_valid_qa(tokenizer, question, answer):
            questions.append(question)
            answers.append(answer)

    if len(questions) < n:
        raise RuntimeError(
            f"Only generated {len(questions)}/{n} valid induction examples after "
            f"{max_attempts} attempts."
        )
    return questions, answers


def get_arithmetic_control_tasks(
    tokenizer, n: int = 200, seed: int = 0, max_operand: int = 20
) -> Tuple[List[str], List[str]]:
    """
    Programmatically generate an addition task used ONLY as an off-target
    specificity control (never for head selection): "Q: a+b=" -> " sum".
    Operands are drawn uniformly from [1, max_operand]. Like
    `get_induction_tasks`, every example is verified to have a clean
    token-level question/answer boundary before being included.

    Returns (questions, answers), length <= n.
    """
    from src.metrics import is_valid_qa

    rng = random.Random(seed)
    questions, answers = [], []
    attempts = 0
    max_attempts = n * 20
    seen = set()
    while len(questions) < n and attempts < max_attempts:
        attempts += 1
        a = rng.randint(1, max_operand)
        b = rng.randint(1, max_operand)
        if (a, b) in seen:
            continue
        question = f"Q: {a}+{b}="
        answer = f" {a + b}"
        if is_valid_qa(tokenizer, question, answer):
            seen.add((a, b))
            questions.append(question)
            answers.append(answer)

    if len(questions) < n:
        raise RuntimeError(
            f"Only generated {len(questions)}/{n} valid arithmetic-control examples "
            f"after {max_attempts} attempts (max_operand={max_operand} may be too small "
            "to produce that many unique pairs)."
        )
    return questions, answers


# Ordered sequences for the format-matched control task. Successor-style
# continuation of a known ordering (Gould et al., 2024, "Successor Heads")
# needs no in-context copy: the answer never appears in the prompt.
_ORDERED_SEQUENCES = {
    "numerals": [str(i) for i in range(1, 100)],
    "months": ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"],
    "letters": [chr(c) for c in range(ord("A"), ord("Z") + 1)],
    "number_words": ["one", "two", "three", "four", "five", "six", "seven", "eight",
                     "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                     "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"],
}


def get_sequence_continuation_tasks(tokenizer) -> Tuple[List[str], List[str], List[str]]:
    """
    Format-matched control for the copying task: the same surface template
    "x1 x2 x3 x4 . x5 x6 x7" -> " x8", but x1..x8 are eight consecutive
    items of an ordered sequence (numerals 1-99, months, capital letters,
    number words one-twenty), so the answer is never present in context.

    Deterministic: every non-wrapping window of every sequence is used once
    (no sampling, no seed). Items that are not a single token with a
    leading space are removed from their sequence before windowing, and
    examples failing the token-prefix check are dropped, as in
    `get_induction_tasks`.

    Returns (questions, answers, families).
    """
    from src.metrics import is_valid_qa

    questions, answers, families = [], [], []
    for family, items in _ORDERED_SEQUENCES.items():
        if any(len(tokenizer.encode(" " + w)) != 1 for w in items):
            # dropping an inner item would make windows non-consecutive,
            # so keep only the longest run of single-token items
            runs, run = [], []
            for w in items:
                if len(tokenizer.encode(" " + w)) == 1:
                    run.append(w)
                else:
                    runs.append(run)
                    run = []
            runs.append(run)
            items = max(runs, key=len)
        for i in range(len(items) - 7):
            w = items[i:i + 8]
            question = f"{w[0]} {w[1]} {w[2]} {w[3]} . {w[4]} {w[5]} {w[6]}"
            answer = f" {w[7]}"
            if is_valid_qa(tokenizer, question, answer):
                questions.append(question)
                answers.append(answer)
                families.append(family)
    if not questions:
        raise RuntimeError("no valid sequence-continuation examples for this tokenizer")
    return questions, answers, families


class SimpleDataset:
    """Simple math reasoning dataset for testing."""
    
    @staticmethod
    def get_arithmetic_tasks() -> Tuple[List[str], List[str]]:
        """
        Get simple arithmetic questions and answers.
        
        Returns:
            Tuple of (questions, answers).
        """
        questions = [
            "What is 5 + 3?",
            "What is 10 - 4?",
            "What is 6 * 2?",
            "What is 12 / 3?",
            "What is 7 + 8?",
            "What is 15 - 7?",
            "What is 3 * 4?",
            "What is 20 / 4?",
        ]
        
        answers = [
            "8",
            "6",
            "12",
            "4",
            "15",
            "8",
            "12",
            "5",
        ]
        
        return questions, answers
    
    @staticmethod
    def get_multistep_tasks() -> Tuple[List[str], List[str]]:
        """
        Get multi-step reasoning questions.
        
        Returns:
            Tuple of (questions, answers).
        """
        questions = [
            "If I have 5 apples and get 3 more, how many do I have? Answer:",
            "Alice has 10 dollars. She spends 4 dollars. How much is left? Answer:",
            "A book costs 15 dollars and pencil costs 2 dollars. What is the total cost? Answer:",
            "There are 6 red balls and 4 blue balls. How many total? Answer:",
            "John has 20 candies. He gives half to his friend. How many does he keep? Answer:",
        ]
        
        answers = [
            "8",
            "6",
            "17",
            "10",
            "10",
        ]
        
        return questions, answers
    
    @staticmethod
    def get_logic_tasks() -> Tuple[List[str], List[str]]:
        """
        Get simple logic reasoning questions.
        
        Returns:
            Tuple of (questions, answers).
        """
        questions = [
            "All dogs are animals. Max is a dog. Is Max an animal? Answer:",
            "If it is raining, the ground is wet. It is raining. Is the ground wet? Answer:",
            "Some cats are black. Tom is a cat. Is Tom black? Answer:",
            "If A > B and B > C, then is A > C? Answer:",
        ]
        
        answers = [
            "Yes",
            "Yes",
            "Unknown",
            "Yes",
        ]
        
        return questions, answers
    
    @staticmethod
    def load_gsm8k_sample(num_samples=10) -> Tuple[List[str], List[str]]:
        """
        Load a small sample of GSM8K-like math problems.
        
        Args:
            num_samples: Number of samples to load.
        
        Returns:
            Tuple of (questions, answers).
        """
        problems = [
            ("Natalia sold clips to 48 of her friends in April, and then she sold clips to 3 fewer friends in May. How many friends did she sell clips to altogether?", "93"),
            ("Weng earns 12 dollars per hour. How much does she earn in 27 hours?", "324"),
            ("Mr. Gabel is buying a new car for 32000 dollars. He is trading in his old car for 5000 dollars off. He is also getting an employee discount of 2000 dollars off. How much will he have to pay?", "25000"),
            ("How many minutes does Elijah spend doing homework every week if he does homework 5 days a week and spends 48 minutes each day?", "240"),
            ("Jake is ordering pizza. A large pizza costs 15 dollars, and each topping costs 2 dollars. If Jake orders a large pizza with 4 toppings, how much will it cost?", "23"),
        ]
        
        questions = [p[0] + " Answer:" for p in problems[:num_samples]]
        answers = [p[1] for p in problems[:num_samples]]
        
        return questions, answers


# ---------------------------------------------------------------------------
# Indirect object identification (IOI), following the 15 templates listed in
# Appendix E (Figure 14) of Wang et al. (2023), "Interpretability in the Wild".
# [B] is the subject (S), [A] the indirect object (IO). Templates are BABA as
# printed; ABBA variants swap the first [B] and [A] (as the paper describes).
# ---------------------------------------------------------------------------
IOI_TEMPLATES_BABA = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to",
]

_IOI_NAMES = [
    "Mary", "John", "James", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Paul", "Steven", "Andrew",
    "Kevin", "Brian", "George", "Edward", "Ryan", "Jacob", "Gary", "Eric", "Jonathan",
    "Stephen", "Larry", "Justin", "Scott", "Frank", "Peter", "Adam", "Henry", "Patrick",
    "Jack", "Dennis", "Jerry", "Tyler", "Aaron", "Jose", "Nathan", "Kyle", "Carl", "Arthur",
    "Roger", "Joe", "Sean", "Christian", "Austin", "Noah", "Jesse", "Alan", "Juan", "Billy",
    "Bruce", "Gabriel", "Logan", "Albert", "Willie", "Alex", "Wayne", "Ralph", "Roy", "Eugene",
    "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah",
    "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra", "Ashley", "Emily", "Donna",
    "Michelle", "Carol", "Amanda", "Melissa", "Deborah", "Laura", "Rebecca", "Sharon",
    "Cynthia", "Kathleen", "Amy", "Anna", "Angela", "Helen", "Emma", "Nicole", "Rachel",
    "Catherine", "Christine", "Samantha", "Maria", "Heather", "Diane", "Julie", "Victoria",
    "Kelly", "Lauren", "Christina", "Joan", "Evelyn", "Judith", "Olivia", "Martha", "Grace",
]
_IOI_PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "house", "station",
               "park", "church", "library", "beach", "market", "museum", "airport", "hotel",
               "bank", "farm", "theater", "court"]
_IOI_OBJECTS = ["drink", "ring", "kiss", "bone", "basketball", "computer", "necklace", "snack",
                "book", "letter", "bottle", "gift", "ticket", "flower", "key", "phone", "hat",
                "ball", "cake", "card"]


def _single_token_words(tokenizer, words):
    return [w for w in words if len(tokenizer.encode(" " + w)) == 1]


def get_ioi_tasks(tokenizer, n: int = 200, seed: int = 0):
    """
    IOI prompts with three aligned variants per example:
      prompt      clean IOI prompt ending before the IO name
      io, s       the indirect object (correct) and subject (distractor),
                  each a single token with a leading space
      abc_prompt  the same template and fillers, with the second subject
                  occurrence replaced by a third name C (Wang et al.'s
                  "ABC" distribution: the IO-vs-S cue is removed)
      template    template index, so resampling can draw a token-aligned
                  source from another example of the same template
    All names, places and objects are verified single tokens, so prompts
    from the same template have identical token length.
    Returns a list of dicts; examples whose tokenization does not align
    with their ABC variant are dropped (never silently kept).
    """
    rng = random.Random(seed)
    names = _single_token_words(tokenizer, _IOI_NAMES)
    places = _single_token_words(tokenizer, _IOI_PLACES)
    objects = _single_token_words(tokenizer, _IOI_OBJECTS)
    if len(names) < 20 or len(places) < 5 or len(objects) < 5:
        raise ValueError("too few single-token IOI fillers for this tokenizer")
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        t_idx = rng.randrange(len(IOI_TEMPLATES_BABA))
        template = IOI_TEMPLATES_BABA[t_idx]
        abba = rng.random() < 0.5
        a, b, c = rng.sample(names, 3)
        place, obj = rng.choice(places), rng.choice(objects)

        def fill(first_b, first_a, second_s):
            # first clause: "[B] and [A]" (BABA) or "[A] and [B]" (ABBA)
            text = template.replace("[B] and [A]", f"{first_b} and {first_a}", 1)
            text = text.replace("[B]", second_s, 1)
            return text.replace("[PLACE]", place).replace("[OBJECT]", obj)

        if abba:
            prompt = fill(a, b, b)
            abc_prompt = fill(a, b, c)
        else:
            prompt = fill(b, a, b)
            abc_prompt = fill(b, a, c)
        if len(tokenizer.encode(prompt)) != len(tokenizer.encode(abc_prompt)):
            continue
        out.append({"prompt": prompt, "io": " " + a, "s": " " + b, "abc_prompt": abc_prompt,
                    "template": t_idx, "order": "ABBA" if abba else "BABA"})
    if len(out) < n:
        raise ValueError(f"only built {len(out)} aligned IOI examples")
    return out
