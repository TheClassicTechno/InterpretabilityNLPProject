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
