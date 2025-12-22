from typing import List, Tuple


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
