import os
from typing import Dict, Any
from copilots.Agents import LLM

class AnswerEvaluator:
    def __init__(self):
        """
        Initialize the AnswerEvaluator with the LLM class.
        """
        self.llm = LLM()

    def evaluate_answer(self, question: str, ground_truth: str, input_answer: str) -> Dict[str, Any]:
        """
        Evaluate an input answer against the ground truth using an LLM.
        
        Args:
            question (str): The question being answered
            ground_truth (str): The correct/expected answer
            input_answer (str): The answer to be evaluated
            
        Returns:
            Dict containing:
                - score (int): Score from 1-5
                - explanation (str): Explanation of the score
                - missing_points (list): List of points from ground truth that were missing
                - additional_points (list): List of points in input answer that weren't in ground truth
        """
        prompt = f"""You are an expert evaluator. Your task is to evaluate an answer to a question by comparing it with the ground truth answer.
        
Question: {question}

Ground Truth Answer: {ground_truth}

Input Answer to Evaluate: {input_answer}

Please evaluate the input answer based on the following criteria:
1. How well does it cover the key points from the ground truth?
2. Is the information accurate and relevant?
3. Is it well-structured and clear?
4. Does it provide any additional valuable information not in the ground truth?

Provide your evaluation in the following format:
Score: [1-5] (5 being best)
Explanation: [Detailed explanation of the score]
Missing Points: [List any key points from ground truth that were missing]
Additional Points: [List any valuable points in input answer not in ground truth]

Your response should be objective and thorough."""

        try:
            self.llm.set_prompt("You are an expert evaluator that provides detailed, objective assessments.", prompt)
            self.llm.set_max_tokens(512)  # Set appropriate token limit
            evaluation_text = self.llm.respond_to_prompt()
            
            # Parse the response
            score = None
            explanation = ""
            missing_points = []
            additional_points = []
            
            current_section = None
            for line in evaluation_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('Score:'):
                    try:
                        score = int(line.split(':')[1].strip().split()[0])
                    except:
                        score = 3  # Default score if parsing fails
                elif line.startswith('Explanation:'):
                    current_section = 'explanation'
                    explanation = line.split(':', 1)[1].strip()
                elif line.startswith('Missing Points:'):
                    current_section = 'missing'
                elif line.startswith('Additional Points:'):
                    current_section = 'additional'
                elif current_section == 'missing' and line.startswith('-'):
                    missing_points.append(line[1:].strip())
                elif current_section == 'additional' and line.startswith('-'):
                    additional_points.append(line[1:].strip())
                elif current_section == 'explanation':
                    explanation += ' ' + line
            
            return {
                'score': score,
                'explanation': explanation,
                'missing_points': missing_points,
                'additional_points': additional_points
            }
            
        except Exception as e:
            raise Exception(f"Error during evaluation: {str(e)}")

def main():
    # Example usage
    evaluator = AnswerEvaluator()
    
    question = "What are the main benefits of using Python for data science?"
    ground_truth = """Python is excellent for data science because:
1. It has extensive libraries like NumPy, Pandas, and Scikit-learn
2. It's easy to learn and read
3. It has great community support
4. It integrates well with other tools and platforms
5. It's versatile and can handle various data types"""
    
    input_answer = """Python is good for data science because it has many useful libraries and is easy to use. 
    You can do data analysis and machine learning with it."""
    
    result = evaluator.evaluate_answer(question, ground_truth, input_answer)
    
    print(f"Score: {result['score']}/5")
    print(f"\nExplanation: {result['explanation']}")
    print("\nMissing Points:")
    for point in result['missing_points']:
        print(f"- {point}")
    print("\nAdditional Points:")
    for point in result['additional_points']:
        print(f"- {point}")

if __name__ == "__main__":
    main() 