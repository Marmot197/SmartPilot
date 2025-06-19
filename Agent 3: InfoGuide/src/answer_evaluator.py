import os
import json
from typing import Dict, Any
from copilots.Agents import LLM
from run_updated_barebones import main as run_barebones_main
import pandas as pd

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
4. Does it miss any additional valuable information present in the ground truth?

Provide your evaluation in the following JSON format:
"Score": [1-5] (5 being best)
"Explanation": [Detailed explanation of the score]

Your response should be objective and thorough."""

        try:
            self.llm.set_prompt("You are an expert evaluator that provides detailed, objective assessments.", prompt)
            self.llm.set_max_tokens(512)  # Set appropriate token limit
            evaluation_text = self.llm.respond_to_prompt_json()
            # print(evaluation_text)

            evaluation_json = json.loads(evaluation_text)['Response']
            print(evaluation_json)
            # Parse the response
            score = evaluation_json["Score"]
            explanation = evaluation_json["Explanation"]
            
            return {
                'score': score,
                'explanation': explanation
            }
            
        except Exception as e:
            raise Exception(f"Error during evaluation: {str(e)}")

def main():
    # Example usage
    evaluator = AnswerEvaluator()
    df = pd.read_csv('C3AN Evaluation - compiled.csv')
    print(df.head())
    
    # Create a list to store results
    results_data = []
    
    for i in range(len(df)):
        question = df.iloc[i]['Question']
        ground_truth = df.iloc[i]['Ground Truth']
        print("Question: ", question)
        try:
            input_answer = run_barebones_main(question)
            print(f"\n🤖 Generated Answer: {input_answer}")
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            input_answer = "Error generating answer"
        
        try: 
            result = evaluator.evaluate_answer(question, ground_truth, input_answer)
        except Exception as e:
            print(f"❌ Error evaluating answer: {e}")
            result = {
                'score': 0,
                'explanation': "Error evaluating answer"
            }
        print(f"Score: {result['score']}/5")
        print(f"\nExplanation: {result['explanation']}")
        
        # Store results in the list
        results_data.append({
            'Question': question,
            'Ground_Truth': ground_truth,
            'Generated_Answer': input_answer,
            'Score': result['score'],
            'Explanation': result['explanation']
        })
        
        print("\n" + "="*50 + "\n")  # Separator between questions
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('results.csv', index=False)
    print(f"✅ Results saved to results.csv with {len(results_df)} evaluations")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"Average Score: {results_df['Score'].mean():.2f}/5")
    print(f"Highest Score: {results_df['Score'].max()}/5")
    print(f"Lowest Score: {results_df['Score'].min()}/5")
    print(f"Total Questions Evaluated: {len(results_df)}")

if __name__ == "__main__":
    main()