import os
import json
from typing import Dict, Any
import pandas as pd
from openai import OpenAI

class OpenAIAnswerEvaluator:
    def __init__(self, api_key: str):
        """
        Initialize the OpenAIAnswerEvaluator with OpenAI client.
        
        Args:
            api_key (str): OpenAI API key
        """
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    def evaluate_answer(self, question: str, ground_truth: str, input_answer: str) -> Dict[str, Any]:
        """
        Evaluate an input answer against the ground truth using GPT-4o mini.
        
        Args:
            question (str): The question being answered
            ground_truth (str): The correct/expected answer
            input_answer (str): The answer to be evaluated
            
        Returns:
            Dict containing:
                - score (int): Score from 1-5
                - explanation (str): Explanation of the score
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

If the above criteria are not met perfectly, ensure you reduce the score. 
If the answer contains extra information that isnt present in the ground truth, do not reduce the score. You can overlook that part.

Provide your evaluation in the following JSON format:
{{
    "Score": [1-5],
    "Explanation": "Detailed explanation of the score"
}}

Your response should be objective and thorough. Only return valid JSON."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator that provides detailed, objective assessments. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.1  # Low temperature for consistent evaluation
            )
            
            evaluation_text = response.choices[0].message.content
            if evaluation_text is None:
                raise Exception("Received empty response from OpenAI")
            evaluation_text = evaluation_text.strip()
            
            # Try to parse the JSON response
            try:
                evaluation_json = json.loads(evaluation_text)
            except json.JSONDecodeError:
                # If direct JSON parsing fails, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
                if json_match:
                    evaluation_json = json.loads(json_match.group())
                else:
                    raise Exception("Could not parse JSON response from OpenAI")
            
            # Extract score and explanation
            score = evaluation_json.get("Score", 0)
            explanation = evaluation_json.get("Explanation", "No explanation provided")
            
            return {
                'score': score,
                'explanation': explanation
            }
            
        except Exception as e:
            raise Exception(f"Error during evaluation: {str(e)}")

def main():
    # Initialize evaluator with your API key
    api_key = '' #secret key
    
    try:
        evaluator = OpenAIAnswerEvaluator(api_key=api_key)
        print("✅ OpenAI Answer Evaluator initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing evaluator: {e}")
        return
    
    # Load the dataset
    df = pd.read_csv('c3an_final_eval.csv')
    print(f"✅ Loaded dataset with {len(df)} rows")
    print(df.head())
    
    # Create a list to store results
    results_data = []
    
    for i in range(len(df)):
        question = df.iloc[i]['Question']
        ground_truth = df.iloc[i]['Ground Truth']
        input_answer = df.iloc[i]['Smartpilot answer']
        try: 
            result = evaluator.evaluate_answer(question, ground_truth, input_answer)
            print(f"✅ Score: {result['score']}/5")
            print(f"📋 Explanation: {result['explanation'][:200]}{'...' if len(result['explanation']) > 200 else ''}")
        except Exception as e:
            print(f"❌ Error evaluating answer: {e}")
            result = {
                'score': 0,
                'explanation': f"Error evaluating answer: {str(e)}"
            }
        
        # Store results in the list
        results_data.append({
            'Question': question,
            'Ground_Truth': ground_truth,
            'Generated_Answer': input_answer,
            'Score': result['score'],
            'Explanation': result['explanation']
        })
        
        print("-" * 50)  # Separator between questions
    
    # Create DataFrame and append to CSV
    try:
        results_df = pd.DataFrame(results_data)
        output_filename = 'openai_results.csv'
        
        # Check if file exists to append, otherwise create new
        # if os.path.exists(output_filename):
        #     # Read existing data and append new results
        #     existing_df = pd.read_csv(output_filename)
        #     combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        #     combined_df.to_csv(output_filename, index=False)
        #     print(f"\n✅ Results appended to {output_filename}")
        #     print(f"📊 Total evaluations in file: {len(combined_df)} (added {len(results_df)} new)")
        # else:
            # Create new file
        results_df.to_csv(output_filename, index=False)
        print(f"\n✅ Results saved to {output_filename} with {len(results_df)} evaluations")
        
        # Print summary statistics
        print(f"\n📊 Summary Statistics:")
        print(f"Average Score: {results_df['Score'].mean():.2f}/5")
        print(f"Highest Score: {results_df['Score'].max()}/5")
        print(f"Lowest Score: {results_df['Score'].min()}/5")
        print(f"Total Questions Evaluated: {len(results_df)}")
        
        # Score distribution
        print(f"\n📈 Score Distribution:")
        score_counts = results_df['Score'].value_counts().sort_index()
        for score, count in score_counts.items():
            percentage = (count / len(results_df)) * 100
            print(f"Score {score}: {count} questions ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    main() 