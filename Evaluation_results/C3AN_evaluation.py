import pandas as pd
from sklearn.metrics import cohen_kappa_score

def calculate_average_scores(filename, human_evaluators, openai_col, groq_col):
    """
    Args:
        filename (str): Path to the CSV file
        human_evaluators (list): List of human evaluator column names
        openai_col (str): Column name for OpenAI scores
        groq_col (str): Column name for Groq scores
    """
    df = pd.read_csv(filename)
    result_df = df.groupby(['Principle', 'New Agent']).agg({
        openai_col: 'mean',
        groq_col: 'mean',
        **{col: 'mean' for col in human_evaluators}
    }).reset_index()
    human_cols = [col for col in result_df.columns if col in human_evaluators]
    result_df['human_score'] = result_df[human_cols].mean(axis=1)

    final_df = result_df[['Principle', 'New Agent', openai_col, groq_col, 'human_score']]
    
    return final_df

def calculate_kappa_matrix(filename, human_evaluators, openai_col, groq_col):
    """
    Args:
        filename (str): Path to the CSV file
        human_evaluators (list): List of human evaluator column names
        openai_col (str): Column name for OpenAI scores
        groq_col (str): Column name for Groq scores
    """
    df = pd.read_csv(filename)
    raters = [openai_col, groq_col] + human_evaluators
    kappa_matrix = pd.DataFrame(index=raters, columns=raters, dtype=float)

    for r1 in raters:
        for r2 in raters:
            if r1 == r2:
                kappa_matrix.loc[r1, r2] = 1.0  # Perfect agreement with self
            else:
                kappa = round(cohen_kappa_score(df[r1], df[r2], weights='quadratic'), 3)
                kappa_matrix.loc[r1, r2] = kappa
    
    return kappa_matrix