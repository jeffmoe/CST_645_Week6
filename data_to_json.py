# data_preprocessing.py
"""
Data preprocessing module for legal text classification.
Prepares data for text-to-text generation models.
"""

import pandas as pd
import json
import re
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path: str, output_json_path: str = None) -> Dict[str, Any]:
    """
    Load CSV data and prepare for text-to-text generation.
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Clean the text
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Keep legal notation
        text = re.sub(r'[^\w\s\.\,\;\:\'\"\(\)\[\]\{\}\@\#\$\%\^\&\*\+\=\-\/\|\\]+', ' ', text)
        return text.strip()
    
    # Clean case titles and text
    df['clean_title'] = df['case_title'].apply(clean_text)
    df['clean_text'] = df['case_text'].apply(clean_text)
    
    # Truncate text to a reasonable length for T5
    df['clean_text'] = df['clean_text'].apply(lambda x: x[:1500] if len(x) > 1500 else x)
    
    # Create prompt-response pairs for text-to-text generation
    df['input_text'] = df.apply(
        lambda row: f"Question: What is the legal outcome for the case '{row['clean_title']}'? Context: {row['clean_text']}",
        axis=1
    )
    
    # The target is a descriptive response
    df['target_text'] = df.apply(
        lambda row: f"The case '{row['clean_title']}' was {row['case_outcome']}. This means the court {get_outcome_description(row['case_outcome'])}.",
        axis=1
    )
    
    # Create dataset dictionary
    data = {
        'inputs': df['input_text'].tolist(),
        'targets': df['target_text'].tolist(),
        'case_ids': df['case_id'].tolist(),
        'outcomes': df['case_outcome'].tolist(),
        'titles': df['clean_title'].tolist(),
        'text_bodies': df['clean_text'].tolist(),
        'original_titles': df['case_title'].tolist(),
    }
    
    # Save as JSON if path provided
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {output_json_path}")
    
    return data

def get_outcome_description(outcome: str) -> str:
    """Get a descriptive phrase for the outcome."""
    descriptions = {
        'cited': 'referred to this case as precedent',
        'applied': 'applied the legal principles from this case',
        'followed': 'followed the reasoning established in this case',
        'considered': 'considered the reasoning in this case',
        'referred_to': 'referred to this case in its reasoning',
        'discussed': 'discussed the legal principles from this case',
        'distinguished': 'distinguished this case from the present circumstances',
        'related': 'found this case to be related',
        'approved': 'approved the reasoning in this case',
        'affirmed': 'affirmed the decision in this case',
        'referred': 'referred to this case for guidance'
    }
    return descriptions.get(outcome, f'handled with outcome: {outcome}')

def split_data(data: Dict[str, Any], test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and test sets.
    """
    inputs = data['inputs']
    targets = data['targets']
    
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        inputs, targets, test_size=test_size, random_state=random_state
    )
    
    train_data = {
        'inputs': train_inputs,
        'targets': train_targets
    }
    
    test_data = {
        'inputs': test_inputs,
        'targets': test_targets
    }
    
    return train_data, test_data

def create_sample_questions(data: Dict[str, Any], num_questions: int = 5) -> List[Dict[str, Any]]:
    """
    Create 5 sample questions for testing.
    """
    import random
    
    indices = list(range(len(data['inputs'])))
    
    # Try to get variety
    selected_indices = []
    outcomes_used = set()
    
    for idx in indices:
        outcome = data['outcomes'][idx]
        if outcome not in outcomes_used and len(selected_indices) < num_questions:
            selected_indices.append(idx)
            outcomes_used.add(outcome)
    
    if len(selected_indices) < num_questions:
        remaining = [i for i in indices if i not in selected_indices]
        random.shuffle(remaining)
        selected_indices.extend(remaining[:num_questions - len(selected_indices)])
    
    questions = []
    for idx in selected_indices:
        questions.append({
            'case_id': data['case_ids'][idx],
            'title': data['original_titles'][idx],
            'actual_outcome': data['outcomes'][idx],
            'input_text': data['inputs'][idx],
            'expected_response': data['targets'][idx],
            'context': data['text_bodies'][idx][:300]
        })
    
    return questions

if __name__ == "__main__":
    # Test the preprocessing
    data = load_and_preprocess_data('legal_text_classification.csv', 'legal_data.json')
    print(f"Loaded {len(data['inputs'])} cases")
    print(f"\nSample input: {data['inputs'][0][:200]}...")
    print(f"\nSample target: {data['targets'][0]}")
    
    # Show sample questions
    questions = create_sample_questions(data, num_questions=5)
    print("\nSample Questions Generated:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q['title']}")
        print(f"   Expected: {q['expected_response']}")
        print()