# test_models.py
"""
Test and compare base T5 model vs fine-tuned model on 5 legal questions.
Compatible with transformers 5.12.1
"""

import json
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_to_json import create_sample_questions

def load_model_and_tokenizer(model_path: str):
    """Load a trained T5 model and tokenizer."""
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def generate_response(input_text: str, tokenizer, model, max_length: int = 256):
    """
    Generate a response for a given input text using T5.
    """
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            early_stopping=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_test_questions(data_path: str, num_questions: int = 5):
    """
    Generate exactly 5 test questions from the dataset.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    import random
    indices = list(range(len(data['inputs'])))
    
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
            'title': data['titles'][idx],
            'actual_outcome': data['outcomes'][idx],
            'input_text': data['inputs'][idx],
            'expected_response': data['targets'][idx],
            'context': data['text_bodies'][idx][:300]
        })
    
    return questions

def compare_models(questions: list, base_model_path: str, fine_tuned_model_path: str):
    """
    Compare base T5 model and fine-tuned T5 model responses.
    """
    # Load models
    print("Loading base T5 model...")
    base_tokenizer, base_model = load_model_and_tokenizer(base_model_path)
    
    print("Loading fine-tuned T5 model...")
    ft_tokenizer, ft_model = load_model_and_tokenizer(fine_tuned_model_path)
    
    base_name = base_model_path.split('/')[-1] if '/' in base_model_path else base_model_path
    ft_name = fine_tuned_model_path.split('/')[-1] if '/' in fine_tuned_model_path else fine_tuned_model_path
    
    results = []
    
    
    print(f"\nMODEL COMPARISON: BASE ({base_name}) vs FINE-TUNED ({ft_name})")
    print(f"Testing on {len(questions)} legal cases:\n")
    
    for i, q in enumerate(questions, 1):
        print(f"{'─'*80}")
        print(f"Question {i}:")
        print(f"  Case: {q['title']}")
        print(f"  Expected outcome: {q['actual_outcome']}")
        print(f"  Expected response: {q['expected_response']}")
        print()
        
        try:
            # Generate responses
            base_response = generate_response(q['input_text'], base_tokenizer, base_model)
            ft_response = generate_response(q['input_text'], ft_tokenizer, ft_model)
            
            # Check if responses contain the expected outcome
            base_contains_outcome = q['actual_outcome'].lower() in base_response.lower()
            ft_contains_outcome = q['actual_outcome'].lower() in ft_response.lower()
            
            results.append({
                'question_id': i,
                'case_title': q['title'],
                'expected_outcome': q['actual_outcome'],
                'expected_response': q['expected_response'],
                'base_model_response': base_response,
                'ft_model_response': ft_response,
                'base_contains_outcome': base_contains_outcome,
                'ft_contains_outcome': ft_contains_outcome,
                'base_correct': base_contains_outcome,
                'ft_correct': ft_contains_outcome
            })
            
            # Print results
            print("BASE MODEL RESPONSE:")
            print(f"  {base_response}")
            print(f"  Contains expected outcome: {'YES' if base_contains_outcome else 'NO'}")
            print()
            
            print("FINE-TUNED MODEL RESPONSE:")
            print(f"  {ft_response}")
            print(f"  Contains expected outcome: {'YES' if ft_contains_outcome else 'NO'}")
            print()
            
            # Comparison summary
            if base_contains_outcome and ft_contains_outcome:
                print("➜ Both models correctly identified the outcome ✓")
            elif ft_contains_outcome and not base_contains_outcome:
                print("➜ Fine-tuned model improved - correctly identified the outcome ✓")
            elif base_contains_outcome and not ft_contains_outcome:
                print("➜ Fine-tuned model worsened - missed the outcome ✗")
            else:
                print("➜ Both models failed to identify the outcome ✗")
            
        except Exception as e:
            print(f"Error generating response for question {i}: {e}")
            results.append({
                'question_id': i,
                'case_title': q['title'],
                'expected_outcome': q['actual_outcome'],
                'expected_response': q['expected_response'],
                'base_model_response': f"ERROR: {str(e)}",
                'ft_model_response': f"ERROR: {str(e)}",
                'base_contains_outcome': False,
                'ft_contains_outcome': False,
                'base_correct': False,
                'ft_correct': False
            })
        
        print()
    
    # Summary statistics
    if results:
        df_results = pd.DataFrame(results)
        
        print("\nSUMMARY")
        
        base_accuracy = df_results['base_correct'].mean()
        ft_accuracy = df_results['ft_correct'].mean()
        
        print(f"Base Model Accuracy (contains outcome): {base_accuracy:.2%} ({df_results['base_correct'].sum()}/{len(df_results)})")
        print(f"Fine-tuned Model Accuracy (contains outcome): {ft_accuracy:.2%} ({df_results['ft_correct'].sum()}/{len(df_results)})")
        print(f"Improvement: {(ft_accuracy - base_accuracy):.2%}")
        
        # Detailed comparison
        improved = ((~df_results['base_correct']) & df_results['ft_correct']).sum()
        worsened = (df_results['base_correct'] & (~df_results['ft_correct'])).sum()
        same_correct = (df_results['base_correct'] & df_results['ft_correct']).sum()
        same_incorrect = ((~df_results['base_correct']) & (~df_results['ft_correct'])).sum()
        
        print(f"\nDetailed Comparison:")
        print(f"  Both correct: {same_correct}")
        print(f"  Both incorrect: {same_incorrect}")
        print(f"  Improved by fine-tuning: {improved}")
        print(f"  Worsened by fine-tuning: {worsened}")
        
        # Per-outcome breakdown
        print("\nPER-OUTCOME BREAKDOWN")
        
        outcome_counts = df_results['expected_outcome'].value_counts()
        for outcome in outcome_counts.index:
            n = outcome_counts[outcome]
            base_correct = df_results[df_results['expected_outcome'] == outcome]['base_correct'].sum()
            ft_correct = df_results[df_results['expected_outcome'] == outcome]['ft_correct'].sum()
            print(f"\n{outcome} ({n} case{'s' if n > 1 else ''}):")
            print(f"  Base model: {base_correct}/{n} correct ({base_correct/n:.1%})")
            print(f"  Fine-tuned: {ft_correct}/{n} correct ({ft_correct/n:.1%})")
        
        # Save detailed results
        df_results.to_csv('model_comparison_results.csv', index=False)
        print(f"\nDetailed results saved to 'model_comparison_results.csv'")
        
        return df_results
    else:
        print("No results to display.")
        return None

def test_models_main(json_data_path: str, base_model_path: str = 't5-small'):
    """
    Main function to test and compare models on 5 questions.
    """
    print("Generating 5 test questions from data...")
    questions = generate_test_questions(json_data_path, num_questions=5)
    
    print(f"\nGenerated {len(questions)} questions from cases:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q['title']}")
        print(f"   Expected: {q['actual_outcome']}")
        print()
    
    results = compare_models(
        questions=questions,
        base_model_path=base_model_path,
        fine_tuned_model_path='./fine_tuned_mini_model'
    )
    
    return results

if __name__ == "__main__":
    from data_to_json import load_and_preprocess_data
    load_and_preprocess_data('legal_text_classification.csv', 'legal_data.json')
    results = test_models_main('legal_data.json')