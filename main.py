# main.py
"""
Main script to run the complete pipeline with T5 text generation.
"""

import os
import argparse
import json
from data_to_json import load_and_preprocess_data, create_sample_questions
from model_train import train_mini_model, train_model
from model_test import test_models_main, generate_test_questions, compare_models
from model_test import load_model_and_tokenizer, generate_response

def run_pipeline(csv_path: str, use_mini_model: bool = True):
    """
    Run the complete pipeline.
    """
    # Step 1: Preprocess data
    print("STEP 1: Preprocessing Data for Text Generation")
    data = load_and_preprocess_data(csv_path, 'legal_data.json')
    print(f"Preprocessed {len(data['inputs'])} cases")
    print(f"Unique outcome types: {set(data['outcomes'])}")
    
    # Show sample
    print("\nSample training pair:")
    print(f"Input: {data['inputs'][0][:150]}...")
    print(f"Target: {data['targets'][0]}")
    
    # Step 2: Train model
    print("\nSTEP 2: Training T5 Model")
    
    if use_mini_model:
        print("Using mini T5 model (t5-small) for faster training...")
        trainer, model, tokenizer = train_mini_model(
            json_data_path='legal_data.json',
            output_dir='./fine_tuned_mini_model'
        )
        base_model = 't5-small'
        ft_model_path = './fine_tuned_mini_model'
    else:
        print("Using T5-base model (slower but potentially better)...")
        trainer, model, tokenizer = train_model(
            json_data_path='legal_data.json',
            model_name='t5-base',
            num_epochs=3,
            output_dir='./fine_tuned_model'
        )
        base_model = 't5-base'
        ft_model_path = './fine_tuned_model'
    
    # Step 3: Test and compare models
    print("\nSTEP 3: Testing and Comparing Models on 5 Legal Cases")
    
    questions = generate_test_questions('legal_data.json', num_questions=5)
    
    print("\nTesting on the following cases:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q['title']} (Expected: {q['actual_outcome']})")
    
    results = compare_models(
        questions=questions,
        base_model_path=base_model,
        fine_tuned_model_path=ft_model_path
    )

    print("\nPIPELINE COMPLETE")
    return results

def interactive_test():
    """
    Run interactive testing with custom legal input.
    """
    print("\nINTERACTIVE LEGAL TEXT GENERATION TEST")
    
    ft_model_path = './fine_tuned_mini_model'
    if not os.path.exists(ft_model_path):
        print("No fine-tuned model found. Please run training first.")
        return
    
    print("Loading models...")
    print("Using t5-small as base model")
    base_tokenizer, base_model = load_model_and_tokenizer('t5-small')
    ft_tokenizer, ft_model = load_model_and_tokenizer(ft_model_path)
    
    print("\nEnter a legal case description or question (or 'quit' to exit):")
    print("Example: 'What is the outcome for the case Alpine Hardwood? Context: [case text]'")
    
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input:
            continue
        
        # Format as T5 input if not already formatted
        if not user_input.startswith("Question:"):
            formatted_input = f"Question: {user_input}"
        else:
            formatted_input = user_input
        
        print("\nGenerating responses...")
        
        try:
            base_response = generate_response(formatted_input, base_tokenizer, base_model)
            ft_response = generate_response(formatted_input, ft_tokenizer, ft_model)
            
            print("\nRESPONSES:")
            print(f"Base Model (t5-small):")
            print(f"  {base_response}")
            print()
            print(f"Fine-tuned Model:")
            print(f"  {ft_response}")
        except Exception as e:
            print(f"Error generating response: {e}")

def show_sample_questions(json_path: str = 'legal_data.json'):
    """Display 5 sample questions without running the full pipeline."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        questions = create_sample_questions(data, num_questions=5)
        print("\n5 SAMPLE LEGAL QUESTIONS GENERATED FROM THE DATA")

        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}:")
            print(f"  Case: {q['title']}")
            print(f"  Expected outcome: {q['actual_outcome']}")
            print(f"  Expected response: {q['expected_response']}")
            print(f"  Input: {q['input_text'][:150]}...")
        return questions
    except FileNotFoundError:
        print("Data file not found. Please run preprocessing first.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Legal Text Generation Pipeline')
    parser.add_argument('--csv', default='legal_text_classification.csv', help='Path to CSV file')
    parser.add_argument('--mini', action='store_true', default=True, help='Use mini T5 model')
    parser.add_argument('--base', action='store_true', help='Use T5-base model (slower)')
    parser.add_argument('--test', action='store_true', help='Run interactive test mode')
    parser.add_argument('--show_questions', action='store_true', help='Show 5 sample questions')
    parser.add_argument('--skip_train', action='store_true', help='Skip training (assumes model already trained)')
    
    args = parser.parse_args()
    
    if args.show_questions:
        load_and_preprocess_data(args.csv, 'legal_data.json')
        show_sample_questions()
    elif args.test:
        if not os.path.exists('./fine_tuned_mini_model') and not args.skip_train:
            print("No fine-tuned model found. Running training first...")
            run_pipeline(args.csv, use_mini_model=True)
        interactive_test()
    else:
        use_mini = not args.base
        run_pipeline(args.csv, use_mini_model=use_mini)