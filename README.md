# CST_645_Week6
This repo is to show how to perform fine tuning of an LLM model using PyTorch and HuggingFace.

---
## Overview
his application demonstrates how to fine-tune a T5 (Text-to-Text Transfer Transformer) model for legal text classification. The system takes legal case data from a CSV file, preprocesses it, fine-tunes a T5 model to generate natural language descriptions of legal outcomes, and provides interactive testing capabilities.

---
## Features
- Processes legal case data: Reads case titles and text from CSV files
- Fine-tunes a T5 model: Adapts a pre-trained T5 model to legal domain text
- Generates natural language responses: Produces descriptive outcome classifications
- Compares models: Evaluates fine-tuned model against base model performance
- Interactive testing: Allows users to test models with custom questions
 
  ---
## Installation
### Prereqs
- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- CUDA-capable GPU (optional, but recommended for faster training)

### Steps
  1. Clone the repository
  2. Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```
  3. Run the Pipeline:
     ```bash
     # Use mini T5 model (faster)
     python main.py --csv legal_text_classification.csv --mini
    
     # Use T5-base model (slower but potentially better)
     python main.py --csv legal_text_classification.csv --base
     
     # Show sample questions without training
     python main.py --show_questions
      ```
  ---
## Further Testing
### Interactive Test Mode
```bash
python main.py --test
```
Ask Questions to the model in the following format:  
Question: What is the outcome for the case 'Alpine Hardwood'? 

---
## Command Line Options

|Option | Description|
|-------|------------|
|--csv PATH | Path to your CSV data file |
|--mini |	Use T5-small (faster training) |
|--base |	Use T5-base (slower, better quality) |
|--test | Run interactive testing mode |
|--show_questions | 	Display 5 sample questions |
|--skip_train |Skip training (use existing model) |

