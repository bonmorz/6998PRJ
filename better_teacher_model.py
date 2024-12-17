"""
Better Teacher Model Script
--------------------------

This script improves the reasoning component of a QA dataset by using GPT-4 to generate
enhanced step-by-step explanations. It takes a JSON file containing questions, original reasoning,
answers, and knowledge context, then generates new reasoning using GPT-4.

Input JSON format:
[
    {
        "question": "Multiple choice question text",
        "reasoning": "Original reasoning text",
        "answer": answer_index,
        "knowledge": "Relevant knowledge text"
    },
    ...
]

Usage:
------
1. Install required packages:
   pip install openai tqdm

2. Run the script:
   python better_teacher_model.py \
       --file_path path/to/input.json \
       --api_key your-openai-api-key \
       --save_path path/to/output.json

Optional arguments:
    --batch_size: Number of questions to process in parallel (default: 1)
    --max_retries: Maximum number of retries for failed API calls (default: 3)
    --retry_delay: Delay between retries in seconds (default: 5)

Example:
    python better_teacher_model.py \
        --file_path data/train.json \
        --api_key sk-xxx... \
        --save_path output/improved_train.json

Notes:
- The script saves progress incrementally to prevent data loss
- Original data is preserved if API calls fail
- The output maintains the same JSON structure as the input
- GPT-4 is prompted to provide structured, step-by-step reasoning
"""

import json
import argparse
import time
from typing import Dict, List
import os
from openai import OpenAI
from tqdm import tqdm

def parse_args():
    """
    Parse command line arguments.
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate improved reasoning using GPT-4')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output JSON file')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of questions to process in parallel')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum number of retries for API calls')
    parser.add_argument('--retry_delay', type=int, default=5, help='Delay between retries in seconds')
    return parser.parse_args()

def setup_openai_client(api_key: str) -> OpenAI:
    """
    Initialize the OpenAI client with the provided API key.
    Args:
        api_key (str): OpenAI API key
    Returns:
        OpenAI: Initialized OpenAI client
    """
    return OpenAI(api_key=api_key)

def create_prompt(question: str, knowledge: str) -> str:
    """
    Create a prompt for GPT-4 that includes the question and relevant knowledge.
    Args:
        question (str): The multiple-choice question
        knowledge (str): Relevant knowledge context
    Returns:
        str: Formatted prompt for GPT-4
    """
    return f"""The following is a multiple-choice question with relevant knowledge. Solve it in a step-by-step fashion to find the correct answer.

Question: {question}

Relevant Knowledge: {knowledge}

Please provide a step-by-step explanation of your reasoning that:
1. Breaks down the key information from both the question and knowledge
2. Analyzes how the knowledge relates to the question
3. Uses logical deduction to arrive at the answer
4. Ends with "Answer: [letter]" where [letter] is the chosen option

Explanation:"""

def generate_reasoning(client: OpenAI, prompt: str, max_retries: int = 3, retry_delay: int = 5) -> str:
    """
    Generate reasoning using GPT-4 with retry mechanism for failed API calls.
    Args:
        client (OpenAI): Initialized OpenAI client
        prompt (str): Formatted prompt for GPT-4
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
    Returns:
        str: Generated reasoning text
    Raises:
        Exception: If all retry attempts fail
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at solving multiple-choice questions through careful reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

def process_dataset(data: List[Dict], client: OpenAI, args) -> List[Dict]:
    """
    Process each item in the dataset to generate improved reasoning.
    Args:
        data (List[Dict]): List of question data dictionaries
        client (OpenAI): Initialized OpenAI client
        args: Command line arguments
    Returns:
        List[Dict]: Processed dataset with improved reasoning
    """
    processed_data = []
    
    for item in tqdm(data, desc="Processing questions"):
        try:
            prompt = create_prompt(item["question"], item["knowledge"])
            new_reasoning = generate_reasoning(
                client,
                prompt,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay
            )
            
            # Create new item with updated reasoning
            new_item = {
                "question": item["question"],
                "reasoning": new_reasoning,
                "answer": item["answer"],
                "knowledge": item["knowledge"]
            }
            processed_data.append(new_item)
            
            # Save progress after each successful processing
            with open(args.save_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing question: {item['question'][:100]}...")
            print(f"Error message: {str(e)}")
            # Keep original reasoning if processing fails
            processed_data.append(item)

    return processed_data

def main():
    """Main function to orchestrate the reasoning improvement process."""
    args = parse_args()
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Load input data
    with open(args.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Setup OpenAI client
    client = setup_openai_client(args.api_key)
    
    # Process the dataset
    processed_data = process_dataset(data, client, args)
    
    # Save final results
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Results saved to {args.save_path}")

if __name__ == "__main__":
    main()