"""
This script enhances reasoning generation by using GPT-3.5-turbo with retrieved knowledge context.
It takes a JSON file containing questions, original reasoning, answers, and knowledge context,
then generates new reasoning incorporating the provided knowledge.

Key Features:
- Incorporates retrieved knowledge into the reasoning process
- Maintains the original question-answer format
- Saves progress incrementally

Input JSON format:
[
    {
        "question": "Multiple choice question text",
        "reasoning": "Original reasoning text",
        "answer": answer_index,
        "knowledge": "Retrieved knowledge text"
    },
    ...
]

Usage:
------

2. Run the script:
   python gpt3.5_RAG.py \
       --file_path path/to/input.json \
       --api_key your-openai-api-key \
       --save_path path/to/output.json

Optional arguments:
    --max_retries: Maximum number of retries for failed API calls (default: 3)
    --retry_delay: Delay between retries in seconds (default: 5)
    --temperature: Temperature for GPT-3.5 generation (default: 0.7)

Example:
    python gpt3.5_RAG.py \
        --file_path data/train.json \
        --api_key sk-xxx... \
        --save_path output/improved_train.json
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
    parser = argparse.ArgumentParser(description='Generate improved reasoning using GPT-3.5-turbo with RAG')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output JSON file')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum number of retries for API calls')
    parser.add_argument('--retry_delay', type=int, default=5, help='Delay between retries in seconds')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
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
    Create a prompt for GPT-3.5-turbo that includes the question and knowledge context.
    Args:
        question (str): The multiple-choice question
        knowledge (str): Retrieved knowledge context
    Returns:
        str: Formatted prompt for GPT-3.5-turbo
    """
    return f"""Given the following multiple-choice question and relevant knowledge, provide a detailed step-by-step reasoning to find the correct answer.

Question: {question}

Relevant Knowledge: {knowledge}

Please provide your reasoning by:
1. First identifying the key points from the question
2. Analyzing the provided knowledge and how it relates to the question
3. Using step-by-step logical reasoning to reach the answer
4. Concluding with "Answer: [letter]" where [letter] is your chosen option

Let's solve this step by step:"""

def generate_rag_reasoning(
    client: OpenAI,
    prompt: str,
    temperature: float = 0.7,
    max_retries: int = 3,
    retry_delay: int = 5
) -> str:
    """
    Generate reasoning using GPT-3.5-turbo with retry mechanism for failed API calls.
    Args:
        client (OpenAI): Initialized OpenAI client
        prompt (str): Formatted prompt
        temperature (float): Generation temperature
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
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at solving multiple-choice questions through careful reasoning and analysis of provided knowledge."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
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
    Process each item in the dataset to generate improved reasoning using RAG.
    Args:
        data (List[Dict]): List of question data dictionaries
        client (OpenAI): Initialized OpenAI client
        args: Command line arguments
    Returns:
        List[Dict]: Processed dataset with improved reasoning
    """
    processed_data = []
    
    for item in tqdm(data, desc="Processing questions with RAG"):
        try:
            # Create prompt with question and knowledge context
            prompt = create_prompt(item["question"], item["knowledge"])
            
            # Generate new reasoning
            new_reasoning = generate_rag_reasoning(
                client,
                prompt,
                temperature=args.temperature,
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
            
            # Save progress incrementally
            with open(args.save_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing question: {item['question'][:100]}...")
            print(f"Error message: {str(e)}")
            # Preserve original data if processing fails
            processed_data.append(item)

    return processed_data

def main():
    """Main function to orchestrate the RAG-enhanced reasoning generation process."""
    args = parse_args()
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Load input data
    print(f"Loading data from {args.file_path}")
    with open(args.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Setup OpenAI client
    print("Initializing OpenAI client")
    client = setup_openai_client(args.api_key)
    
    # Process the dataset
    print("Starting RAG-enhanced reasoning generation")
    processed_data = process_dataset(data, client, args)
    
    # Save final results
    print(f"Saving results to {args.save_path}")
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Results saved to {args.save_path}")

if __name__ == "__main__":
    main()