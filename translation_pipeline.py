import argparse
import os
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

def translate_text(text, model_name="gpt-3.5-turbo"):
    if not text:
        return ""

    prompt = (
        "Translate the following German sentence to English:\n\n"
        f"{text}\n\nReturn only the translated sentence."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        print(f" OpenAI API error: {e}")
        return "<ERROR>", e

def translate_column_samplewise(dataset, column, model="gpt-3.5-turbo", preprocess_fn=None):
    translations = []

    for item in tqdm(dataset, desc=f"Translating '{column}'"):
        text = item[column]

        if preprocess_fn:
            text = preprocess_fn(text)
        else:
            text = str(text) if text is not None else ""

        translated = translate_text(text, model_name=model)
        translations.append(translated)

    return translations

def main():
    parser = argparse.ArgumentParser(description="Translate German QA dataset including context using OpenAI.")
    parser.add_argument("--path", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to translate")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output .jsonl file")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use (e.g., gpt-4o)")

    args = parser.parse_args()

    # Load and sample
    full_dataset = load_dataset("json", data_files=args.path, split="train")
    dataset = full_dataset.select(range(min(args.samples, len(full_dataset))))
    print(f" Loaded {len(dataset)} samples from: {args.path}")

    # Translate columns
    dataset = dataset.add_column("question_en", translate_column_samplewise(dataset, "question", model=args.model))
    dataset = dataset.add_column("answer_en", translate_column_samplewise(dataset, "answer", model=args.model))

    # Translate 'context' safely
    if "context" in dataset.column_names:
        def flatten_context(c):
            if isinstance(c, list):
                return " ".join(str(x) for x in c)
            return str(c) if c is not None else ""

        dataset = dataset.add_column("context_en", translate_column_samplewise(dataset, "context", model=args.model, preprocess_fn=flatten_context))
        print("Translated 'context' column.")
    else:
        print("No 'context' column found — skipping.")

    # Save as .jsonl file
    dataset.to_json(args.output_path, orient="records", lines=True)
    print(f"Translated dataset saved at: {args.output_path}")

if __name__ == "__main__":
    main()
