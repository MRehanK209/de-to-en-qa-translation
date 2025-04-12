
# Translation Pipeline for German QA Datasets (Using OpenAI GPT)

This Python script translates a JSONL-format dataset containing **German-language question-answer pairs** (and optionally context) into **English**, using the OpenAI GPT model via API.

## Overview

The pipeline performs the following steps:

1. **Reads a `.jsonl` dataset** from a specified path (must include `"question"` and `"answer"` columns).
2. **Translates** the content of each specified column (question, answer, and optional context) from **German to English** using OpenAI's GPT models (e.g., `gpt-3.5-turbo`, `gpt-4`, etc.).
3. **Appends** new columns to the dataset: `question_en`, `answer_en`, and optionally `context_en`.
4. **Saves** the output as a `.jsonl` file.

---

## Dependencies

Make sure the following packages are installed:

```bash
pip install openai datasets tqdm
```

---

## OpenAI API Key

Set your OpenAI API key in the environment before running:

```bash
export OPENAI_API_KEY='your-key-here'
```

> **Note:** In the script, the key is hardcoded for demonstration. It is recommended to load it securely from the environment or `.env` file instead.

---

## Input File Format

- The input file must be a `.jsonl` (JSON Lines) file.
- Each line should represent a JSON object with at least the following fields:
  - `"question"` (in German)
  - `"answer"` (in German)
- Optionally, the dataset can include a `"context"` field, which can be a string or list of strings.

**Example line in `.jsonl`:**

```json
{"question": "Was ist KI?", "answer": "Künstliche Intelligenz ist ...", "context": ["Einleitung zur KI", "Anwendungsfälle"]}
```

---

## How It Works (Code Flow)

### 1. **Argument Parsing** (`main()`)

The script accepts the following CLI arguments:
- `--path`: Path to the input `.jsonl` file
- `--samples`: Number of samples to translate (default: 10)
- `--output_path`: Output path for the translated `.jsonl` file
- `--model`: OpenAI model name (default: `gpt-3.5-turbo`)

### 2. **Loading the Dataset**

```python
full_dataset = load_dataset("json", data_files=args.path, split="train")
```

Uses Hugging Face's `datasets` library to read the JSONL file.

### 3. **Selecting Samples**

```python
dataset = full_dataset.select(range(min(args.samples, len(full_dataset))))
```

Limits processing to `n` samples for faster testing.

### 4. **Translation Process**

Each field is passed to `translate_text()` which:
- Sends a prompt to the OpenAI API:  
  _"Translate the following German sentence to English..."_
- Receives the translated text
- Returns the clean translation

### 5. **Column Translation Function**

```python
translate_column_samplewise(dataset, column, model, preprocess_fn)
```

- Iterates through each item in the specified column
- Applies optional preprocessing (e.g., flattening context)
- Collects and returns all translations as a list

### 6. **Saving the Output**

The translated dataset is saved in `.jsonl` format using:

```python
dataset.to_json(args.output_path, orient="records", lines=True)
```

---

## Assumptions

- The input dataset must include at least `"question"` and `"answer"` fields.
- The `"context"` field is optional but will be translated if present.
- You must have a valid OpenAI API key and quota.

---

## Sample Run Command

```bash
python translate_pipeline.py \
  --path data/german_qa.jsonl \
  --samples 20 \
  --output_path data/translated_qa.jsonl \
  --model gpt-3.5-turbo
```

This command:
- Translates the first 20 samples
- Uses `gpt-3.5-turbo`
- Reads from `data/german_qa.jsonl`
- Saves results to `data/translated_qa.jsonl`

---

## Output

The output file will contain the original fields and these new translated fields:
- `question_en`
- `answer_en`
- `context_en` (if `context` exists)

---

## Example Output Entry

```json
{
  "question": "Was ist KI?",
  "question_en": "What is AI?",
  "answer": "Künstliche Intelligenz ist ...",
  "answer_en": "Artificial intelligence is ...",
  "context": ["Einleitung zur KI", "Anwendungsfälle"],
  "context_en": "Introduction to AI Application cases"
}
```
