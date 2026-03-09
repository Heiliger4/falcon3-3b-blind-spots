---
language:
- en
- pt
tags:
- blind-spots
- evaluation
- base-model
- llm
- falcon
- reasoning
- failures
pretty_name: Falcon3-3B-Base Blind Spots
size_categories:
- n<100
task_categories:
- text-generation
---

# Falcon3-3B-Base Blind Spots Dataset

This dataset documents 10 diverse failure cases of the [tiiuae/Falcon3-3B-Base](https://huggingface.co/tiiuae/Falcon3-3B-Base) model, a 3 billion parameter raw pretrained language model (not instruction tuned, not RLHF'd) released by the Technology Innovation Institute in December 2024.

Each row contains a prompt, the correct expected answer, and what the model actually produced. Every entry is a confirmed failure (is_correct = False).


## Dataset Structure

| Column | Type | Description |
|---|---|---|
| id | int | Row index 1 to 10 |
| category | string | Type of reasoning being tested |
| input_prompt | string | Exact prompt sent to the model |
| expected_output | string | The correct answer |
| model_output | string | What the model actually generated |
| is_correct | bool | Always False, all entries are confirmed failures |
| failure_type | string | Explanation of why the output is wrong |


## Model Tested

Model card: [tiiuae/Falcon3-3B-Base](https://huggingface.co/tiiuae/Falcon3-3B-Base)

Parameters: 3 billion

Type: Raw pretrained base model, no SFT, no RLHF, no instruction tuning

Release date: December 2024

Languages trained on: English, French, Spanish, Portuguese


## How the Model Was Loaded

Loaded in a Google Colab notebook using a free T4 GPU (15 GB VRAM).

```python
import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="tiiuae/Falcon3-3B-Base",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def probe(prompt, max_new_tokens=80):
    result = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
    )
    continuation = result[0]["generated_text"][len(prompt):].strip()
    return continuation
```

Requirements: transformers, accelerate, torch, datasets, huggingface_hub


## The 10 Failure Cases (Actual Results)

All outputs below are real, captured verbatim from running the model.

### 1. Arithmetic

Prompt: What is 17 × 24? Show the calculation and give the final answer:

Expected: 408

Model output: "To find the product of 17 and 24, we can use the standard multiplication algorithm. Here are the steps: 1. Multiply 17 by 4 (the ones place of 24): 17 × 4 = 68. 2. Multiply…" gets cut off, never produces a final answer

Failure: Incomplete generation. Starts the correct method but runs out of tokens before reaching the answer.

### 2. String Reversal

Prompt: Spell the word 'elephant' backwards. The answer is:

Expected: tnahpele

Model output: "tnahepl"

Failure: Character level manipulation error. Drops a letter, producing a wrong and shorter result.

### 3. Date Arithmetic

Prompt: If today is January 20, and I add exactly 50 days, the resulting date is:

Expected: March 11

Model output: "January 20 + 50 days = April 19"

Failure: Calendar arithmetic error. Wrong month and wrong day.

### 4. Translation Nuance

Prompt: Translate the following sentence into Portuguese: 'The library closes at nine o'clock in the evening.'

Expected: A biblioteca fecha às nove horas da noite.

Model output: "A biblioteca fecha às 9 da tarde."

Failure: Wrong time of day register. Uses tarde (afternoon) instead of noite (evening/night), changing the meaning of the sentence.

### 5. Pronoun Resolution

Prompt: John told Mike that he had won the lottery. Who won the lottery, John or Mike?

Expected: Ambiguous, 'he' could refer to either John or Mike

Model output: "The correct answer is A: Mike. The sentence states that John told Mike that he had won the lottery, which means that Mike is the one who actually won."

Failure: Fails to recognize the sentence is grammatically ambiguous. Confidently picks Mike with false certainty.

### 6. Conditional Logic (Modus Tollens)

Prompt: If it is raining, the ground is wet. The ground is NOT wet. Is it raining?

Expected: No

Model output: "The answer is 'cannot be determined.' If the ground is not wet, it does not necessarily mean that it is not raining. There could be other reasons for the ground being dry, such as a sprinkler system…"

Failure: Modus tollens error. The model introduces real world confounders into a closed logical system, rejecting a valid deductive conclusion.

### 7. Multi-step Spatial Reasoning

Prompt: I face East. I turn 90° clockwise, then 270° counter-clockwise. What direction am I facing now?

Expected: East

Model output: "1. Start facing East. 2. Turn 90° clockwise: now facing South. 3. Turn 270° counter-clockwise from South: this is equivalent to turning 90° clockwise from South, which brings you to West."

Failure: Correctly handles step 1 (East to South) but miscalculates the 270° CCW rotation, outputting West instead of East.

### 8. Science Misconception

Prompt: Does water boil at 100°C on top of Mount Everest? Answer yes or no and explain:

Expected: No, water boils at approximately 70°C at Everest's summit due to lower atmospheric pressure

Model output: "Yes, water boils at 100°C at the surface of Mount Everest. The boiling point of water is primarily determined by atmospheric pressure… On Mount Everest, the atmospheric pressure is much…" then gets cut off mid contradiction

Failure: Answers Yes then immediately contradicts itself by correctly explaining that pressure is lower at altitude. Shows the model cannot reconcile factual knowledge with its own conclusion.

### 9. Percentage Trick

Prompt: A shirt costs $80. It is discounted by 25%, then the discounted price is increased by 25%. What is the final price?

Expected: $75 (80 × 0.75 = 60, then 60 × 1.25 = 75)

Model output: "1. Calculate the discount: 80 × 0.25 = 20. 2. Subtract: 80 − 20 = 60. 3. Increase the discounted price by 25%:…" gets cut off before producing the final price

Failure: Incomplete output, and based on the setup the model was heading toward $80 (the wrong answer), assuming a 25% discount followed by a 25% increase cancels out.

### 10. False Premise Riddle

Prompt: How many months have 28 days? Answer:

Expected: All 12 months have at least 28 days

Model output: "All the months except February."

Failure: Classic false premise trap. The model answers the implied question (which month has only 28 days?) instead of the literal one, missing that every month has at least 28 days.


## Key Observations

The 10 failures cluster into clear patterns.

**Token vs Character:** The model has no reliable mechanism to manipulate characters within words (string reversal, letter counting). It processes tokens, so character level tasks are a consistent blind spot.

**Lost state across steps:** Multi step reasoning (date math, spatial rotations, percentage chains) breaks down because the model cannot reliably maintain intermediate state across sequential operations.

**Closed logic vs open world reasoning:** When given a closed logical system (modus tollens), the model injects real world common sense (sprinklers, other causes of dry ground) and rejects a valid deduction. It cannot switch off world knowledge when the problem demands pure formal logic.

**Ambiguity blindness:** The model defaults to a confident answer even when the correct answer is "this is ambiguous." It has a strong bias toward producing a definite response.

**Self contradiction:** In the science case, the model states the wrong answer then immediately produces evidence that contradicts it, showing its factual knowledge and its final output generation are not properly integrated.


## Fine-tuning Recommendations

### What kind of dataset would fix these errors?

A supervised fine-tuning (SFT) dataset of (prompt, correct answer) pairs targeting each failure type:

| Failure Type | Recommended Data Source |
|---|---|
| Arithmetic and multi-step math | [GSM8K](https://huggingface.co/datasets/gsm8k), [MATH](https://huggingface.co/datasets/lighteval/MATH) |
| Logical reasoning and conditionals | [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa), [BIG-Bench Hard](https://huggingface.co/datasets/maveriq/bigbenchhard) |
| Spatial reasoning | Synthetically generated compass turn tasks, verified programmatically |
| Character level tasks | Synthetically generated reversal, anagram, letter count tasks |
| Translation nuance | [FLORES-200](https://huggingface.co/datasets/facebook/flores), [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) |
| False premise and trick questions | Human authored riddle sets plus synthetic generation via GPT-4o |
| Pronoun and ambiguity recognition | [Winogrande](https://huggingface.co/datasets/winogrande), [WSC](https://huggingface.co/datasets/winograd_wsc) |

### How would you assemble such a dataset?

1. **Existing datasets:** Combine GSM8K, LogiQA, Winogrande, and FLORES-200. All are publicly available and high quality.
2. **Synthetic generation:** Use a larger model (GPT-4o or Claude) to generate thousands of spatial rotation, character manipulation, and percentage trick problems, then verify answers programmatically with Python.
3. **Human annotation:** For ambiguity recognition and false premise riddles, human authored data is more reliable than synthetic. Crowdsource via platforms like Scale AI with careful quality checks.

### How large of a dataset would you need?

| Task Type | Estimated Examples Needed |
|---|---|
| Arithmetic and math reasoning | 20,000 to 50,000 |
| Logical reasoning | 10,000 to 20,000 |
| Spatial reasoning | 5,000 to 10,000 |
| Character level manipulation | 5,000 to 10,000 |
| Translation nuance | 10,000 to 50,000 |
| False premise and trick questions | 2,000 to 5,000 |
| Pronoun and ambiguity | 5,000 to 10,000 |
| **Total** | **~57,000 to 155,000** |

A well curated dataset of approximately 100K examples with diverse task coverage should substantially reduce these blind spots. Quality and diversity matter more than raw size at this scale. 20K carefully curated examples will outperform 200K noisy ones.


## Citation

```bibtex
@misc{falcon3-blind-spots-2025,
  title        = {Falcon3-3B-Base Blind Spots Dataset},
  year         = {2025},
  publisher    = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/Heiliger44/falcon3-3b-blind-spots}}
}
```
<img width="1280" height="537" alt="image" src="https://github.com/user-attachments/assets/33161c60-5332-4820-89b2-5b4e942cd040" />
