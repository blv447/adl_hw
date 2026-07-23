import json

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    """
    Use the CoTModel to generate `oversample` diverse reasoning completions per
    training question (via sampling with temperature > 0). Keep only the first
    completion per question whose parsed answer matches the ground truth (within
    the dataset's tolerance), and write the resulting (question, answer, reasoning)
    triples to `output_json`.
    """
    model = CoTModel()
    trainset = Dataset("train")

    questions = [item[0] for item in trainset]
    correct_answers = [item[1] for item in trainset]

    # Build the CoT-formatted prompt for every question up front.
    prompts = [model.format_prompt(q) for q in questions]

    # base_llm.batched_generate's internal micro-batching only chunks the prompt
    # list - it doesn't account for num_return_sequences multiplying memory use.
    # A "batch of 32" here actually generates 32 * oversample sequences at once,
    # which can OOM on smaller GPUs. So we chunk manually into small groups here,
    # independent of that internal micro_batch_size.
    chunk_size = 4
    generations = []
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        generations.extend(
            model.batched_generate(chunk, num_return_sequences=oversample, temperature=temperature)
        )
    # generations is a list[list[str]]: one sub-list of `oversample` completions per question

    results = []
    for question, correct_answer, completions in zip(questions, correct_answers, generations):
        # Try each sampled completion in order and keep the first one whose parsed
        # answer is correct. If none match, skip this question entirely.
        for completion in completions:
            parsed = model.parse_answer(completion)
            if parsed == parsed and is_answer_valid(parsed, correct_answer):  # parsed == parsed filters out NaN
                results.append([question, correct_answer, completion])
                break

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} / {len(questions)} correct reasoning traces -> {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)