from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a unit conversion expert. For each question, give the "
                    "conversion factor, do the multiplication in one short line, then "
                    "immediately end with the final answer wrapped in <answer></answer> "
                    "tags. The content inside <answer></answer> must be ONLY a plain "
                    "number with no units and no extra text. For example, always write "
                    "<answer>2048</answer>, never <answer>2048 MB</answer> or "
                    "<answer>2048 bits</answer>. Always close the tag, even for rate "
                    "conversions like mi/h to m/s. Do not add any text after the "
                    "closing </answer> tag. Be concise."
                ),
            },
            # One-shot example: simple single-step conversion. Uses inch->cm (not a
            # unit pair likely to appear verbatim in the test set) so the model learns
            # the STRUCTURE of the reasoning rather than memorizing these numbers.
            {
                "role": "user",
                "content": "How many centimetres are there in 5 inch?",
            },
            {
                "role": "assistant",
                "content": "1 inch = 2.54 cm. 5 * 2.54 = <answer>12.7</answer>",
            },
            # Second example: a rate/compound conversion, since these are the cases
            # where the model tends to skip the <answer> tag entirely.
            {
                "role": "user",
                "content": "How do we express 10 km/h in terms of m/s?",
            },
            {
                "role": "assistant",
                "content": "1 km/h = 0.277778 m/s. 10 * 0.277778 = <answer>2.77778</answer>",
            },
            # Third example: memory-size units use DECIMAL factors here (1 GB = 1000 MB),
            # not binary (1024). The model otherwise defaults to binary factors, which is
            # wrong for this dataset's convention.
            {
                "role": "user",
                "content": "How many MB is 3 GB?",
            },
            {
                "role": "assistant",
                "content": "1 GB = 1000 MB. 3 * 1000 = <answer>3000</answer>",
            },
            # Actual question to answer
            {
                "role": "user",
                "content": question,
            },
        ]

        # Convert chat messages into the model's expected prompt string, and
        # append the generation prompt so the model knows to start its reply.
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})