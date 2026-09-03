"""
The generation half of the RAG support chatbot.

The retrieval half (whmcs_export.py, pinecone_script.py, build_index.py) turns past
support tickets into searchable vectors. This module takes a user question, retrieves
the most similar past tickets, and asks a local instruction-tuned model to answer
using only those tickets as context.

Two design choices worth noting:
  - Prompt construction is separated from generation (build_prompt) so the retrieval
    and prompting logic can be tested without loading model weights.
  - If the best retrieval score falls below MIN_SCORE the model is never called and
    the bot says it does not know, rather than inventing an answer from nothing.

Usage:
    python chatbot.py --question "How do I increase the upload size limit?"
    python chatbot.py                      # interactive loop

Requires:
    pip install torch transformers numpy
"""

from __future__ import annotations

import argparse

from embeddings import Embedder
from vector_store import LocalVectorStore

DEFAULT_GENERATOR = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_INDEX = "index"
TOP_K = 3
MIN_SCORE = 0.35
MAX_CONTEXT_CHARS = 1200
MAX_NEW_TOKENS = 220

SYSTEM_PROMPT = (
    "You are a support assistant for a MediaWiki hosting company. "
    "Answer the customer's question using only the past support tickets provided. "
    "If the tickets do not contain the answer, say you do not have that information "
    "and suggest opening a ticket. Be concise and specific. Do not invent settings, "
    "prices, or policies that are not in the tickets."
)

NO_ANSWER = (
    "I do not have information about that in past support tickets. "
    "Please open a ticket and a support engineer will help."
)


def format_context(results: list[dict]) -> str:
    """Turn retrieved tickets into a numbered context block for the prompt."""
    blocks = []
    for i, r in enumerate(results, start=1):
        messages = r.get("messages", "")
        if len(messages) > MAX_CONTEXT_CHARS:
            messages = messages[:MAX_CONTEXT_CHARS] + " [truncated]"
        blocks.append(
            f"[Ticket {r['ticket_id']}] Subject: {r['subject']}\n{messages}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, results: list[dict]) -> list[dict]:
    """Build the chat messages sent to the generator.

    Kept free of any model dependency so it can be unit tested on its own.
    """
    context = format_context(results)
    user_content = (
        f"Past support tickets:\n\n{context}\n\n"
        f"Customer question: {question}\n\n"
        "Answer using only the tickets above, and cite the ticket numbers you used."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class SupportChatbot:
    """Retrieval plus generation over the past-ticket index."""

    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX,
        generator_model: str = DEFAULT_GENERATOR,
        embedder: Embedder | None = None,
        load_generator: bool = True,
    ):
        self.store = LocalVectorStore.load(index_dir)
        print(f"Loaded index with {len(self.store)} tickets")

        self.embedder = embedder or Embedder()

        self.tokenizer = None
        self.model = None
        if load_generator:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"Loading generator {generator_model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
            self.model = AutoModelForCausalLM.from_pretrained(generator_model)
            self.model.eval()
            print("Generator ready.")

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Find the most similar past tickets for a question."""
        vector = self.embedder.encode_one(question)
        return self.store.query(vector, top_k=top_k)

    def generate(self, messages: list[dict]) -> str:
        """Run the local model over prepared chat messages."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Generator was not loaded. Construct with load_generator=True.")

        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def answer(self, question: str, top_k: int = TOP_K) -> dict:
        """Full pipeline: retrieve, check confidence, generate, return with sources."""
        results = self.retrieve(question, top_k=top_k)

        if not results or results[0]["score"] < MIN_SCORE:
            return {
                "answer": NO_ANSWER,
                "sources": [],
                "top_score": results[0]["score"] if results else 0.0,
                "grounded": False,
            }
        results = [r for r in results if r["score"] >= MIN_SCORE]
        messages = build_prompt(question, results)
        answer = self.generate(messages) 

        return {
            "answer": answer,
            "sources": [
                {"ticket_id": r["ticket_id"], "subject": r["subject"], "score": r["score"]}
                for r in results
            ],
            "top_score": results[0]["score"],
            "grounded": True,
        }


def print_answer(result: dict) -> None:
    print("\n" + result["answer"] + "\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['ticket_id']}] {s['subject']}  (score {s['score']:.3f})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the support chatbot a question.")
    parser.add_argument("--question", help="a single question, otherwise start interactive mode")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="index directory")
    parser.add_argument("--model", default=DEFAULT_GENERATOR, help="generator model name")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="tickets to retrieve")
    args = parser.parse_args()

    bot = SupportChatbot(index_dir=args.index, generator_model=args.model)

    if args.question:
        print_answer(bot.answer(args.question, top_k=args.top_k))
        return

    print("Ask a question, or press Enter on an empty line to quit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        print_answer(bot.answer(question, top_k=args.top_k))


if __name__ == "__main__":
    main()
