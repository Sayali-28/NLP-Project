"""RAG answer generation: retrieved passages + question -> answer (FLAN-T5, runs locally)."""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

GEN_MODEL = "google/flan-t5-base"  # use "google/flan-t5-large" for better answers (needs more RAM)
MIN_SCORE = 0.15


def load_generator(name: str = GEN_MODEL):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tok, model


def answer_question(generator, index, question: str, top_k: int = 3):
    """Return (answer, [(chunk, score), ...])."""
    hits = index.search(question, top_k=top_k)
    if not hits or hits[0][1] < MIN_SCORE:
        return "The answer was not found in the paper.", hits
    tok, model = generator
    context = "\n\n".join(c.text for c, _ in hits)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say 'Not found in the paper.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = model.generate(**inputs, max_new_tokens=200, num_beams=4, early_stopping=True)
    answer = tok.decode(out[0], skip_special_tokens=True).strip()
    return answer or "The answer was not found in the paper.", hits
