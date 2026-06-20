import os
import re
import sys
import glob

print("--> SCRIPT STARTED SUCCESSFULLY")

try:
    import fitz  # PyMuPDF
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.llms.google_genai import GoogleGenAI
    from rank_bm25 import BM25Okapi
    print("--> LLAMAINDEX CORE LIBRARIES + PYMUPDF + BM25 IMPORTED CORRECTLY")
except ImportError as e:
    print(f"--> IMPORT ERROR: {e}")
    print("--> If this is fitz, run: pip install pymupdf")
    print("--> If this is rank_bm25, run: pip install rank-bm25")
    sys.exit(1)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("--> ERROR: GEMINI_API_KEY environment variable is not set!")
    sys.exit(1)

# 1. LLM for Generation ONLY (No embeddings used!)
llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)

# Words that carry no retrieval signal on their own (used for company/intent detection,
# NOT for BM25 itself -- BM25's own IDF already down-weights common words in the corpus).
STOP_WORDS = {
    "what", "did", "report", "reports", "as", "its", "it", "in", "the", "from",
    "a", "an", "is", "for", "to", "and", "of", "on", "did", "was", "were",
}

# Words that signal "give me an overview" rather than "find this specific number".
GENERIC_INTENT_WORDS = {
    "explain", "summarize", "summarise", "describe", "overview", "tell",
    "about", "give", "provide", "detail", "details", "discuss", "walk",
}

# Used only for the generic/vague-query fallback, to bias toward substantive
# financial sections instead of cover pages / boilerplate.
FINANCIAL_KEYWORDS = [
    "revenue", "net income", "operating income", "operating margin",
    "cash flow", "gross margin", "earnings", "expenses", "assets",
    "liabilities", "eps", "guidance", "segment", "diluted", "income tax",
]

COMPANY_MAP = {"microsoft": "msft", "apple": "aapl", "amazon": "amzn", "intel": "intc", "nvidia": "nvda"}


def tokenize(text):
    """Lowercase alphanumeric tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """
    BM25-ranked retriever over the node corpus, with:
      - a large additive boost for nodes belonging to a company explicitly
        named in the query (so "microsoft" pulls from MSFT.pdf, not just
        whatever scores highest globally), and
      - a fallback path for vague/"explain the report" style queries, where
        plain term-overlap (and even BM25) has little signal to work with.
        In that case we rank by financial-keyword density and pick chunks
        spread across the beginning/middle/end of the filtered document set,
        instead of silently defaulting to document order.
    """

    def __init__(self, nodes):
        self.nodes = nodes
        self.node_texts_lower = [n.get_content().lower() for n in nodes]
        self.corpus_tokens = [tokenize(t) for t in self.node_texts_lower]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"--> BM25 index built over {len(nodes)} chunks. Quota limits bypassed.")

    def _detect_company(self, raw_terms):
        for term in raw_terms:
            if term in COMPANY_MAP:
                return COMPANY_MAP[term]
            if term in COMPANY_MAP.values():
                return term
        return None

    def _is_generic_query(self, raw_terms, target_file_tag):
        # Strip stopwords, the company word itself, and generic-intent words.
        leftover = raw_terms - STOP_WORDS - GENERIC_INTENT_WORDS
        leftover -= set(COMPANY_MAP.keys()) | set(COMPANY_MAP.values())
        return len(leftover) == 0

    def _candidate_indices(self, target_file_tag):
        if not target_file_tag:
            return list(range(len(self.nodes)))
        return [
            i for i, n in enumerate(self.nodes)
            if target_file_tag in n.metadata.get("file_name", "").lower()
        ] or list(range(len(self.nodes)))  # fallback: no filename match, search everything

    def _generic_fallback(self, target_file_tag, top_k):
        """Spread selection across early/middle/late thirds of the candidate
        set, ranked within each third by financial-keyword density, so a
        vague 'explain the report' query doesn't just return the cover page
        and table of contents in document order."""
        candidates = self._candidate_indices(target_file_tag)

        def kw_score(idx):
            text = self.node_texts_lower[idx]
            return sum(text.count(kw) for kw in FINANCIAL_KEYWORDS)

        scored = sorted(candidates, key=kw_score, reverse=True)

        # Split candidates into thirds by their ORIGINAL document position
        # (not by score) so we get structural coverage of the filing.
        ordered_by_position = sorted(candidates)
        n = len(ordered_by_position)
        thirds = [
            set(ordered_by_position[: n // 3]),
            set(ordered_by_position[n // 3: 2 * n // 3]),
            set(ordered_by_position[2 * n // 3:]),
        ]

        picked, seen = [], set()
        # Round-robin: take the best-scoring not-yet-picked chunk from each
        # third in turn until top_k is filled.
        while len(picked) < top_k and len(seen) < n:
            for bucket in thirds:
                for idx in scored:
                    if idx in bucket and idx not in seen:
                        picked.append(idx)
                        seen.add(idx)
                        break
                if len(picked) >= top_k:
                    break
        return [self.nodes[i] for i in picked[:top_k]]

    def retrieve(self, query, top_k=10):
        raw_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        target_file_tag = self._detect_company(raw_terms)

        if self._is_generic_query(raw_terms, target_file_tag):
            print("--> Detected a vague/summary-style query -> using financial-keyword "
                  "+ document-coverage fallback instead of BM25 term matching.")
            return self._generic_fallback(target_file_tag, top_k)

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        if target_file_tag:
            for idx, n in enumerate(self.nodes):
                if target_file_tag in n.metadata.get("file_name", "").lower():
                    scores[idx] += 50  # same large company boost as before, now on top of BM25

        ranked = sorted(range(len(self.nodes)), key=lambda i: scores[i], reverse=True)
        return [self.nodes[i] for i in ranked[:top_k]]


def load_and_index_documents():
    print("--> Scanning root directory for PDFs...")
    pdf_files = glob.glob("*.pdf")

    if len(pdf_files) == 0:
        print("--> ERROR: No PDF files found.")
        sys.exit(1)

    print(f"--> Processing {len(pdf_files)} files FULLY (No truncation):")
    documents = []
    for filepath in pdf_files:
        file_name = os.path.basename(filepath)
        print(f"  - {file_name}")
        try:
            pdf = fitz.open(filepath)
        except Exception as e:
            print(f"  --> FAILED TO OPEN {file_name}: {e}")
            continue

        empty_pages = 0
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                documents.append(Document(
                    text=text,
                    metadata={"file_name": file_name, "page_label": str(page_num)},
                ))
            else:
                empty_pages += 1
        pdf.close()

        if empty_pages:
            print(f"  --> WARNING: {empty_pages} page(s) in {file_name} returned no text "
                  f"(likely scanned/image-only pages -- would need OCR, not handled here)")

    print(f"--> Extracted {len(documents)} text-bearing pages via PyMuPDF.")
    if len(documents) == 0:
        print("--> ERROR: PyMuPDF extracted zero usable pages from all PDFs. Check the files.")
        sys.exit(1)

    print("--> Splitting documents into memory nodes...")
    # Keep chunk size large to capture entire tables
    splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(documents)

    print("--> Building BM25 Retriever (Zero C++ Dependencies)...")
    retriever = BM25Retriever(nodes)
    return retriever


def ask_question(retriever, question):
    # Fetch top 20 chunks to guarantee the table is included
    top_nodes = retriever.retrieve(question, top_k=10)

    context_blocks = []
    used_sources = set()

    for idx, node in enumerate(top_nodes):
        source_name = node.metadata.get('file_name', 'Unknown Document')
        used_sources.add(source_name)
        context_blocks.append(f"--- BLOCK {idx+1} (Source: {source_name}) ---\n{node.get_content()}\n")

    context_str = "\n".join(context_blocks)

    # Dump exactly what's being sent to the LLM, so a bad answer can always
    # be checked against what context it actually had to work with.
    try:
        with open("last_context.txt", "w", encoding="utf-8") as f:
            f.write(f"QUESTION: {question}\n\n{context_str}")
    except OSError:
        pass  # debug dump is best-effort, never block the actual query

    qa_prompt = f"""You are a professional financial data analyst.
Answer the user's question using STRICTLY the provided context blocks below.
If the context does not contain the answer, reply EXACTLY with: 'The provided documents do not contain the answer.' Do not extrapolate or guess.
For every factual claim or data metric you pull, you MUST append the exact source document file_name in brackets at the end of the sentence, e.g., [2022 Q3 MSFT.pdf].

Context:
{context_str}

Question: {question}
Answer: """

    response = llm.complete(qa_prompt)

    print("\n" + "="*20 + " SYSTEM ANSWER " + "="*20)
    print(response.text)
    print("="*55)

    print("\n--- RETRIEVED SOURCE DOCUMENTS ---")
    for idx, source in enumerate(used_sources, 1):
        print(f"{idx}. {source}")
    print("(Full retrieved context written to last_context.txt for debugging)")
    print("-" * 55)


if __name__ == "__main__":
    retriever = load_and_index_documents()

    while True:
        try:
            user_q = input("\nEnter your query (or type 'exit' to quit): ")
            if user_q.strip().lower() == 'exit':
                break
            if not user_q.strip():
                continue
            ask_question(retriever, user_q)
        except KeyboardInterrupt:
            break
