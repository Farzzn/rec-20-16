import os
import sys
import glob

print("--> SCRIPT STARTED SUCCESSFULLY")

try:
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.llms.google_genai import GoogleGenAI
    print("--> LLAMAINDEX CORE LIBRARIES IMPORTED CORRECTLY")
except ImportError as e:
    print(f"--> IMPORT ERROR: {e}")
    sys.exit(1)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("--> ERROR: GEMINI_API_KEY environment variable is not set!")
    sys.exit(1)

# 1. LLM for Generation ONLY (No embeddings used!)
llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)

# 2. Custom Pure-Python Retriever (Upgraded with Stop-Word Filtering & Doc Boosting)
class CustomPythonRetriever:
    def __init__(self, nodes):
        self.nodes = nodes
        self.node_texts = [n.get_content().lower() for n in nodes]
        print("--> Custom Search Engine built successfully! Quota limits bypassed.")
        
    def retrieve(self, query, top_k=20):
        # 1. Filter out filler words so we only search for the hard data
        stop_words = {"what", "did", "report", "as", "its", "it", "in", "the", "from", "a", "an", "is", "for", "to", "and", "of", "on"}
        raw_terms = set(query.lower().replace("?", "").replace(".", "").split())
        query_terms = raw_terms - stop_words
        
        # 2. Smart routing: If they ask for Microsoft, boost MSFT.pdf to the top!
        company_map = {"microsoft": "msft", "apple": "aapl", "amazon": "amzn", "intel": "intc", "nvidia": "nvda"}
        target_file_tag = None
        for term in raw_terms:
            if term in company_map:
                target_file_tag = company_map[term]
            elif term in company_map.values():
                target_file_tag = term

        scores = []
        for idx, text in enumerate(self.node_texts):
            # Base score: count matching important keywords (net, cash, operating, etc.)
            score = sum(1 for term in query_terms if term in text)
            
            # Massive Boost: Force the correct company's documents to the top of the pile
            file_name = self.nodes[idx].metadata.get('file_name', '').lower()
            if target_file_tag and target_file_tag in file_name:
                score += 50 
                
            scores.append((score, self.nodes[idx]))
            
        # Sort by highest score and return the best chunks
        scores.sort(key=lambda x: x[0], reverse=True)
        return [node for score, node in scores[:top_k]]

def load_and_index_documents():
    print("--> Scanning root directory for PDFs...")
    pdf_files = glob.glob("*.pdf")
    
    if len(pdf_files) == 0:
        print("--> ERROR: No PDF files found.")
        sys.exit(1)
        
    print(f"--> Processing {len(pdf_files)} files FULLY (No truncation):")
    for filepath in pdf_files:
        print(f"  - {os.path.basename(filepath)}")

    print("--> Parsing files with SimpleDirectoryReader...")
    reader = SimpleDirectoryReader(input_files=pdf_files)
    documents = reader.load_data()
    
    print("--> Splitting documents into memory nodes...")
    # Keep chunk size large to capture entire tables
    splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(documents)
    
    print("--> Building Custom Pure-Python Retriever (Zero C++ Dependencies)...")
    retriever = CustomPythonRetriever(nodes)
    return retriever

def ask_question(retriever, question):
    # Fetch top 20 chunks to guarantee the table is included
    top_nodes = retriever.retrieve(question, top_k=20)
    
    context_blocks = []
    used_sources = set()
    
    for idx, node in enumerate(top_nodes):
        source_name = node.metadata.get('file_name', 'Unknown Document')
        used_sources.add(source_name)
        context_blocks.append(f"--- BLOCK {idx+1} (Source: {source_name}) ---\n{node.get_content()}\n")
        
    context_str = "\n".join(context_blocks)
    
    qa_prompt = f"""You are a professional financial data analyst.
Answer the user's question using STRICTLY the provided context blocks below.
If the context does not contain the answer,  Do not extrapolate or guess.
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