# Jaggaer RAG Task - Multi-Document Financial Q&A

## System Architecture & Engineering Decisions
This system utilizes **LlamaIndex** connected to **Gemini 2.5 Flash** for generation. 

Due to the constraints of the testing environment, I implemented a custom architecture:
1. **Bypassing API Rate Limits:** Attempting to pass 5 uncut SEC 10-Q reports (400+ pages of tabular data) to a cloud Embedding API triggered `429 RESOURCE_EXHAUSTED` quotas. 
2. **Handling Environment Constraints:** Attempting to run local HuggingFace embeddings failed due to missing C++ Build Tools (`c10.dll`) on the locked-down machine.
3. **The Solution:** I engineered a custom, pure-Python Sparse Retriever using TF-IDF style keyword matching. This bypassed all C++ dependencies and API quotas. 

## Guardrails
The system features strict anti-hallucination prompt engineering. If a document does not explicitly contain the answer (e.g., asking "Explain the reports"), the LLM is constrained to output exactly: *"The provided documents do not contain the answer."* Every factual claim pulled successfully is cited with the exact source filename.