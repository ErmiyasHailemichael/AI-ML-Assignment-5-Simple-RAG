# Simple Retrieval-Augmented Generation (RAG) System

**Author:** Ermiyas H.
**Course:** AI/ML Assignment 5  

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Models Used](#models-used)
3. [Knowledge Base](#knowledge-base)
4. [RAG Pipeline Architecture](#rag-pipeline-architecture)
5. [Test Cases and Results](#test-cases-and-results)
6. [Retrieval Analysis](#retrieval-analysis)
7. [RAG vs Raw LLM Comparison](#rag-vs-raw-llm-comparison)
8. [Limitations and Future Improvements](#limitations-and-future-improvements)
9. [Installation and Usage](#installation-and-usage)
10. [References](#references)

---

## Project Overview

This project implements a basic **Retrieval-Augmented Generation (RAG)** system to answer questions about a custom knowledge base. The system demonstrates how RAG can ground Large Language Models (LLMs) in specific, factual information and mitigate hallucination issues.

**Key Features:**
- Custom knowledge base about a fictional coffee shop
- Semantic search using sentence embeddings
- Context-aware answer generation using a pre-trained LLM
- Three comprehensive test cases demonstrating different RAG capabilities

---

## Models Used

### Embedding Model
- **Model Name:** `all-MiniLM-L6-v2`
- **Source:** Sentence Transformers (Hugging Face)
- **Embedding Dimension:** 384
- **Purpose:** Convert text chunks and queries into dense vector representations for semantic similarity search

**Why this model?**
- Lightweight and efficient (only 80MB)
- Good balance between speed and quality
- Well-suited for semantic search tasks
- Pre-trained on large corpus of sentence pairs

### Language Model (LLM)
- **Model Name:** `google/flan-t5-base`
- **Source:** Google/Hugging Face Transformers
- **Model Size:** 250M parameters
- **Purpose:** Generate natural language answers based on retrieved context

**Why this model?**
- Instruction-tuned for better following prompts
- Strong performance on question-answering tasks
- Reasonable size for Google Colab environment
- Good at grounding answers in provided context

---

## Knowledge Base

### Topic
**The Moonbean Café** - A fictional specialty coffee shop in Portland, Oregon

### Content Structure
The knowledge base consists of **3 paragraphs** (chunks) covering:

1. **Chunk 1:** Business information, locations, and hours
   - Founding year and founder
   - Store hours for each day of the week
   - Location addresses
   - Loyalty program details

2. **Chunk 2:** Menu items and pricing
   - Signature drinks with descriptions and prices
   - Classic espresso drink pricing
   - Coffee sourcing information
   - Dairy alternative options

3. **Chunk 3:** Policies and dietary accommodations
   - Dietary restriction accommodations
   - Pastry offerings (vegan/gluten-free options)
   - Return and refund policy
   - Catering requirements

**Total Knowledge Base Size:** ~450 words across 3 chunks

---

## RAG Pipeline Architecture

### Overview
```
User Query → Embedding → Similarity Search → Context Retrieval → Prompt Construction → LLM Generation → Answer
```

### Detailed Steps

#### 1. Knowledge Base Preparation
- Split raw text into logical chunks (by paragraph)
- Generate embeddings for each chunk using Sentence Transformer
- Store embeddings in memory (NumPy array)

#### 2. Query Processing
- Receive user query as input
- Generate query embedding using the same Sentence Transformer model
- Ensures query and KB chunks exist in the same semantic space

#### 3. Retrieval (Semantic Search)
- Calculate cosine similarity between query embedding and all KB chunk embeddings
- Rank chunks by similarity score (0 to 1, higher = more relevant)
- Retrieve top-k most relevant chunks (k=2 in this implementation)

#### 4. Prompt Construction
- Combine retrieved chunks into a single context string
- Create structured prompt with:
  - Instruction to answer based only on provided context
  - The retrieved context
  - The original user query
  - Instruction to say "I don't have that information" if answer not in context

#### 5. Answer Generation
- Pass constructed prompt to FLAN-T5 model
- Generate answer using beam search (num_beams=4)
- Return final answer to user

---

## Test Cases and Results

### Test Case 1: Factual Question (Direct KB Answer)

**Query:**  
`"What are the hours for The Moonbean Café on Saturday?"`

**Expected Behavior:**  
Should retrieve the chunk containing hours information and provide accurate answer.

**Retrieved Chunks:**
- **Rank 1:** Chunk 1 (Similarity: 0.7147)
- **Rank 2:** Chunk 3 (Similarity: 0.3646)

**Generated Answer:**  
`"7:00 AM to 9:00 PM"`

**Analysis:**  
✅ **SUCCESS** - The RAG system correctly:
- Retrieved the most relevant chunk (Chunk 1 contains hours)
- Extracted the specific information requested
- Generated a concise, accurate answer
- Did not hallucinate or add information not in the KB

---

### Test Case 2: Foil/General Question (NOT in KB)

**Query:**  
`"Does The Moonbean Café serve pizza?"`

**Expected Behavior:**  
Should recognize that pizza is not mentioned in the KB and either say "no" or indicate the information is not available.

**Retrieved Chunks:**
- **Rank 1:** Chunk 1 (Similarity: 0.6761)
- **Rank 2:** Chunk 3 (Similarity: 0.3763)

**Generated Answer:**  
`"No"`

**Analysis:**  
✅ **SUCCESS** - The RAG system correctly:
- Recognized that pizza is not mentioned in any retrieved chunks
- Avoided hallucinating a menu item not in the KB
- Provided a clear negative response
- Demonstrated the value of grounding in preventing false information

**Key Insight:**  
Without RAG, an LLM might hallucinate details about a coffee shop serving pizza or provide generic information. The retrieval-based grounding helps prevent this.

---

### Test Case 3: Synthesis Question (Multiple Chunks)

**Query:**  
`"I'm lactose intolerant and only have $5. What drink options do I have at The Moonbean Café?"`

**Expected Behavior:**  
Should synthesize information from multiple chunks:
- Dairy alternatives from Chunk 2
- Pricing information from Chunk 2
- Combine both to recommend suitable options

**Retrieved Chunks:**
- **Rank 1:** Chunk 2 (Similarity: 0.6186) - Contains menu and pricing
- **Rank 2:** Chunk 1 (Similarity: 0.4494) - Contains general info

**Generated Answer:**  
`"Moonbeam Latte"`

**Analysis:**  
⚠️ **PARTIAL SUCCESS** - The RAG system:
- ✅ Retrieved the correct chunks containing both pricing and dairy alternatives
- ✅ Identified a drink from the menu
- ❌ Suggested Moonbeam Latte ($5.50) which exceeds the $5 budget
- ❌ Did not explicitly mention dairy-free milk alternatives

**What Should Have Been Answered:**  
"The Cosmic Cold Brew ($4.75) or a regular cappuccino ($3.50), both available with oat milk, almond milk, or soy milk at no extra charge."

**Root Cause:**  
The LLM struggled to synthesize multiple constraints (budget + dietary restriction + available options). This demonstrates a limitation of smaller LLMs in complex reasoning tasks.

---

## Retrieval Analysis

### Summary of Retrieval Performance

| Test Case | Query Type | Top Chunk | Similarity Score | Correct? |
|-----------|------------|-----------|------------------|----------|
| 1 | Factual | Chunk 1 | 0.7147 | ✅ Yes |
| 2 | Foil | Chunk 1 | 0.6761 | ✅ Yes |
| 3 | Synthesis | Chunk 2 | 0.6186 | ✅ Yes |

### Key Observations

1. **Similarity Scores:**
   - Range: 0.17 to 0.71
   - Higher scores consistently corresponded to more relevant chunks
   - Even "lower" scores (0.24) still retrieved semantically relevant content

2. **Retrieval Accuracy:**
   - 100% accuracy in identifying the most relevant chunk for each query
   - The semantic search correctly prioritized chunks based on query intent

3. **Top-2 Retrieval Strategy:**
   - Retrieving 2 chunks provided good context without overwhelming the LLM
   - Second-ranked chunks often provided supporting information

---

## RAG vs Raw LLM Comparison

### Experiment Setup
To demonstrate the value of RAG, I compared the same query with and without retrieval:

**Query:** `"What are the hours for The Moonbean Café on Saturday?"`

### Results

| Approach | Answer | Grounded in Facts? | Accuracy |
|----------|--------|-------------------|----------|
| **RAG System** | "7:00 AM to 9:00 PM" | ✅ Yes (from KB) | 100% Accurate |
| **Raw LLM (no context)** | Generic/Hallucinated response | ❌ No | Unknown/Incorrect |

### Analysis

**RAG System Benefits:**
1. **Factual Grounding:** Answer is directly extracted from knowledge base
2. **No Hallucination:** LLM cannot invent hours that don't exist in KB
3. **Verifiable:** Can trace answer back to specific chunk in KB
4. **Up-to-date:** If KB is updated, answers automatically reflect changes

**Raw LLM Limitations:**
1. **No Specific Knowledge:** LLM has no information about this fictional café
2. **Hallucination Risk:** May generate plausible-sounding but incorrect hours
3. **Not Verifiable:** No way to confirm where information came from
4. **Static Knowledge:** Cannot update without retraining

### Conclusion
RAG successfully mitigated hallucination by grounding the LLM's responses in the provided knowledge base. This is especially valuable for domain-specific applications where accuracy is critical.

---

## Limitations and Future Improvements

### Current Limitations

1. **Small Knowledge Base**
   - Only 3 chunks limits the system's knowledge scope
   - Real-world applications would need hundreds or thousands of chunks

2. **Simple Chunking Strategy**
   - Currently splits by paragraph breaks
   - May split related information across chunks
   - **Improvement:** Use sliding window or semantic chunking

3. **Limited Context Window**
   - Only retrieves top-2 chunks
   - May miss relevant information in other chunks
   - **Improvement:** Implement re-ranking or dynamic k selection

4. **LLM Synthesis Capability**
   - FLAN-T5-base struggles with complex multi-constraint queries
   - **Improvement:** Use larger models (FLAN-T5-large, GPT-3.5) or fine-tune

5. **No Source Citation**
   - System doesn't cite which chunk the answer came from
   - **Improvement:** Add citation metadata to responses

6. **Static Retrieval**
   - No query expansion or reformulation
   - **Improvement:** Implement iterative retrieval or query decomposition

### Proposed Improvements

#### 1. Enhanced Retrieval
```python
# Implement re-ranking after initial retrieval
# Use cross-encoder models for more accurate ranking
# Add hybrid search (dense + sparse retrieval)
```

#### 2. Better Chunking
```python
# Implement semantic chunking based on topic boundaries
# Add chunk overlap to preserve context
# Optimize chunk size (current: paragraph-based)
```

#### 3. Improved Generation
```python
# Use larger LLMs or GPT-based models
# Add chain-of-thought prompting for complex queries
# Implement self-consistency checking
```

#### 4. Evaluation Metrics
```python
# Add automated evaluation (BLEU, ROUGE, F1)
# Implement human evaluation rubric
# Track retrieval precision/recall
```

---

## Installation and Usage

### Prerequisites
- Python 3.7+
- Google Colab (recommended) or Jupyter Notebook
- Internet connection for model downloads

### Installation
```bash
# Install required packages
pip install sentence-transformers transformers torch scikit-learn
```

### Usage

#### 1. Clone or Download Repository
```bash
git clone https://github.com/[your-username]/AI-ML-Assignment-5-Simple-RAG.git
cd AI-ML-Assignment-5-Simple-RAG
```

#### 2. Open Notebook
- Upload `Simple_RAG_System.ipynb` to Google Colab
- Or open locally: `jupyter notebook Simple_RAG_System.ipynb`

#### 3. Run All Cells
- Execute cells in order from top to bottom
- First run will download models (~300MB total)
- Subsequent runs use cached models

#### 4. Test Custom Queries
```python
# Run custom queries using the RAG pipeline
custom_query = "Your question here"
answer = rag_pipeline(custom_query, top_k=2)
print(answer)
```

### Expected Runtime
- **First run:** 3-5 minutes (model downloads)
- **Subsequent runs:** 30-60 seconds (per test case)

---

## References

### Models
1. **Sentence Transformers:** Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv:1908.10084
2. **FLAN-T5:** Chung, H. W., et al. (2022). Scaling Instruction-Finetuned Language Models. arXiv:2210.11416

### RAG Methodology
3. **RAG Paper:** Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
4. **Dense Passage Retrieval:** Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP 2020.

### Libraries
5. **Hugging Face Transformers:** https://huggingface.co/docs/transformers
6. **Sentence Transformers:** https://www.sbert.net/
7. **Scikit-learn:** https://scikit-learn.org/

---

## Project Structure
```
AI-ML-Assignment-5-Simple-RAG/
│
├── Simple_RAG_System.ipynb          # Main Jupyter notebook
├── README.md                         # This file
├── knowledge_base.txt                # Raw KB text (optional separate file)
└── requirements.txt                  # Python dependencies (optional)
```

---

## Youtube Video
[Link](https://youtu.be/VfgCVrLyt5E)

---

## Acknowledgments

- Hugging Face for providing open-source models
- Sentence Transformers library developers
- Google Colab for free computational resources

---

**End of README**# AI-ML-Assignment-5-Simple-RAG
