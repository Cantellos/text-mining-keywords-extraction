# Text Mining: Explicit and Implicit Keywords Extraction

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-Text%20Mining-green)](https://en.wikipedia.org/wiki/Text_mining)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LocalMaxs algorithm implementation for extracting explicit and implicit keywords from text documents using cohesion metrics (SCP, Dice, φ²) and semantic proximity analysis.

**Data Preprocessing and Text Mining Project (2023/2024)**  
Universidade Nova de Lisboa - Faculty of Science and Technology

**Team:**
- Julia Cwynar (68846)
- Andrea Cantelli (69609)
- Micol Curci (69675)

## Overview

This project implements a comprehensive text mining pipeline to automatically identify relevant expressions and extract both explicit and implicit keywords from large text corpora. Using the **LocalMaxs algorithm** with three different cohesion metrics, the system analyzes n-grams (up to 7 words) to discover meaningful word combinations and their semantic relationships across documents.

### Key Features

- **Automated Relevant Expression Extraction**: Uses LocalMaxs algorithm with three cohesion metrics
- **Dual Keyword Extraction**: Identifies both explicit (directly mentioned) and implicit (contextually inferred) keywords
- **Multi-Metric Analysis**: Compares SCP, Dice Coefficient, and φ² (phi-squared) metrics
- **Stop Words Detection**: Automatic identification using neighSyl metric (254 stop words)
- **Semantic Proximity**: Calculates inter-document and intra-document relationships
- **TF-IDF Weighting**: Ranks explicit keywords by term importance
- **Large-Scale Processing**: Handles 3,170 documents with 2.2M+ words

## Project Objective

The primary goal is to analyze text document collections and identify the most significant relative expressions (REs) - word combinations that frequently occur together with high contextual significance. The system extracts:

1. **Explicit Keywords**: Top 15 most informative expressions per document (TF-IDF based)
2. **Implicit Keywords**: Top 5 contextually related terms per document (semantic proximity)

This enables efficient content summarization, semantic analysis, and automatic document categorization without manual reading.

## Dataset

**Corpus Statistics:**
- **Documents**: 3,170 text files
- **Total Words**: 2,209,654
- **Content**: Extracted fragments from various topics (philosophy, history, technology)
- **Example Topics**: Aristotle's philosophy, Greek Christian scribes, ASCII encoding

**Sample Text:**
```
Greek Christian scribes played a crucial role in the preservation of 
Aristotle by copying all the extant Greek language manuscripts of the 
corpus. The first Greek Christians to comment extensively on Aristotle 
were John Philoponus, Elias, and David in the sixth century...
```

## Methodology

### 1. Data Preprocessing

**Tokenization:**
- Break down text into individual words
- Add spaces before/after special characters: `,;:!<>()[]&?=+\"'./\\`
- Preserve word boundaries

**Result**: 2,209,654 tokens across 3,170 documents

### 2. N-Grams Generation

Generate all possible n-grams for each document:
- **Range**: n = 1 to n = 7
- **Filtering**: Remove n-grams with special characters at boundaries
- **Minimum Frequency**: At least 2 occurrences

For each n-gram, track:
- Occurrence count per document
- Document frequency (number of documents containing it)
- Position within documents

### 3. Stop Words Detection

Automatic stop word identification using **neighSyl metric**:

```
biGramNeigh(w) = number of unique neighbors (prev + next words)
syllables(w) = vowel_count - vowel_pairs_count

neighSyl(w) = biGramNeigh(w) / syllables(w)
```

**Threshold**: `r = set{w | neighSyl(w) > b}` where b = 254

**Result**: 254 stop words identified (0.01% of corpus)

**Examples**: 'the', 'and', 'of', 'in', 'to', 'a', 'was', 'is', 'for', 'with', 'as', 'on', 'from', 'at', 'that', 'or', 'his', 'an', 'their', 'has', 'had', 'which', 'are', 'were', 'he', 'who', 'it', 'her', 'first', 'be', 'two'...

### 4. Cohesion Metrics

Three metrics evaluate n-gram relevance:

#### Symmetric Conditional Probability (SCP)

Measures mutual dependence between words:

```
SCP(w1,...,wn) = f(w1,...,wn)² / D

D = (1/(n-1)) × Σ f(w1,...,wi,...,wn)
```

**Results**: 16,020 relevant expressions

#### Dice Coefficient

Similarity measure for co-occurrence:

```
Dice(w1,...,wn) = (2 × f(w1,...,wn)) / D

D = (1/(n-1)) × Σ [f(w1,...,wi) + f(wi+1,...,wn)]
```

**Results**: 26,384 relevant expressions (highest)

#### φ² (Phi-Squared)

Statistical association strength:

```
φ²(w1,...,wn) = (N × f(w1,...,wn) - LHS)² / (LHS × RHS)

LHS = (1/(n-1)) × Σ f(w1,...,wi) × f(wi+1,...,wn)
RHS = (1/(n-1)) × Σ f(w1,...,wi) × (N - f(wi+1,...,wn))
```

**Results**: 16,017 relevant expressions

### 5. Evaluation Metrics

#### Precision

Percentage of algorithm-identified expressions that are truly relevant:

```
Precision = Correct REs / Total REs Found
```

**Results:**
- SCP: **0.875** (87.5%) ← Highest precision
- Dice: 0.825 (82.5%)
- φ²: 0.750 (75.0%)

#### Recall

Percentage of human-identified relevant expressions found by algorithm:

```
Recall = Algorithm Found / Human Identified (200)
```

**Results:**
- SCP: 0.035 (3.5%)
- Dice: **0.040** (4.0%) ← Highest recall
- φ²: 0.035 (3.5%)

#### F-Metric

Harmonic mean of Precision and Recall:

```
F-metric = 2 × (Precision × Recall) / (Precision + Recall)
```

**Results:**
- SCP: 0.0673
- Dice: **0.0763** ← Best overall balance
- φ²: 0.0669

### 6. Explicit Keywords Extraction

Use **TF-IDF** (Term Frequency - Inverse Document Frequency) to rank n-grams:

```
TF = freq(RE, dj) / size(dj)

IDF = log(||D|| / ||{d ∈ D ∧ freq(RE,d) > 0}||)

TF-IDF = TF × IDF
```

**Output**: Top 15 highest TF-IDF n-grams per document = Explicit Keywords

### 7. Implicit Keywords Extraction

Identify contextually related terms not directly mentioned.

#### Step 1: Create Candidate Set

- All relevant expressions
- First and last words of each RE
- **Total**: 27,869 unique candidates

#### Step 2: Calculate Inter-Document Proximity (Correlation)

Measures how often terms A and B appear together across documents:

```
Corr(A,B) = Cov(A,B) / (√Cov(A,A) × √Cov(B,B))

Cov(A,B) = (1/(||D||-1)) × Σ (f(A,di) - f(A,.)) × (f(B,di) - f(B,.))
```

High correlation → Strong relationship between terms

#### Step 3: Calculate Semantic Proximity

*(Note: Intra-Document Proximity not implemented due to time constraints)*

```
SemanticProx(A,B) ≈ Corr(A,B)
```

Full formula (not implemented):
```
SemanticProx(A,B) = Corr(A,B) × √IP(A,B)

IP(A,B) = 1 - (1/||D*||) × Σ (dist(A,B,d) / farthest(A,B,d))
```

#### Step 4: Calculate Score

For each candidate RE, sum semantic proximities with top explicit keywords:

```
Score(RE,d) = Σ SemanticProx(RE, ki) / i

ki = i-th ranked explicit keyword of d
```

**Output**: Top 5 highest-scoring candidates per document = Implicit Keywords

## Results Summary

### Cohesion Metrics Comparison

| Metric | REs Found | Precision | Recall | F-Metric | Best For |
|--------|-----------|-----------|--------|----------|----------|
| **SCP** | 16,020 | **0.875** | 0.035 | 0.0673 | High precision |
| **Dice** ✅ | 26,384 | 0.825 | **0.040** | **0.0763** | **Overall balance** |
| **φ²** | 16,017 | 0.750 | 0.035 | 0.0669 | Statistical rigor |

**Conclusion**: **Dice coefficient** provides best overall performance with highest F-metric.

### Output per Document

- **15 Explicit Keywords**: Directly mentioned, high TF-IDF
- **5 Implicit Keywords**: Contextually inferred, high semantic proximity

## Project Structure

```
text-mining-keywords-extraction/
├── data/
│   ├── corpus/                   # 3,170 text files
│   └── stopwords.txt             # 254 identified stop words
├── src/
│   ├── preprocessing.py          # Tokenization and cleaning
│   ├── ngram_generator.py        # N-gram extraction (n=1-7)
│   ├── stopwords_detector.py     # neighSyl calculation
│   ├── cohesion_metrics.py       # SCP, Dice, φ² implementations
│   ├── explicit_keywords.py      # TF-IDF calculation
│   ├── implicit_keywords.py      # Semantic proximity
│   └── evaluation.py             # Precision, Recall, F-metric
├── results/
│   ├── relevant_expressions/     # REs by metric
│   ├── explicit_keywords/        # Top 15 per document
│   └── implicit_keywords/        # Top 5 per document
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cohesion_analysis.ipynb
│   └── 03_evaluation.ipynb
├── report/
│   └── PAD_Project_Report.pdf    # Full academic report
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

### Prerequisites

- Python 3.8+
- NumPy, Pandas for data processing
- NLTK (optional, for additional NLP features)

### Setup

1. **Clone repository**
```bash
git clone https://github.com/Cantellos/text-mining-keywords-extraction.git
cd text-mining-keywords-extraction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess Corpus

```python
from src.preprocessing import tokenize_corpus

# Tokenize all documents
corpus = tokenize_corpus('data/corpus/')
print(f"Total tokens: {sum(len(doc) for doc in corpus)}")
```

### 2. Generate N-Grams

```python
from src.ngram_generator import generate_ngrams

# Generate n-grams (n=1 to 7)
ngrams_dict = {}
for doc in corpus:
    ngrams_dict[doc] = generate_ngrams(doc, max_n=7, min_freq=2)
```

### 3. Detect Stop Words

```python
from src.stopwords_detector import detect_stopwords

# Automatic stop word detection
stopwords = detect_stopwords(corpus, threshold=254)
print(f"Detected {len(stopwords)} stop words")
```

### 4. Calculate Cohesion Metrics

```python
from src.cohesion_metrics import calculate_scp, calculate_dice, calculate_phi2

# Filter n-grams by cohesion
relevant_expressions_scp = calculate_scp(ngrams_dict, stopwords)
relevant_expressions_dice = calculate_dice(ngrams_dict, stopwords)
relevant_expressions_phi2 = calculate_phi2(ngrams_dict, stopwords)

print(f"SCP REs: {len(relevant_expressions_scp)}")
print(f"Dice REs: {len(relevant_expressions_dice)}")
print(f"Phi² REs: {len(relevant_expressions_phi2)}")
```

### 5. Extract Explicit Keywords

```python
from src.explicit_keywords import extract_explicit_keywords

# Top 15 per document using TF-IDF
explicit_kw = extract_explicit_keywords(
    relevant_expressions_dice,  # Use Dice REs
    corpus,
    top_k=15
)

for doc_id, keywords in explicit_kw.items():
    print(f"Document {doc_id}: {keywords[:5]}...")  # Show first 5
```

### 6. Extract Implicit Keywords

```python
from src.implicit_keywords import extract_implicit_keywords

# Top 5 per document using semantic proximity
implicit_kw = extract_implicit_keywords(
    explicit_kw,
    relevant_expressions_dice,
    corpus,
    top_k=5
)

for doc_id, keywords in implicit_kw.items():
    print(f"Document {doc_id} implicit: {keywords}")
```

### 7. Evaluate Algorithm

```python
from src.evaluation import evaluate_precision, evaluate_recall, calculate_fmetric

# Manual evaluation on sample
precision = evaluate_precision(relevant_expressions_dice, sample_size=200)
recall = evaluate_recall(relevant_expressions_dice, human_labeled_sample)
f_metric = calculate_fmetric(precision, recall)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F-Metric: {f_metric:.4f}")
```

## Algorithm Performance

### Execution Time

- **Total Runtime**: ~1 hour for 3,170 documents (2.2M words)
- **Bottlenecks**: 
  - N-gram generation with frequency tracking
  - Cohesion metric calculation for all n-grams
  - Semantic proximity computation (27,869 candidates × documents)

### Memory Requirements

- **N-grams Dictionary**: Large nested structure with metrics
- **Correlation Matrix**: 27,869 × documents pairs

### Optimization Opportunities

1. **Parallel Processing**: Distribute document processing
2. **Incremental Calculation**: Update metrics without full recomputation
3. **Sparse Matrix Representation**: Reduce memory for correlations
4. **Caching**: Store intermediate results
5. **Early Pruning**: Filter low-frequency n-grams earlier

## Key Findings

### Precision vs Recall Trade-off

- **High Precision** (SCP: 87.5%): Most identified REs are truly relevant
- **Low Recall** (~4%): Many human-identified REs are missed

**Explanation**: Difficult to match exact n-grams between human judgment and algorithm. Human evaluators use different granularity and phrasing.

### Dice Metric Superiority

- **Highest F-Metric** (0.0763): Best precision-recall balance
- **Most REs Found** (26,384): Comprehensive coverage
- **Good Precision** (82.5%): Reliable results

**Recommendation**: Use Dice coefficient for practical applications

### Challenges

1. **Low Recall**: Inherent difficulty matching human-identified expressions exactly
2. **Some Irrelevant REs**: Not all algorithm-identified expressions are meaningful in context
3. **Performance**: Long execution time for large corpora
4. **IP Not Implemented**: Semantic proximity approximated only by correlation

## Applications

This text mining approach is valuable for:

### Content Summarization
- Extract key themes without reading full documents
- Generate automatic document abstracts
- Identify main topics quickly

### Information Retrieval
- Improve search with semantic keywords
- Document categorization and tagging
- Topic modeling

### Academic Research
- Literature review automation
- Citation network analysis
- Trend identification in research domains

### Marketing & Media
- Brand mention analysis
- Sentiment trend detection
- Content strategy optimization

### Knowledge Management
- Automatic indexing of document repositories
- Related document discovery
- Expertise identification

## Future Improvements

### 1. Implement Intra-Document Proximity (IP)

Complete semantic proximity calculation:
```python
IP(A,B) = 1 - (1/||D*||) × Σ (dist(A,B,d) / farthest(A,B,d))
SemanticProx(A,B) = Corr(A,B) × √IP(A,B)
```

**Expected Impact**: More accurate implicit keyword identification

### 2. Performance Optimization

- Parallel processing with multiprocessing/threading
- Incremental metric updates
- Early filtering of low-frequency n-grams
- Optimize data structures (use numpy arrays, sparse matrices)

**Target**: Reduce runtime from 1 hour to <15 minutes

### 3. Enhanced Evaluation

- Larger human-labeled dataset (>200 samples)
- Multiple human evaluators for inter-rater agreement
- Domain-specific evaluation criteria
- Comparison with baseline methods (pure TF-IDF, word2vec)

### 4. Advanced NLP Techniques

- Word embeddings (Word2Vec, GloVe, BERT) for semantic similarity
- Named Entity Recognition (NER) for entity-focused keywords
- Dependency parsing for grammatical relevance
- Topic modeling (LDA, NMF) for thematic extraction

### 5. Interactive Visualization

- Web dashboard for exploring keywords per document
- Network graphs showing semantic relationships
- Timeline analysis for temporal keyword trends
- Comparison view across different metrics

## Academic Context

**Course**: Data Preprocessing and Text Mining (2023/2024)  
**Institution**: Universidade Nova de Lisboa - Faculty of Science and Technology  
**Authors**: Julia Cwynar, Andrea Cantelli, Micol Curci

This project demonstrates:
- Implementation of state-of-the-art text mining algorithms
- Comparative analysis of multiple cohesion metrics
- Rigorous evaluation methodology
- Practical application to large-scale corpus analysis

## References

### Academic Papers
- Silva, J. F., & Lopes, G. P. (1999). "A Local Maxima method and a Fair Dispersion Normalization for extracting multi-word units"
- Smadja, F. (1993). "Retrieving collocations from text: Xtract"
- Manning, C. D., & Schütze, H. (1999). "Foundations of Statistical Natural Language Processing"

### Metrics
- **SCP**: Symmetric Conditional Probability for collocation extraction
- **Dice Coefficient**: Sørensen–Dice coefficient for set similarity
- **φ²**: Phi coefficient for association strength
- **TF-IDF**: Term Frequency-Inverse Document Frequency weighting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Course instructors and teaching assistants
- Universidade Nova de Lisboa for computational resources
- Text corpus sources and contributors

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on the [GitHub repository](https://github.com/Cantellos/text-mining-keywords-extraction).

---

**Note**: This project was developed for educational purposes as part of a Data Preprocessing and Text Mining course. The implementation prioritizes clarity and understanding of text mining concepts over production-level optimization.
