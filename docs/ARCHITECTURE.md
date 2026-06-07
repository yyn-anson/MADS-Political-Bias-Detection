# System Architecture

This document describes the architecture of the Multi-Agent Political Bias Detection System.

---

## 🏗️ System Overview

The system employs a collaborative multi-agent framework where three independent LLMs analyze articles and engage in structured discussions to reach consensus on political bias detection.

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE COORDINATOR                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │  Phase 1:       │
        │  Individual     │
        │  Analysis       │
        └────┬────────────┘
             │
             ├──► Model 1 (Qwen3/Llama3.2) → Score, Direction, Reasoning
             ├──► Model 2 (GPT-OSS/Qwen3-4B) → Score, Direction, Reasoning
             └──► Model 3 (Mistral-Small) → Score, Direction, Reasoning
                  │
                  ▼
        ┌─────────────────┐
        │  Phase 2:       │
        │  Consensus      │
        │  Check          │
        └────┬────────────┘
             │
             ├──► All agree? → Final Result
             ├──► 2/3 agree? → Majority Result
             └──► All differ? → Phase 3
                       │
                       ▼
        ┌──────────────────────────┐
        │  Phase 3:                │
        │  Two-Stage Discussion    │
        └──────┬───────────────────┘
               │
               ├─► Stage 1: All-model debate
               │   Until majority consensus
               │
               └─► Stage 2: Winner vs Minority
                   Final convergence
```

---

## 🔄 Processing Workflow

### 1. Batch Processing

Articles are processed in memory-efficient batches:

```python
# Regular ensemble: 3 articles/batch (24GB VRAM)
# Small ensemble: 8 articles/batch (12GB VRAM)

for batch in article_batches:
    # Load model
    model = load_model(model_name)

    # Process batch
    results = model.label_articles_batch(batch)

    # Unload model (free memory)
    unload_model(model)
```

**Benefits**:
- Sequential model loading prevents OOM errors
- GPU memory freed between models
- Resumable if interrupted

### 2. Individual Analysis

Each model independently analyzes articles:

```
Input Article → Model Pipeline → Output
                     │
                     ├─► Tokenization
                     ├─► Context injection
                     ├─► LLM inference
                     └─► JSON extraction
                              │
                              ▼
                        {
                          "lean": -2.0,
                          "reason": "Analysis...",
                          "direction": "Left"
                        }
```

**Key Features**:
- Thinking mode (Qwen3): Chain-of-thought reasoning
- Structured prompts: Consistent output format
- Robust parsing: Multiple JSON extraction strategies

### 3. Consensus Analysis

Three consensus scenarios:

```
Scenario 1: Unanimous (All 3 agree)
  Left + Left + Left → CONSENSUS: Left (average score)

Scenario 2: Majority (2/3 agree)
  Left + Left + Right → CONSENSUS: Left (average of matching)

Scenario 3: Complete Disagreement (All differ)
  Left + Center + Right → DISCUSSION TRIGGERED
```

---

## 💬 Two-Stage Discussion Framework

### Stage 1: All-Model Debate

**Goal**: Reach majority consensus (2/3 agree on direction)

**Process**:
1. Select discussion pair (largest score difference)
2. Challenger presents argument to target
3. Target responds, may adjust position
4. Repeat until majority or max rounds (8)

```
Round 1:
  Qwen (Left) challenges GPT-OSS (Right)
  ├─► Challenge: "I argue Left because..."
  ├─► Response: "I maintain Right because..."
  └─► Position updates checked

Round 2:
  Mistral (Center) challenges Qwen (Left)
  ...

Continue until 2/3 agree or max rounds
```

**Termination Conditions**:
- ✅ Full consensus (all 3 agree)
- ✅ Majority reached (2/3 agree)
- ⏱️ Max rounds (8)

### Stage 2: Winner vs. Minority

**Goal**: Final resolution between majority representative and minority

**Process**:
1. Identify Stage 1 winner direction
2. Select representative (initially held that direction)
3. Select minority model
4. Single round of debate
5. Winner determined by convergence

```
Stage 1 Result: 2 models say Left, 1 says Right

Stage 2:
  Representative (Qwen, Left) vs Minority (GPT-OSS, Right)
  ├─► Challenge: Representative argues for Left
  ├─► Response: Minority responds
  └─► Winner determination:
       ├─► Did minority converge to Left? → Representative wins
       ├─► Did representative converge to Right? → Minority wins
       ├─► Both maintain positions? → Use conviction (|score|)
       └─► Final winner's position adopted by ALL models
```

**Winner Determination Logic**:
1. **Convergence**: Did one side move to the other's initial direction?
2. **Conviction**: If no convergence, highest |score| wins
3. **Tiebreaker**: Stage 1 winner takes precedence

---

## 📊 Data Flow

### Input → Processing → Output

```
Input: Article JSON
{
  "source_name": "cnn.com",
  "content": "Article text...",
  "bias": -1.5  // Optional ground truth
}
         │
         ▼
   ┌──────────────┐
   │   Qwen3      │ → {"lean": -2.0, "reason": "...", "direction": "Left"}
   │   GPT-OSS    │ → {"lean": 1.5, "reason": "...", "direction": "Right"}
   │   Mistral    │ → {"lean": 0.0, "reason": "...", "direction": "Center"}
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Consensus   │ → All differ → Discussion needed
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Discussion  │ → Stage 1 + Stage 2
   └──────┬───────┘
          │
          ▼
Output: Final Result
{
  "article_id": 5,
  "individual_scores": {...},
  "final_score": -1.8,
  "final_direction": "Left",
  "discussion_method": "two_stage",
  "convergence_achieved": true
}
```

---

## 🧩 Component Architecture

### Core Components

```
src/
├── ensemble/
│   ├── ensemble_regular.py          # Main ensemble coordinator
│   │   ├── Class: EnsembleMultiModelDetector
│   │   ├── Methods:
│   │   │   ├── process_articles()           # Main entry point
│   │   │   ├── _run_individual_analysis()   # Phase 1
│   │   │   ├── _check_consensus()           # Phase 2
│   │   │   ├── _run_collaborative_discussion()  # Phase 3
│   │   │   ├── _run_stage2_debate()
│   │   │   └── _generate_challenge/response()
│   │   └── Features:
│   │       ├── Sequential model loading
│   │       ├── Memory management
│   │       └── Discussion orchestration
│   │
│   └── ensemble_small.py             # Small model variant
│       └── (Same structure, different models)
│
├── models/
│   ├── qwen3_labeler.py              # Qwen3-14B wrapper
│   ├── gptoss_labeler.py             # GPT-OSS-20B wrapper
│   ├── mistral_labeler.py            # Mistral-Small-22B wrapper
│   └── llama32_labeler.py            # Llama3.2-3B wrapper
│       └── Class: PoliticalLeaningLabeler
│           ├── label_articles_batch()        # Batch inference
│           ├── generate_discussion_challenge()  # For debate
│           └── generate_discussion_response()
│
├── utils/
│   └── json_extractor.py             # Robust JSON parsing
│       └── Class: RobustJSONExtractor
│           ├── extract_json()                # Multiple strategies
│           ├── extract_challenge_fields()
│           └── extract_response_fields()
│
└── evaluation/
    └── outlet_evaluation.py          # Outlet-level metrics
        ├── compute_outlet_statistics()
        ├── generate_visualizations()
        └── create_evaluation_report()
```

---

## 🎯 Design Principles

### 1. Memory Efficiency
- Sequential model loading (not parallel)
- Explicit GPU memory cleanup
- Batch processing with size limits

### 2. Robustness
- Multiple JSON extraction fallbacks
- Error handling and recovery
- Batch resumption on failure

### 3. Transparency
- Complete I/O logging (prompts + responses)
- Discussion transcript storage
- Detailed metrics tracking

### 4. Modularity
- Swappable model components
- Independent ensemble configurations
- Reusable evaluation tools

---

## 📈 Performance Optimizations

### GPU Memory Management

```python
# Load → Process → Unload cycle
def _process_single_model(self, model_name, articles, labeler):
    # Process with model
    results = labeler.label_articles_batch(articles)

    # Free GPU memory
    del labeler.model, labeler.tokenizer
    torch.cuda.empty_cache()

    return results
```

### Discussion Timeout

```python
# Prevent infinite discussions
discussion_result = await asyncio.wait_for(
    self._run_collaborative_discussion(...),
    timeout=1800  # 30 minutes
)
```

### Batch Size Auto-Tuning

```python
# Adapt to available memory
if torch.cuda.get_device_properties(0).total_memory > 20e9:
    batch_size = 8  # High-memory GPU
else:
    batch_size = 3  # Standard GPU
```

---

## 🔐 Error Handling

### Three-Level Error Strategy

**Level 1: Graceful Degradation**
```python
try:
    discussion_result = await run_discussion(article)
except asyncio.TimeoutError:
    # Skip article, continue to next
    logger.error("Discussion timeout - SKIPPING")
    continue
```

**Level 2: Fallback Values**
```python
try:
    score = extract_score(response)
except ValueError:
    # Use default or previous value
    score = agent.current_score
```

**Level 3: Critical Failures**
```python
if not all_models_loaded:
    raise RuntimeError("Critical: Cannot proceed without models")
```

---

## 📝 Logging & Monitoring

### Multi-Level Logging

```
INFO: High-level progress
  ├─► "Processing batch 1-3"
  ├─► "Consensus reached for article 5"
  └─► "Discussion triggered for article 7"

DEBUG: Detailed operations
  ├─► "Loading Qwen3 model..."
  ├─► "Extracted JSON: {..."
  └─► "Agent updated score: -2.0 → -1.5"

ERROR: Issues and failures
  ├─► "JSON extraction failed, using fallback"
  ├─► "Discussion timeout after 30 minutes"
  └─► "Article skipped due to critical error"
```

### Statistics Tracking

```python
self.stats = {
    'total_articles': 0,
    'consensus_unanimous': 0,
    'consensus_majority': 0,
    'discussion_triggered': 0,
    'discussion_converged': 0,
    'articles_skipped': 0
}
```

---

## 🚀 Scalability Considerations

### Horizontal Scaling

```bash
# Process different datasets in parallel
python run_batches.py --dataset baly &
python run_batches.py --dataset budak &
python run_batches.py --dataset ad_fontes &
```

### Vertical Scaling

```python
# Adjust batch size based on GPU
if gpu_memory > 40GB:
    batch_size = 12
elif gpu_memory > 24GB:
    batch_size = 8
else:
    batch_size = 3
```

---

## 📚 References

- **Ensemble Learning**: Zhou, Z. H. (2012). "Ensemble methods: foundations and algorithms."
- **Multi-Agent Debate**: Du, Y. et al. (2023). "Improving Factuality and Reasoning in Language Models through Multiagent Debate."
- **LLM Calibration**: Kadavath, S. et al. (2022). "Language Models (Mostly) Know What They Know."

---

For implementation details, see the source code documentation in `src/`.
