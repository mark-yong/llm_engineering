# Performance Analysis and Optimization Recommendations

## Current Performance Bottlenecks

### 1. **Sequential Processing (CRITICAL)**
- **Issue**: All 150 tests are processed one-by-one in a loop
- **Impact**: If each test takes 2-3 seconds, total time = 150 × 2.5s = **6.25 minutes**
- **Location**: `evaluate_all_retrieval()` and `evaluate_all_answers()` in `evaluation/eval.py`

### 2. **Blocking LLM API Calls (CRITICAL)**
- **Issue**: Each LLM call blocks until completion
- **Impact**: For answer evaluation, each test makes 2 LLM calls:
  - Answer generation (~1-2s)
  - Judge evaluation (~1-2s)
  - Total per test: ~2-4s × 150 = **5-10 minutes**
- **Location**: `answer_question()` and `evaluate_answer()` in `evaluation/eval.py`

### 3. **Blocking Vector Database Queries (HIGH)**
- **Issue**: Vector similarity search is synchronous
- **Impact**: Each retrieval takes ~100-500ms, processed sequentially
- **Location**: `fetch_context()` in `implementation/answer.py`

### 4. **No Concurrency Control (MEDIUM)**
- **Issue**: No rate limiting or batching
- **Impact**: Risk of API rate limit errors, inefficient resource usage
- **Location**: All evaluation functions

### 5. **Inefficient Event Loop Usage (LOW)**
- **Issue**: Comments mention "async" but no actual async code
- **Impact**: Missing opportunity for parallelization
- **Location**: `evaluate_all_answers()` comment says "batched async execution" but uses sync code

## Optimization Strategies

### Strategy 1: Parallel Processing with Async/Await (RECOMMENDED)
**Expected Speedup**: 5-10x for retrieval, 3-5x for answer evaluation

**Implementation**:
- Use `asyncio.gather()` to process multiple tests concurrently
- Use semaphores to limit concurrency (avoid API rate limits)
- Run blocking I/O operations in thread pool executors

**Code Changes**:
- Convert `evaluate_all_retrieval()` and `evaluate_all_answers()` to async
- Use `acompletion()` from litellm for async LLM calls
- Use `loop.run_in_executor()` for blocking vector DB calls

### Strategy 2: Batch Processing
**Expected Speedup**: 2-3x

**Implementation**:
- Process tests in batches (e.g., 10 at a time)
- Better progress reporting
- More efficient resource usage

### Strategy 3: Optimize Vector Search
**Expected Speedup**: 1.5-2x for retrieval

**Implementation**:
- Cache embeddings if possible
- Use async vector DB queries if supported
- Batch multiple queries if the vector DB supports it

### Strategy 4: LLM Call Optimization
**Expected Speedup**: 1.5-2x

**Implementation**:
- Use async LLM calls (`acompletion` instead of `completion`)
- Parallelize answer generation and judge evaluation where possible
- Consider using faster/cheaper models for judge evaluation

## Recommended Implementation

The optimized version (`evaluation/eval_optimized.py`) implements:

1. **Async/await pattern** for all I/O operations
2. **Semaphore-based rate limiting** (10 concurrent retrievals, 5 concurrent LLM calls)
3. **Batch processing** (10 tests per batch for retrieval, 5 for answers)
4. **Thread pool executors** for blocking operations

### Expected Performance Improvements

**Retrieval Evaluation**:
- Current: ~6-8 minutes (150 tests × 2.5s average)
- Optimized: ~1-2 minutes (10x parallelization)
- **Speedup: 4-6x**

**Answer Evaluation**:
- Current: ~10-15 minutes (150 tests × 4-6s average)
- Optimized: ~3-5 minutes (3-5x parallelization)
- **Speedup: 3-5x**

## Usage

To use the optimized version, update `evaluator.py`:

```python
# Change from:
from evaluation.eval import evaluate_all_retrieval, evaluate_all_answers

# To:
from evaluation.eval_optimized import evaluate_all_retrieval, evaluate_all_answers
```

The API remains the same, so no other changes are needed.

## Additional Recommendations

1. **Monitor API Rate Limits**: Adjust `MAX_CONCURRENT_LLM_CALLS` based on your API provider's limits
2. **GPU Utilization**: If using local models, ensure GPU is being utilized (check with `nvidia-smi`)
3. **Vector DB Optimization**: Consider using a vector DB that supports async queries natively
4. **Caching**: Cache retrieval results for repeated evaluations
5. **Progress Reporting**: The batch-based approach provides better progress updates

## Testing

To verify improvements:
1. Run original version and time it
2. Run optimized version and compare
3. Monitor CPU/GPU/network usage during both runs

