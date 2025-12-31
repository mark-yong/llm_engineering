"""
Optimized evaluation functions with async/parallel processing support.

Performance improvements:
1. Async LLM calls using litellm's async support
2. Parallel processing with controlled concurrency
3. Batch processing for vector searches where possible
4. Semaphore-based rate limiting to avoid API overload
"""
import sys
import os
import math
import asyncio
from typing import AsyncIterator
from pydantic import BaseModel, Field
from litellm import acompletion
from dotenv import load_dotenv

from evaluation.test import TestQuestion, load_tests, TEST_FILE
from implementation.answer import answer_question, fetch_context

# Try to import async versions if available
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
    HAS_ASYNC_LANGCHAIN = hasattr(ChatOpenAI, 'ainvoke')
except ImportError:
    HAS_ASYNC_LANGCHAIN = False

load_dotenv(override=True)

MODEL = "openrouter/openai/gpt-4.1-nano"
db_name = "vector_db"

# Concurrency limits to avoid overwhelming APIs
MAX_CONCURRENT_RETRIEVALS = 10  # Vector DB can handle multiple concurrent queries
MAX_CONCURRENT_LLM_CALLS = 5    # LLM API rate limits


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()

    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]

    # DCG
    dcg = calculate_dcg(relevances, k)

    # Ideal DCG (best case: keyword in first position)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


async def evaluate_retrieval_async(test: TestQuestion, k: int = 10, semaphore: asyncio.Semaphore = None) -> RetrievalEval:
    """
    Evaluate retrieval performance for a test question (async version).
    
    Args:
        test: TestQuestion object containing question and keywords
        k: Number of top documents to retrieve (default 10)
        semaphore: Optional semaphore for rate limiting
        
    Returns:
        RetrievalEval object with MRR, nDCG, and keyword coverage metrics
    """
    # Use semaphore to limit concurrent retrievals
    if semaphore:
        async with semaphore:
            # Run blocking retrieval in thread pool
            loop = asyncio.get_event_loop()
            retrieved_docs = await loop.run_in_executor(None, fetch_context, test.question)
    else:
        loop = asyncio.get_event_loop()
        retrieved_docs = await loop.run_in_executor(None, fetch_context, test.question)

    # Calculate metrics (CPU-bound, but fast)
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


async def evaluate_answer_async(test: TestQuestion, semaphore: asyncio.Semaphore = None) -> tuple[AnswerEval, str, list]:
    """
    Evaluate answer quality using LLM-as-a-judge (async version).
    
    Args:
        test: TestQuestion object containing question and reference answer
        semaphore: Optional semaphore for rate limiting
        
    Returns:
        Tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """
    # Get RAG response (run blocking calls in executor)
    loop = asyncio.get_event_loop()
    
    # Run answer_question in executor (it has blocking LLM and vector DB calls)
    if semaphore:
        async with semaphore:
            generated_answer, retrieved_docs = await loop.run_in_executor(
                None, answer_question, test.question
            )
    else:
        generated_answer, retrieved_docs = await loop.run_in_executor(
            None, answer_question, test.question
        )

    # LLM judge prompt
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]

    # Call LLM judge with structured outputs (async)
    if semaphore:
        async with semaphore:
            judge_response = await acompletion(
                model=MODEL,
                messages=judge_messages,
                response_format=AnswerEval,
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
    else:
        judge_response = await acompletion(
            model=MODEL,
            messages=judge_messages,
            response_format=AnswerEval,
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    # Extract content from response
    content = judge_response.choices[0].message.content  # type: ignore
    if content is None:
        raise ValueError("No content in judge response")
    answer_eval = AnswerEval.model_validate_json(content)

    return answer_eval, generated_answer, retrieved_docs


async def evaluate_all_retrieval_async() -> AsyncIterator[tuple[TestQuestion, RetrievalEval, float]]:
    """
    Evaluate all retrieval tests with parallel processing.
    
    Yields:
        Tuple of (test, result, progress) for each test
    """
    tests = load_tests(TEST_FILE)
    total_tests = len(tests)
    
    # Create semaphore to limit concurrent retrievals
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_RETRIEVALS)
    
    # Process tests in batches for better progress reporting
    batch_size = 10
    completed = 0
    
    for batch_start in range(0, total_tests, batch_size):
        batch = tests[batch_start:batch_start + batch_size]
        
        # Process batch in parallel
        tasks = [evaluate_retrieval_async(test, semaphore=semaphore) for test in batch]
        results = await asyncio.gather(*tasks)
        
        # Yield results
        for test, result in zip(batch, results):
            completed += 1
            progress = completed / total_tests
            yield test, result, progress


async def evaluate_all_answers_async() -> AsyncIterator[tuple[TestQuestion, AnswerEval, float]]:
    """
    Evaluate all answers to tests using parallel async execution.
    
    Yields:
        Tuple of (test, result, progress) for each test
    """
    tests = load_tests(TEST_FILE)
    total_tests = len(tests)
    
    # Create semaphore to limit concurrent LLM calls
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    
    # Process tests in batches for better progress reporting
    batch_size = 5  # Smaller batches for answer eval (more expensive)
    completed = 0
    
    for batch_start in range(0, total_tests, batch_size):
        batch = tests[batch_start:batch_start + batch_size]
        
        # Process batch in parallel
        tasks = [evaluate_answer_async(test, semaphore=semaphore) for test in batch]
        results = await asyncio.gather(*tasks)
        
        # Yield results
        for test, (result, _, _) in zip(batch, results):
            completed += 1
            progress = completed / total_tests
            yield test, result, progress


# Synchronous wrappers for backward compatibility
# These use a shared event loop to avoid creating new loops
_shared_loop = None

def _get_event_loop():
    """Get or create a shared event loop."""
    global _shared_loop
    if _shared_loop is None or _shared_loop.is_closed():
        _shared_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_shared_loop)
    return _shared_loop

def evaluate_all_retrieval():
    """Evaluate all retrieval tests (synchronous wrapper for async version)."""
    loop = _get_event_loop()
    
    async def _run():
        async for item in evaluate_all_retrieval_async():
            yield item
    
    gen = _run()
    while True:
        try:
            yield loop.run_until_complete(gen.__anext__())
        except StopAsyncIteration:
            break


def evaluate_all_answers():
    """Evaluate all answers to tests (synchronous wrapper for async version)."""
    loop = _get_event_loop()
    
    async def _run():
        async for item in evaluate_all_answers_async():
            yield item
    
    gen = _run()
    while True:
        try:
            yield loop.run_until_complete(gen.__anext__())
        except StopAsyncIteration:
            break

