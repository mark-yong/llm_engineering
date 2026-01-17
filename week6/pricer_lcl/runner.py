import asyncio
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Use notebook tqdm if available (for Jupyter), otherwise regular tqdm
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_URL = "http://192.168.68.104:8000"  # no /v1 here; requests include /v1/... in req["url"]

CONCURRENCY = 256         # tune: 8-128 depending on GPU + model
TIMEOUT_S = 600           # per-request timeout
RETRIES = 2               # simple retry for transient errors

# These constants match batch.py for compatibility
MODEL = "openai/gpt-oss-20b"
BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"
SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


def create_request_from_item(item) -> Dict[str, Any]:
    """Create a request dict from an item, formatted for vLLM (not Groq batch format)."""
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.full},
        ],
        "include_reasoning": False,  # vLLM parameter (not reasoning_effort which is Groq-specific)
    }
    return {
        "custom_id": str(item.id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def normalize_req(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize request for vLLM format.
    
    Converts Groq batch format (reasoning_effort) to vLLM format (include_reasoning).
    vLLM expects: {"custom_id": ..., "method": "POST", "url": "/v1/chat/completions", "body": {...}}
    where body contains vLLM-compatible parameters (include_reasoning, not reasoning_effort).
    """
    method = obj.get("method", "POST").upper()
    url = obj.get("url", "/v1/chat/completions")
    body = obj.get("body", {})
    custom_id = obj.get("custom_id")
    
    # Ensure body has required fields
    if "model" not in body:
        body["model"] = MODEL
    if "messages" not in body:
        body["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ""}
        ]
    
    # Convert Groq's reasoning_effort to vLLM's include_reasoning
    # reasoning_effort is Groq-specific, include_reasoning is vLLM-specific
    if "reasoning_effort" in body and "include_reasoning" not in body:
        # Convert: reasoning_effort="low" -> include_reasoning=False (skip CoT for faster output)
        # reasoning_effort="high" -> include_reasoning=True (include CoT)
        reasoning_effort = body.pop("reasoning_effort")
        body["include_reasoning"] = (reasoning_effort in ["high", "medium"])
    elif "include_reasoning" not in body:
        # Default to False (no reasoning) for faster output
        body["include_reasoning"] = False
    
    return {"custom_id": custom_id, "method": method, "url": url, "body": body}


def create_httpx_client() -> httpx.AsyncClient:
    """Create httpx client with optimized connection limits and keepalive.
    
    Note: Explicitly excludes Authorization header as vLLM doesn't require API keys
    and may reject requests with invalid/empty Authorization headers.
    
    Connection pool is capped at a reasonable maximum to avoid overwhelming the server.
    With high concurrency, connections are reused efficiently.
    """
    # Cap connection pool to avoid overwhelming the server
    # Too many connections can cause ReadError/connection resets
    max_conns = min(CONCURRENCY * 2, 200)  # Cap at 200 connections max
    limits = httpx.Limits(
        max_connections=max_conns,
        max_keepalive_connections=max_conns
    )
    timeout = httpx.Timeout(TIMEOUT_S)
    # Explicitly exclude Authorization header for vLLM (doesn't require API keys)
    # httpx may pick up OPENAI_API_KEY from environment, so we override it
    return httpx.AsyncClient(
        limits=limits, 
        timeout=timeout,
        # Don't set default headers - we'll set them per-request to avoid Authorization
    )


async def send_one(
    client: httpx.AsyncClient,
    req: Dict[str, Any],
    sem: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    custom_id = req.get("custom_id", "unknown")
    request_start_time = time.time()
    # Use semaphore if provided (for layered concurrency control)
    # Otherwise rely on connection limits
    if sem:
        async with sem:
            return await _send_one_core(client, req, request_start_time, custom_id)
    else:
        return await _send_one_core(client, req, request_start_time, custom_id)


async def _send_one_core(
    client: httpx.AsyncClient,
    req: Dict[str, Any],
    request_start_time: float,
    custom_id: str,
) -> Dict[str, Any]:
    last_err: Optional[str] = None
    for attempt in range(RETRIES + 1):
        try:
            full_url = BASE_URL + req["url"]
            if attempt == 0:
                logger.debug(f"Sending request custom_id={custom_id} to {full_url} (attempt {attempt + 1}/{RETRIES + 1})")
            
            # Explicitly exclude Authorization header for vLLM (doesn't require API keys)
            # Set headers explicitly to avoid any default Authorization header
            headers = {"Content-Type": "application/json"}
            # Remove Authorization if present (httpx might add it from env vars)
            r = await client.request(
                req["method"],
                full_url,
                json=req["body"],
                headers=headers,
            )

            # vLLM returns JSON for these endpoints
            content_type = r.headers.get("content-type", "")
            body = r.json() if "application/json" in content_type else r.text

            # Output format matches batch.py's apply_output() expectations:
            # json_line["response"]["body"]["choices"][0]["message"]["content"]
            out = {
                "custom_id": req.get("custom_id"),
                "response": {"status_code": r.status_code, "body": body},
                "error": None,
            }
            if r.status_code >= 400:
                out["error"] = body
                logger.warning(f"Request custom_id={custom_id} returned status {r.status_code}: {str(body)[:200]}")
            else:
                if attempt > 0:
                    logger.debug(f"Request custom_id={custom_id} succeeded on attempt {attempt + 1}")
                # Verify output structure matches batch.py expectations
                if isinstance(body, dict) and "choices" in body:
                    if len(body.get("choices", [])) > 0:
                            if "message" not in body["choices"][0] or "content" not in body["choices"][0].get("message", {}):
                                logger.warning(f"Request custom_id={custom_id} response missing expected structure")
            
            return out

        except httpx.TimeoutException as e:
            last_err = repr(e)
            logger.error(f"Request custom_id={custom_id} timeout on attempt {attempt + 1}/{RETRIES + 1}: {e}")
            if attempt < RETRIES:
                # exponential backoff
                await asyncio.sleep(0.5 * (2 ** attempt))
        except httpx.ConnectError as e:
            last_err = repr(e)
            logger.error(f"Request custom_id={custom_id} connection error on attempt {attempt + 1}/{RETRIES + 1}: {e}")
            if attempt < RETRIES:
                # exponential backoff
                await asyncio.sleep(0.5 * (2 ** attempt))
        except httpx.ReadError as e:
            last_err = repr(e)
            logger.error(f"Request custom_id={custom_id} read error on attempt {attempt + 1}/{RETRIES + 1}: {e}")
            if attempt < RETRIES:
                # exponential backoff - ReadError often indicates connection reset, give it more time
                await asyncio.sleep(1.0 * (2 ** attempt))
        except httpx.WriteError as e:
            last_err = repr(e)
            logger.error(f"Request custom_id={custom_id} write error on attempt {attempt + 1}/{RETRIES + 1}: {e}")
            if attempt < RETRIES:
                # exponential backoff
                await asyncio.sleep(0.5 * (2 ** attempt))
        except Exception as e:
            last_err = repr(e)
            logger.exception(f"Request custom_id={custom_id} exception on attempt {attempt + 1}/{RETRIES + 1}")
            if attempt < RETRIES:
                # exponential backoff
                await asyncio.sleep(0.5 * (2 ** attempt))

    logger.error(f"Request custom_id={custom_id} failed after {RETRIES + 1} attempts. Last error: {last_err}")
    return {
        "custom_id": req.get("custom_id"),
        "response": None,
        "error": last_err or "unknown error",
    }


class Batch:
    BATCH_SIZE = 1_000

    batches = []

    def __init__(self, items, start, end, lite):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.done = False
        folder = Path("lite") if lite else Path("full")
        self.batch_dir = folder / BATCHES_FOLDER
        self.output_dir = folder / OUTPUT_FOLDER
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Store results in memory instead of files
        self.results: List[Dict[str, Any]] = []

    def make_jsonl(self, item):
        """Create JSONL line from item, formatted for vLLM (not Groq batch format)."""
        req = create_request_from_item(item)
        return json.dumps(req)

    def make_file(self):
        """Create JSONL file (optional, for compatibility with batch.py interface)"""
        batch_file = self.batch_dir / self.filename
        with batch_file.open("w") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    async def run_async(self, client: httpx.AsyncClient, sem: Optional[asyncio.Semaphore] = None, pbar: Optional[tqdm] = None):
        """Process this batch using async HTTP client.
        
        Note: semaphore is optional. Connection limits on the client handle concurrency,
        but semaphore can provide additional layered control if needed.
        """
        # Create requests for all items in this batch
        reqs = [create_request_from_item(item) for item in self.items[self.start : self.end]]

        # Process all requests in this batch
        # For batches, creating all tasks is fine since batch size is limited (BATCH_SIZE = 1000)
        tasks = [send_one(client, req, sem) for req in reqs]
        results = await asyncio.gather(*tasks)
        
        # Update progress bar every 50 requests
        if pbar:
            batch_size = len(reqs)
            # Update in chunks of 50
            for i in range(0, batch_size, 50):
                chunk_size = min(50, batch_size - i)
                pbar.update(chunk_size)
                pbar.refresh()
        
        self.results = results
        self.done = True

    def write_output(self):
        """Write results to output file, matching batch.py's fetch_output() format."""
        output_file = self.output_dir / self.filename
        with output_file.open("w", encoding="utf-8") as f:
            for result in self.results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def apply_output(self):
        """Apply results back to items, matching batch.py's apply_output() format."""
        applied_count = 0
        error_count = 0
        
        for result in self.results:
            if result.get("error") is not None:
                error_count += 1
                continue
                
            response = result.get("response", {})
            if response.get("status_code", 0) >= 400:
                error_count += 1
                continue
                
            body = response.get("body", {})
            if not isinstance(body, dict) or "choices" not in body:
                error_count += 1
                continue
                
            choices = body.get("choices", [])
            if not choices:
                error_count += 1
                continue
                
            message = choices[0].get("message", {})
            content = message.get("content")
            if not content:
                error_count += 1
                continue
                
            custom_id = result.get("custom_id")
            if custom_id is None:
                error_count += 1
                continue
                
            try:
                item_id = int(custom_id)
                if 0 <= item_id < len(self.items):
                    self.items[item_id].summary = content
                    applied_count += 1
                else:
                    logger.warning(f"Item ID {item_id} out of range (max: {len(self.items) - 1})")
                    error_count += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid custom_id '{custom_id}': {e}")
                error_count += 1
        
        if error_count > 0:
            logger.warning(f"Applied {applied_count} results, {error_count} errors in batch {self.filename}")

    @classmethod
    def create(cls, items, lite):
        """Create batches from items, matching batch.py interface"""
        cls.batches = []
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = Batch(items, start, end, lite)
            cls.batches.append(batch)
        print(f"Created {len(cls.batches)} batches")

    @classmethod
    async def _run_async(cls):
        """Internal async method to process all batches."""
        if not cls.batches:
            print("No batches to process")
            return
            
        # Semaphore provides additional concurrency control beyond connection limits
        # If batches are run concurrently in the future, use a single shared semaphore
        sem = asyncio.Semaphore(CONCURRENCY)
        total_items = sum(len(batch.items[batch.start:batch.end]) for batch in cls.batches)
        
        async with create_httpx_client() as client:
            with tqdm(total=total_items, desc="Processing batches", unit="req") as pbar:
                for batch in cls.batches:
                    await batch.run_async(client, sem, pbar)
        print(f"Processed {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        """Process all batches using async HTTP client.
        
        Works in both notebooks and regular Python. In notebooks, automatically
        runs the async code using nest_asyncio if available.
        In regular Python or if nest_asyncio is not available, returns a coroutine
        that should be awaited: `await Batch.run()` or use `asyncio.run(Batch.run())`.
        """
        try:
            # Check if we're in IPython/Jupyter
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython and hasattr(ipython, 'kernel'):
                # We're in a notebook - try to use nest_asyncio for seamless execution
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return asyncio.run(cls._run_async())
                except ImportError:
                    # nest_asyncio not available - return coroutine for user to await
                    logger.warning("nest_asyncio not available. Use 'await Batch.run()' in notebook")
                    return cls._run_async()
            else:
                # Not in notebook - return coroutine for user to await
                return cls._run_async()
        except (NameError, ImportError):
            # get_ipython() not available - regular Python
            # Return coroutine - user should use await or asyncio.run()
            return cls._run_async()

    @classmethod
    def fetch(cls):
        """Apply outputs to all batches (synchronous, matching batch.py interface).
        
        Writes results to output files and applies them to items.
        """
        for batch in tqdm(cls.batches):
            if not batch.done:
                continue
            # Write results to file (matching batch.py's fetch_output behavior)
            batch.write_output()
            # Apply results to items
            batch.apply_output()
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls, state_path: Path = Path("batches.pkl")):
        """Save batches to disk (matching batch.py interface)."""
        if not cls.batches:
            print("No batches to save")
            return
        items = cls.batches[0].items
        # Temporarily remove items to reduce pickle size, but preserve results
        # in case user wants to resume without re-processing
        for batch in cls.batches:
            batch.items = None
        try:
            with state_path.open("wb") as f:
                pickle.dump(cls.batches, f)
            # Restore items
            for batch in cls.batches:
                batch.items = items
            print(f"Saved {len(cls.batches)} batches to {state_path}")
        except Exception as e:
            # Restore items even if save fails
            for batch in cls.batches:
                batch.items = items
            logger.error(f"Failed to save batches: {e}")
            raise

    @classmethod
    def load(cls, items, state_path: Path = Path("batches.pkl")):
        """Load batches from disk (matching batch.py interface)."""
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")
        try:
            with state_path.open("rb") as f:
                cls.batches = pickle.load(f)
            for batch in cls.batches:
                batch.items = items
                # Initialize results list if not present
                if not hasattr(batch, "results"):
                    batch.results = []
            print(f"Loaded {len(cls.batches)} batches from {state_path}")
        except Exception as e:
            logger.error(f"Failed to load batches: {e}")
            raise

    @classmethod
    async def run_and_fetch(cls):
        """Convenience method to run and fetch in one call"""
        await cls._run_async()
        cls.fetch()
    
    @classmethod
    def run_and_fetch_sync(cls):
        """Convenience method to run and fetch in one call (synchronous wrapper)"""
        cls.run()
        cls.fetch()


async def main(in_path: str, out_path: str):
    script_start_time = time.time()
    logger.info(f"Starting main() - BASE_URL: {BASE_URL}, in_path: {in_path}, out_path: {out_path}")
    logger.debug(f"CONCURRENCY: {CONCURRENCY}, TIMEOUT_S: {TIMEOUT_S}, RETRIES: {RETRIES}")
    
    # Semaphore not needed in main() - worker queue pattern + connection limits handle concurrency

    # Read JSONL requests
    read_start_time = time.time()
    reqs: List[Dict[str, Any]] = []
    read_time = 0.0
    logger.debug(f"Reading requests from {in_path}")
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    reqs.append(normalize_req(json.loads(line)))
                    line_count += 1
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON on line {line_count}: {e}")
                    logger.error(f"Line content: {line[:200]}")
        read_time = time.time() - read_start_time
        logger.info(f"Successfully loaded {len(reqs)} requests from input file (took {read_time:.2f}s)")
    except FileNotFoundError:
        logger.error(f"Input file not found: {in_path}")
        return
    except Exception as e:
        logger.exception("Exception while reading input file")
        return

    if not reqs:
        logger.error("No valid requests loaded. Exiting.")
        return

    # Execute HTTP requests with progress tracking using worker queue pattern
    # This scales much better for large inputs (100k+) by avoiding creating all tasks at once
    logger.info(f"Starting HTTP requests with {len(reqs)} requests...")
    request_start_time = time.time()
    results: List[Dict[str, Any]] = []
    success_count = 0
    error_count = 0
    request_time = 0.0
    
    try:
        async with create_httpx_client() as client:
            # Create progress bar
            with tqdm(total=len(reqs), desc="Processing requests", unit="req") as pbar:
                # Worker queue pattern: avoids creating all tasks upfront
                queue: asyncio.Queue = asyncio.Queue()
                results_dict: Dict[str, Dict[str, Any]] = {}
                
                # Put all requests in queue
                for req in reqs:
                    await queue.put(req)
                
                # Add sentinel values to signal workers to stop
                for _ in range(CONCURRENCY):
                    await queue.put(None)
                
                # Counter for progress bar updates (update every 50 requests)
                update_counter = 0
                update_lock = asyncio.Lock()
                
                async def worker(worker_id: int):
                    """Worker coroutine that processes requests from queue."""
                    nonlocal success_count, error_count, update_counter
                    while True:
                        req = await queue.get()
                        if req is None:  # Sentinel value
                            queue.task_done()
                            break
                        
                        # No semaphore needed - connection limits handle concurrency
                        result = await send_one(client, req, sem=None)
                        
                        custom_id = result.get("custom_id")
                        if custom_id:
                            results_dict[custom_id] = result
                        
                        if result.get("error") is None and result.get("response", {}).get("status_code", 0) < 400:
                            success_count += 1
                        else:
                            error_count += 1
                        
                        # Update progress bar every 50 requests
                        should_update = False
                        update_amount = 0
                        async with update_lock:
                            update_counter += 1
                            current_count = update_counter
                            # Update every 50 requests
                            if current_count % 50 == 0:
                                should_update = True
                                update_amount = 50
                        
                        if should_update:
                            pbar.update(update_amount)
                            elapsed = time.time() - request_start_time
                            if elapsed > 0:
                                rate = success_count / elapsed
                                pbar.set_postfix({
                                    "success": success_count,
                                    "errors": error_count,
                                    "rate": f"{rate:.1f}/s"
                                })
                            else:
                                pbar.set_postfix({
                                    "success": success_count,
                                    "errors": error_count,
                                })
                        
                        queue.task_done()
                
                # Start worker tasks
                workers = [asyncio.create_task(worker(i)) for i in range(CONCURRENCY)]
                
                # Wait for all work to complete
                await queue.join()
                
                # Wait for all workers to finish (they exit on sentinel)
                await asyncio.gather(*workers, return_exceptions=True)
                
                # Final progress bar update to ensure 100% completion
                remaining = len(reqs) - pbar.n
                if remaining > 0:
                    pbar.update(remaining)
                    elapsed = time.time() - request_start_time
                    if elapsed > 0:
                        rate = success_count / elapsed
                        pbar.set_postfix({
                            "success": success_count,
                            "errors": error_count,
                            "rate": f"{rate:.1f}/s"
                        })
                
                # Reconstruct results in original order
                results = [results_dict.get(req.get("custom_id"), {}) for req in reqs]
        
        request_time = time.time() - request_start_time
        if request_time > 0:
            rate = len(reqs) / request_time
            logger.info(f"All {len(results)} requests completed in {request_time:.2f}s ({rate:.2f} req/s)")
        else:
            logger.info(f"All {len(results)} requests completed in {request_time:.2f}s")
        logger.info(f"Success: {success_count}, Errors: {error_count}")
    except Exception as e:
        logger.exception("Exception during HTTP requests")
        return

    # Write JSONL results
    write_start_time = time.time()
    write_time = 0.0
    logger.debug(f"Writing results to {out_path}")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            written_count = 0
            for obj in results:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written_count += 1
        write_time = time.time() - write_start_time
        logger.info(f"Successfully wrote {written_count} results to {out_path} (took {write_time:.2f}s)")
    except Exception as e:
        logger.exception("Exception while writing output file")
        return
    
    total_time = time.time() - script_start_time
    logger.info("main() completed successfully")
    logger.info(f"Total execution time: {total_time:.2f}s (read: {read_time:.2f}s, requests: {request_time:.2f}s, write: {write_time:.2f}s)")


if __name__ == "__main__":
    import sys
    script_start = time.time()
    logger.info("Script started")
    try:
        if len(sys.argv) < 3:
            logger.error("Usage: python runner.py <input_path> <output_path>")
            sys.exit(1)
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        asyncio.run(main(in_path, out_path))
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.exception("Unhandled exception in main execution")
    total_script_time = time.time() - script_start
    logger.info(f"Script finished (total time: {total_script_time:.2f}s)")
