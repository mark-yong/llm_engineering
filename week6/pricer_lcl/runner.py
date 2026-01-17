import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_URL = "http://192.168.68.104:8000"  # no /v1 here; requests include /v1/... in req["url"]
IN_PATH = "/root/llm_engineering/week6/jsonl/0_1000_lcl.jsonl"
OUT_PATH = "/root/llm_engineering/week6/jsonl/0_1000_lcl_results.jsonl"

CONCURRENCY = 24          # tune: 8-128 depending on GPU + model
TIMEOUT_S = 600           # per-request timeout
RETRIES = 2               # simple retry for transient errors


def normalize_req(obj: Dict[str, Any]) -> Dict[str, Any]:
    # minimal validation / normalization
    method = obj.get("method", "POST").upper()
    url = obj["url"]
    body = obj.get("body", {})
    custom_id = obj.get("custom_id")
    return {"custom_id": custom_id, "method": method, "url": url, "body": body}


async def send_one(
    client: httpx.AsyncClient,
    req: Dict[str, Any],
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    custom_id = req.get("custom_id", "unknown")
    async with sem:
        last_err: Optional[str] = None
        for attempt in range(RETRIES + 1):
            try:
                full_url = BASE_URL + req["url"]
                if attempt == 0:
                    logger.debug(f"Sending request custom_id={custom_id} to {full_url} (attempt {attempt + 1}/{RETRIES + 1})")
                
                r = await client.request(
                    req["method"],
                    full_url,
                    json=req["body"],
                    timeout=TIMEOUT_S,
                )

                # vLLM returns JSON for these endpoints
                content_type = r.headers.get("content-type", "")
                body = r.json() if "application/json" in content_type else r.text

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
                return out

            except httpx.TimeoutException as e:
                last_err = repr(e)
                logger.error(f"Request custom_id={custom_id} timeout on attempt {attempt + 1}/{RETRIES + 1}: {e}")
                if attempt < RETRIES:
                    # basic backoff
                    await asyncio.sleep(0.5 * (attempt + 1))
            except httpx.ConnectError as e:
                last_err = repr(e)
                logger.error(f"Request custom_id={custom_id} connection error on attempt {attempt + 1}/{RETRIES + 1}: {e}")
                if attempt < RETRIES:
                    # basic backoff
                    await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                last_err = repr(e)
                logger.exception(f"Request custom_id={custom_id} exception on attempt {attempt + 1}/{RETRIES + 1}")
                if attempt < RETRIES:
                    # basic backoff
                    await asyncio.sleep(0.5 * (attempt + 1))

        logger.error(f"Request custom_id={custom_id} failed after {RETRIES + 1} attempts. Last error: {last_err}")
        return {
            "custom_id": req.get("custom_id"),
            "response": None,
            "error": last_err or "unknown error",
        }


async def main():
    logger.info(f"Starting main() - BASE_URL: {BASE_URL}, IN_PATH: {IN_PATH}, OUT_PATH: {OUT_PATH}")
    logger.debug(f"CONCURRENCY: {CONCURRENCY}, TIMEOUT_S: {TIMEOUT_S}, RETRIES: {RETRIES}")
    
    sem = asyncio.Semaphore(CONCURRENCY)

    # Read JSONL requests
    reqs: List[Dict[str, Any]] = []
    logger.debug(f"Reading requests from {IN_PATH}")
    try:
        with open(IN_PATH, "r", encoding="utf-8") as f:
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
        logger.info(f"Successfully loaded {len(reqs)} requests from input file")
    except FileNotFoundError:
        logger.error(f"Input file not found: {IN_PATH}")
        return
    except Exception as e:
        logger.exception("Exception while reading input file")
        return

    if not reqs:
        logger.error("No valid requests loaded. Exiting.")
        return

    logger.info(f"Starting HTTP requests with {len(reqs)} requests...")
    try:
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(*(send_one(client, req, sem) for req in reqs))
        logger.info(f"All {len(results)} requests completed")
    except Exception as e:
        logger.exception("Exception during HTTP requests")
        return

    # Write JSONL results
    logger.debug(f"Writing results to {OUT_PATH}")
    try:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            written_count = 0
            for obj in results:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written_count += 1
        logger.info(f"Successfully wrote {written_count} results to {OUT_PATH}")
    except Exception as e:
        logger.exception("Exception while writing output file")
        return
    
    logger.info("main() completed successfully")


if __name__ == "__main__":
    logger.info("Script started")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.exception("Unhandled exception in main execution")
    logger.info("Script finished")
