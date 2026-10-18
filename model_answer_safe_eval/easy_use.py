"""One chat request with explicit endpoint/model and an environment API key."""
import argparse
import os
from llm_quick_use.http import ChatClient


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--endpoint", default=os.environ.get("LLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        if not args.endpoint or not args.model or not args.prompt.strip():
            raise ValueError("Provide a nonempty prompt, endpoint and model")
        client = ChatClient(args.endpoint, args.model, api_key=os.environ.get("LLM_API_KEY"),
                            timeout=args.timeout, max_attempts=args.max_attempts)
    except ValueError as exc:
        parser.error(str(exc))
    result = client.complete([{"role": "user", "content": args.prompt}])
    print(result["content"] if result["status"] == "ok" else result["error"])
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
