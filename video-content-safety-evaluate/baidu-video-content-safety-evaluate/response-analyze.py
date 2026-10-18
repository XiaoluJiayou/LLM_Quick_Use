"""Inspect HTTP metadata while masking request credentials and URL query values."""
import argparse
import json
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

SENSITIVE = {"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key", "api-key"}


def sanitized_url(value):
    parts = urlsplit(value)
    host = parts.hostname or ""
    if parts.port:
        host += ":" + str(parts.port)
    query = urlencode([(key, "[REDACTED]") for key, _ in parse_qsl(parts.query, keep_blank_values=True)])
    return urlunsplit((parts.scheme, host, parts.path, query, ""))


def analyze_response(response):
    summary = {"status_code": response.status_code, "ok": response.ok,
               "url": sanitized_url(response.url), "content_length": len(response.content),
               "redirect_count": len(response.history),
               "response_headers": {k: "[REDACTED]" if k.lower() in SENSITIVE else v
                                    for k, v in response.headers.items()}}
    request = getattr(response, "request", None)
    if request:
        summary["request_method"] = request.method
        summary["request_headers"] = {k: "[REDACTED]" if k.lower() in SENSITIVE else v
                                      for k, v in request.headers.items()}
    try:
        body = response.json()
        summary["json_type"] = type(body).__name__
        if isinstance(body, dict):
            summary["json_keys"] = list(body)
            if "conclusion" in body:
                summary["conclusion"] = body["conclusion"]
    except ValueError:
        summary["json_type"] = "invalid"
    # Never dump bodies: auth endpoints may return tokens even without an error.
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="检查显式指定URL的响应元数据，不打印鉴权信息或响应正文")
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args(argv)
    import math
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout 必须为有限正数")
    parts = urlsplit(args.url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        parser.error("url 必须为HTTP/HTTPS地址且不含用户凭据")
    try:
        import requests
        with requests.Session() as session:
            response = session.get(args.url, timeout=(min(5, args.timeout), args.timeout))
            analyze_response(response)
            return int(not response.ok)
    except Exception as error:
        parser.exit(1, f"请求失败 ({type(error).__name__})\n")


if __name__ == "__main__":
    raise SystemExit(main())
