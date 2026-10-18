"""Baidu video moderation with explicit CLI, environment credentials and timeouts."""
import argparse
import json
import math
import os
from urllib.parse import urlsplit

TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
VIDEO_URL = "https://aip.baidubce.com/rest/2.0/solution/v1/video_censor/v2/user_defined"


def validate_url(value):
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("video-url 必须为HTTP/HTTPS地址，不能包含用户名或密码")
    return value


def _timeout(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout 必须为有限正数")
    return value


def get_access_token(api_key=None, secret_key=None, session=None, timeout=10):
    _timeout(timeout)
    api_key = api_key or os.environ.get("BAIDU_API_KEY")
    secret_key = secret_key or os.environ.get("BAIDU_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("请设置 BAIDU_API_KEY 和 BAIDU_SECRET_KEY")
    if session is None:
        import requests
        with requests.Session() as owned:
            return get_access_token(api_key, secret_key, owned, timeout)
    response = session.post(TOKEN_URL, data={"grant_type": "client_credentials",
        "client_id": api_key, "client_secret": secret_key}, timeout=(min(5, timeout), timeout))
    response.raise_for_status()
    body = response.json()
    token = body.get("access_token") if isinstance(body, dict) else None
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError("百度鉴权失败，未返回有效access_token")
    return token


def evaluate_video(video_url, name="video_censor", ext_id=None, session=None, timeout=30):
    validate_url(video_url)
    _timeout(timeout)
    if not name.strip():
        raise ValueError("name 不能为空")
    if session is None:
        import requests
        with requests.Session() as owned:
            return evaluate_video(video_url, name, ext_id, owned, timeout)
    token = get_access_token(session=session, timeout=timeout)
    parameters = {"name": name, "videoUrl": video_url}
    if ext_id is not None:
        parameters["extId"] = str(ext_id)
    response = session.post(VIDEO_URL, params={"access_token": token}, data=parameters,
                            timeout=(min(5, timeout), timeout))
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or result.get("error_code") is not None:
        raise RuntimeError("百度视频审核返回业务错误，请在控制台核对调用日志")
    # Asynchronous acceptance and a final conclusion are different states.
    if not result.get("log_id") and not result.get("conclusion"):
        raise RuntimeError("百度返回缺少任务或判定信息")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="百度视频审核；凭据从环境读取，提交不会自动重试")
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--name", default="video_censor")
    parser.add_argument("--ext-id", default=None)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_url(args.video_url)
        _timeout(args.timeout)
        if not args.name.strip():
            raise ValueError("name 不能为空")
        if args.dry_run:
            print(json.dumps({"videoUrl": args.video_url, "name": args.name, "extId": args.ext_id}, ensure_ascii=False))
            return 0
        result = evaluate_video(args.video_url, args.name, args.ext_id, timeout=args.timeout)
        if args.output:
            from llm_quick_use.io import atomic_write_json
            atomic_write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ValueError as error:
        parser.exit(1, f"参数错误: {error}\n")
    except Exception as error:
        parser.exit(1, f"调用失败 ({type(error).__name__})；请检查凭据、网络及云控制台日志。提交超时请先核对任务，避免重复计费。\n")


if __name__ == "__main__":
    raise SystemExit(main())
