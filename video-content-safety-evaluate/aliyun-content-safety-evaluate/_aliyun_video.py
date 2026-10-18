"""Explicit Aliyun submissions and result queries. No SDK import before execution."""
import argparse
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

SUBMIT_SERVICES = {"videoDetection", "videoDetectionByVL", "liveStreamDetection", "liveStreamDetectionByVL"}


def validate_url(value):
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("url 必须为可公开访问的HTTP/HTTPS地址，且不能包含用户名或密码")
    return value


def service_parameters(url=None, task_id=None):
    if task_id is not None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id 不能为空")
        return {"taskId": task_id.strip()}
    return {"url": validate_url(url or "")}


def create_client(region="cn-shanghai", endpoint=None, timeout=10):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout 必须为有限正数")
    from alibabacloud_green20220302.client import Client
    from alibabacloud_credentials.client import Client as CredentialClient
    from alibabacloud_tea_openapi.models import Config
    config = Config(credential=CredentialClient(), region_id=region,
                    endpoint=endpoint or f"green-cip.{region}.aliyuncs.com",
                    connect_timeout=int(timeout * 1000), read_timeout=int(timeout * 1000))
    return Client(config)


def normalize_response(response):
    if getattr(response, "status_code", None) != 200:
        raise RuntimeError("云服务返回非200 HTTP状态")
    body = response.body.to_map() if hasattr(response.body, "to_map") else response.body
    if not isinstance(body, dict) or str(body.get("Code")) != "200":
        raise RuntimeError("云服务返回业务错误，请在控制台核对调用日志")
    return body


def submit_video(url, service="videoDetection", client=None):
    if service not in SUBMIT_SERVICES:
        raise ValueError("不支持的审核服务")
    parameters = service_parameters(url=url)
    from alibabacloud_green20220302 import models
    request = models.VideoModerationRequest(service=service, service_parameters=json.dumps(parameters))
    # A submission may already be accepted when a response times out. Do not retry blindly.
    return normalize_response((client or create_client()).video_moderation(request))


def pull_result(task_id, service="videoDetection", client=None):
    if service not in {"videoDetection", "videoDetectionByVL"}:
        raise ValueError("结果查询仅支持视频文件审核服务")
    parameters = service_parameters(task_id=task_id)
    from alibabacloud_green20220302 import models
    request = models.VideoModerationResultRequest(service=service, service_parameters=json.dumps(parameters))
    return normalize_response((client or create_client()).video_moderation_result(request))


def main(argv=None, operation="submit", default_service="videoDetection"):
    parser = argparse.ArgumentParser(description="阿里云视频审核；导入与 --dry-run 不发起云调用")
    parser.add_argument("--task-id" if operation == "pull" else "--url", required=True)
    choices = {"videoDetection", "videoDetectionByVL"} if operation == "pull" else SUBMIT_SERVICES
    parser.add_argument("--service", choices=sorted(choices), default=default_service)
    parser.add_argument("--region", default="cn-shanghai")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--output", default=None, help="将完整响应保存为JSON，用于保留taskId")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout 必须为有限正数")
        parameters = service_parameters(task_id=args.task_id) if operation == "pull" else service_parameters(url=args.url)
        if args.dry_run:
            print(json.dumps({"service": args.service, "parameters": parameters}, ensure_ascii=False))
            return 0
        client = create_client(args.region, args.endpoint, args.timeout)
        result = pull_result(args.task_id, args.service, client) if operation == "pull" else submit_video(args.url, args.service, client)
        if args.output:
            from llm_quick_use.io import atomic_write_json
            atomic_write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ValueError as error:
        parser.exit(1, f"参数错误: {error}\n")
    except Exception as error:
        # SDK exceptions can contain signed URLs or credentials. Keep logs free of them.
        parser.exit(1, f"调用失败 ({type(error).__name__})；请检查凭据、端点、网络和云控制台日志。提交超时请先核对任务，避免重复计费。\n")
