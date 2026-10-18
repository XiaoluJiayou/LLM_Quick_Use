"""Explicit CLI wrapper; importing this module never calls a cloud service."""
from _aliyun_video import create_client, submit_video, pull_result, main as _main


def main(argv=None):
    return _main(argv, operation='submit', default_service='liveStreamDetection')


if __name__ == "__main__":
    raise SystemExit(main())
