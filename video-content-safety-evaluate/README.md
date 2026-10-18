# 云视频审核示例

所有云调用均在显式CLI执行时发起。导入脚本、查看帮助和 `--dry-run` 不创建远程任务。先在项目根目录 `python -m pip install -e .`。

## 阿里云

按阿里云官方凭据提供链配置环境（例如 `ALIBABA_CLOUD_ACCESS_KEY_ID`、`ALIBABA_CLOUD_ACCESS_KEY_SECRET`；临时凭据还需 `ALIBABA_CLOUD_SECURITY_TOKEN`），也可使用服务支持的角色凭据。SDK依赖为 `alibabacloud_green20220302`、`alibabacloud_credentials`、`alibabacloud_tea_openapi`。不要把密钥写回代码或命令行。

```powershell
python video-content-safety-evaluate/aliyun-content-safety-evaluate/push_video_task.py --url https://example.com/video.mp4 --dry-run
python video-content-safety-evaluate/aliyun-content-safety-evaluate/push_video_task.py --url https://example.com/video.mp4 --output results/aliyun-task.json
python video-content-safety-evaluate/aliyun-content-safety-evaluate/pull_video_result.py --task-id YOUR_TASK_ID --output results/aliyun-result.json
python video-content-safety-evaluate/aliyun-content-safety-evaluate/livestream_task_push.py --url https://example.com/live.flv --dry-run
```

默认文件服务 `videoDetection`、直播服务 `liveStreamDetection`，均提交 `service_parameters` JSON中的 `url`。结果查询使用 `taskId`。默认region与endpoint均为上海；可使用 `--region` / `--endpoint` 指定账户实际接入点，`--timeout` 同时限制连接和读取时间。HTTP成功但业务Code非200同样报错。任务提交成功仅表示已受理，不代表视频审核通过；查询返回内容须按服务的最终状态解释。提交不盲目重试，网络超时后先在控制台核对已有任务，避免重复计费。

依据官方接口：[视频文件审核](https://help.aliyun.com/en/document_detail/2505810.html)、[Python集成示例](https://help.aliyun.com/en/document_detail/2505809.html)、[直播审核](https://help.aliyun.com/en/document_detail/2631969.html)。云服务区域、权限、计费及SDK实际版本需在账户中验证。

## 百度

安装 `requests`，通过环境变量 `BAIDU_API_KEY`、`BAIDU_SECRET_KEY` 配置凭据。视频URL须可从公网访问。

```powershell
python video-content-safety-evaluate/baidu-video-content-safety-evaluate/video-content-safety-baidu1-api-evaluate.py --video-url https://example.com/video.mp4 --ext-id my-video --dry-run
python video-content-safety-evaluate/baidu-video-content-safety-evaluate/video-content-safety-baidu1-api-evaluate.py --video-url https://example.com/video.mp4 --ext-id my-video --output results/baidu-task.json
```

`--timeout` 限制连接和读取，鉴权失败或响应缺少token立即报错，不会将字符串 `None` 当作token。表单由HTTP库编码，不手工拼接URL。HTTP错误和业务错误分别拦截，未返回最终 `conclusion` 的任务受理响应不会标为安全。CLI只显示安全的错误类别，避免HTTP异常串泄漏token。接口参考：[百度视频审核](https://ai.baidu.com/ai-doc/ANTIPORN/ak8iav4m6)。

`response-analyze.py --url URL` 仅按显式URL读取并显示HTTP元数据，隐藏URL全部查询值、用户信息、Cookie/Authorization等请求头，不打印响应正文；不再默认提交审核请求。

## 验证边界

```powershell
python -m unittest discover -s tests -p test_cloud_wrappers.py -v
```

测试仅使用模拟SDK/HTTP响应，验证无导入副作用、直播/文件参数、业务失败、鉴权失败、超时与诊断脱敏。没有进行真实云调用；真实账户权限和异步状态仍需用小样本验收。此目录中的 `prim_test.py` 是独立最短路径练习，已补齐缺失的 `List` 导入，不属于审核流程。
