# coding=utf-8
# python version >= 3.6
from alibabacloud_green20220302.client import Client
from alibabacloud_green20220302 import models
from alibabacloud_tea_openapi.models import Config
import json

config = Config(
            # 阿里云账号AccessKey拥有所有API的访问权限，建议您使用RAM用户进行API访问或日常运维。
            # 强烈建议不要把AccessKey ID和AccessKey Secret保存到工程代码里，否则可能导致AccessKey泄露，威胁您账号下所有资源的安全。
            # 常见获取环境变量方式：
            # 获取RAM用户AccessKey ID：os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
            # 获取RAM用户AccessKey Secret：os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
            access_key_id='',
            access_key_secret='',
            # 连接时超时时间，单位毫秒（ms）。
            connect_timeout=3000,
            # 读取时超时时间，单位毫秒（ms）。
            read_timeout=6000,
            # 接入区域和地址请根据实际情况修改。
            region_id='cn-beijing',
            endpoint='green-cip.cn-shanghai.aliyuncs.com'
)

clt = Client(config)

# 提交任务时返回的taskId。
serviceParameters = {
    "taskId": 'vi_f_8WWwWuNhTqMp2EDalTHoK2-1CZVzN'
}
videoModerationResultRequest = models.VideoModerationResultRequest(
    # 检测类型：videoDetection
    service='videoDetection',
    service_parameters=json.dumps(serviceParameters)
)

try:
    response = clt.video_moderation_result(videoModerationResultRequest)
    if response.status_code == 200:
        # 调用成功。
        # 获取审核结果。
        result = response.body
        print('response success. result:{}'.format(result))
        # print(result.data)
        # 探索result对象的所有属性
        print("result对象的属性和方法:", dir(result))
        # 查看具体属性值
        if hasattr(result, '__dict__'):
            print("result的__dict__:", result.__dict__)


    else:
        print('response not success. status:{} ,result:{}'.format(response.status_code, response))
except Exception as err:
    print(err)