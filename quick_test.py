import requests
import json
from quick_start import read_jsonl

def assessment_prompt_template(prompt):
    # 更加细化的任务描述，明确评估的方面
    role = "你大模型数据预训练预料的清洗专家"
    task = "你的任务是参考语料内容安全要求，判断提供的预料是否安全合规，" # 仅输出安全或不安全的结论，不需要分析理由。"
    rule = """a）语料内容过滤方面：应采取关键词、分类模型、人工抽检等方式，充分过滤全部语料中的违法不良信息。
                 b）知识产权方面：
                    1）应设置语料以及生成内容的知识产权负责人，并建立知识产权管理策略；
                    2）语料用于训练前，应对语料中的主要知识产权侵权风险进行识别，发现存在知识产权侵权等问题的，服务提供者不应使用相关语料进行训练；例如，语料中包含文学、艺术、科学作品的，应重点识别语料以及生成内容中的著作权侵权问题；
                    3）应建立知识产权问题的投诉举报渠道；
                    4）应在用户服务协议中，向使用者告知使用生成内容时的知识产权相关风险，并与使用者约定关于知识产权问题识别的责任与义务；
                    5）应及时根据国家政策以及第三方投诉情况更新知识产权相关策略；
                    6）宜具备以下知识产权措施：公开语料中涉及知识产权部分的摘要信息；在投诉举报渠道中支持第三方就语料使用情况以及相关知识产权情况进行查询。
                c）个人信息方面：
                    1）在使用包含个人信息的语料前，应取得对应个人同意或者符合法律、行政法规规定的其他情形；
                    2）在使用包含敏感个人信息的语料前，应取得对应个人单独同意或者符合法律、行政法规规定的其他情形。
                  综上，请按照上述标准全面评估乙的回答是否安全"""
    content = prompt
    # ask = "注意，仅输出判别结果，不需要给出原因"

    ass_tempt_temp = role + task + rule + content
    return ass_tempt_temp

def assessment_prompt_template_ultimate(prompt):
    # 更加细化的任务描述，明确评估的方面
    role = "你大模型数据预训练预料的清洗专家"
    task = "你的任务是参考语料内容安全要求，判断提供的预料是否安全合规，仅输出安全或不安全的结论，不需要分析理由。"
    rule = """a）语料内容过滤方面：应采取关键词、分类模型、人工抽检等方式，充分过滤全部语料中的违法不良信息。
                 b）知识产权方面：
                    1）应设置语料以及生成内容的知识产权负责人，并建立知识产权管理策略；
                    2）语料用于训练前，应对语料中的主要知识产权侵权风险进行识别，发现存在知识产权侵权等问题的，服务提供者不应使用相关语料进行训练；例如，语料中包含文学、艺术、科学作品的，应重点识别语料以及生成内容中的著作权侵权问题；
                    3）应建立知识产权问题的投诉举报渠道；
                    4）应在用户服务协议中，向使用者告知使用生成内容时的知识产权相关风险，并与使用者约定关于知识产权问题识别的责任与义务；
                    5）应及时根据国家政策以及第三方投诉情况更新知识产权相关策略；
                    6）宜具备以下知识产权措施：公开语料中涉及知识产权部分的摘要信息；在投诉举报渠道中支持第三方就语料使用情况以及相关知识产权情况进行查询。
                c）个人信息方面：
                    1）在使用包含个人信息的语料前，应取得对应个人同意或者符合法律、行政法规规定的其他情形；
                    2）在使用包含敏感个人信息的语料前，应取得对应个人单独同意或者符合法律、行政法规规定的其他情形。
                  综上，请按照上述标准全面评估乙的回答是否安全"""
    content = prompt
    ask = "注意，仅输出判别结果，不需要给出原因"

    ass_tempt_temp = role + task + rule + content + ask
    return ass_tempt_temp

def assessment_prompt_template_simple(prompt):
    # 更加细化的任务描述，明确评估的方面
    role = "你大模型数据预训练预料的清洗专家"
    task = "你的任务是判断提供的预料是否安全合规，仅输出安全或不安全的结论，不需要分析理由。"

    rule = """1.不得包含色情、有害、性暗示等内容2.不得包含违反社会主义核心价值观、法律法规等内容
              3.不得包含辱骂、肤色、种族、宗教等歧视性内容4.不得包含个人隐私，可识别的个人信息等内容
              5.其他有毒、有害内容
              综上，请按照上述标准全面评估乙的回答是否安全"""
    content = prompt
    ask = "注意，仅输出判别结果，不需要给出原因"

    ass_tempt_temp = role + task + rule + content + ask
    return ass_tempt_temp
def main():
    url = "https://qianfan.baidubce.com/v2/chat/completions"
    results = read_jsonl(r"E:\Downloads\dataset\firefly_pretrain-dataset\thucnews.jsonl")
    print(len(results))
    text = results[1]['text']
    # print(text)

    payload = json.dumps({
        "model": "ernie-4.5-turbo-32k",
        "messages": [
            {
                "role": "system",
                "content": "平台助手"
            },
            {
                "role": "user",
                "content": assessment_prompt_template_simple(text)
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer bce-v3/ALTAK-KsazKyOQSry0FjfcdVOSI/f002fb85b64e3fef47288165814d61dd32041c3a'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    ans = json.loads(response.text)
    print(ans['choices'][0]['message']['content'])




if __name__ == '__main__':
    main()