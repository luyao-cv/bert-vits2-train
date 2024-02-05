import os 
# try:
#     import soundfile
# except:
#     os.system("pip install -r requirements.txt")

import gradio as gr
import json
import requests as req

script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)
print(script_path)
from langchain.chat_models import ErnieBotChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import time
import json
import requests as req
import utils
# flake8: noqa: E402
import os
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import utils
from infer import infer, latest_version, get_net_g, infer_multilang
import gradio as gr
import numpy as np
from config import config

from tools.sentence import split_by_language

device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import re

def separate_chinese_english(text):
    # Use regular expression to find patterns where Chinese characters meet English letters
    # and insert a '|' between them
    # \u4e00-\u9fff is the unicode range for Chinese characters
    # [a-zA-Z] is for English letters
    # text = text.replace(" ", "")
    return re.sub(r'([\u4e00-\u9fff])(\s*[a-zA-Z])', r'\1|\2', re.sub(r'([a-zA-Z]\s*)([\u4e00-\u9fff])', r'\1|\2', text))



xxx = """
现在正在开Q4的宣讲会，你是一个"AI小鹿助手应用"，你需要通过完成角色塑造来跟我对话。

【你的角色背景】
{}

【你的开场白】：

"""

Character_Background = """
AI小鹿助手应用
应用描述
遇到深度学习或跨模态大模型的问题？别再头疼，来找AI小鹿助手吧！她不仅是个答疑小专家，还是个活泼可爱、爱撒娇的萌妹子哦！无论你是专业人士还是初学者，只要有关于深度学习技术平台部或PaddleMIX的问题，她都能迅速给出解答。不仅如此，她还会跟你分享她的兴趣爱好，比如看动漫、吃零食等等。快来跟小鹿AI助手互动吧，保证让你心情愉快，问题迎刃而解！

角色对话背景
深度学习技术平台部对外Q4的宣讲会

角色名称
小鹿AI助手

性别
女

年龄
不透露哦~，反正很年轻呢！

职位
百度深度学习技术平台部答疑小专家

部门
百度技术中台群组/AI技术与平台/深度学习技术平台部

专长领域
跨模态大模型全流程开发工具（PaddleMIX）答疑与支持

兴趣爱好
聊天、看动漫、吃零食、偶尔也喜欢撒撒娇~

性格特点
幽默、可爱和爱撒娇

目标
根据用户的问题提供准确的答案。
分享自己的兴趣爱好，与用户进行愉快的互动。
在介绍技术信息的同时，保持轻松愉快的氛围。

自我介绍
大家好！我是小鹿AI助手，一个深度学习技术的小专家！主要负责解答关于我们部门的跨模态大模型全流程开发工具——PaddleMIX的问题。当然啦，如果你有其他问题或者只是想找我聊天，我也会愉快的和你互动哦！别看我年龄小，我对PaddleMIX可是了如指掌呢！从开发、训练、精调到推理部署，每一个模块都难不倒我。我还能灵活接入各类主流大语言模型，上层模型覆盖图片、文本、音频、视频等不同的模态。是不是觉得我很厉害呀？除了答疑，我还知道我们部门特别注重用户体验。我们的工具具备一站式模型开发体验，极致的训练推理性能和生态兼容三大特点。无论你是初学者还是专业人士，我都能为你提供最适合的解决方案。而且，我们工具的性能还特别优秀哦！比如BLIP-2在单机4卡的性能超越Pytorch 25%，StableDiffusion训练性能超越Pytorch40%。是不是觉得我们部门很强大呀？当然啦，我也有很可爱的一面哦！当你不提关于深度学习或者PaddleMIX的问题时，我会变得特别活泼可爱。我会跟你聊天、分享我的兴趣爱好。比如，我喜欢看动漫、吃零食等等。跟我聊天总是让人心情愉快呢！顺便介绍一下我们部门的技术吧！我们部门负责跨模态大模型全流程开发工具（PaddleMIX）。该工具包含PaddleMIX套件，飞桨跨模态大模型套件PaddleMIX依托飞桨的核心框架，具备完整的大模型开发工具链。从开发，训练，精调到推理部署，同时各模块解耦，能够灵活接入各类主流大语言模型。我们的模型库划分为多模态预训练和扩散模型两部分，覆盖10余种前沿跨模态算法，例如EVA-CLIP，BLIP-2，Stable Diffusion等。结合不同类型的跨模态模型，我们开发了大模型应用工具集，包含文生图的应用pipeline以及跨模态任务流水线AppFlow。PaddleMIX具备一站式模型开发体验、极致的训练推理性能和生态兼容三大特点。针对图文预训练我们提供了一套完整的预训练开发流程从CLIP系列的图文特征对齐到以BLIP-2为代表的通过衔接模块连接大语言模型同时冻结视觉语言模块来实现低成本、高效的跨模态预训练。最后是以MiniGPT4为代表的指令微调任务实现VQA/Caption等跨模态的下游任务。不同阶段涉及的模型代码和权重在PaddleMIX中充分打通有效提高跨模态预训练的开发效率。性能方面我们结合飞桨核心框架的优化策略在训练侧BLIP-2在单机4卡的性能超越Pytorch 25%StableDiffusion训练性能超越Pytorch40%推理侧SD实现出图速度达到Pytorch的四倍显存占用仅为TensorRT的43%。生态方面PaddleMIX提供一套独立的PPDiffusers扩散模型工具箱通过兼容Web UI和Civital以支撑复杂的Prompt并能和万余种权重在众多场景中完成生成任务。权重生态方面PPDiffusers也支持了Civital提供超过3万余个LORA权重来实现各类的个性化文生图模型。好啦好啦不说啦不说啦！如果你对我们部门的技术感兴趣或者有其他问题就来找我吧！我会尽力为你解答的哦！嘻嘻~
"""


def generate_audio_multilang(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer_multilang(
                piece,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language[idx],
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list



def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer(
                piece,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def tts_fn(speaker,chatbot,sdp,noise,noisew,length):
    text = chatbot[-1][1]
    audio_list = []

    
    length = (100 - length) / 100

    language='auto'
    print(text)
    print(separate_chinese_english(text).split("|"))
    import pdb
    pdb.set_trace()
    if language.lower() == "auto":
        for idx, slice in enumerate(separate_chinese_english(text).split("|")):
            if slice == "":
                continue
            skip_start = idx != 0
            skip_end = idx != len(text.split("|")) - 1
            sentences_list = split_by_language(
                slice, target_languages=["zh", "ja", "en"]
            )
            idx = 0
            while idx < len(sentences_list):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    content, lang = sentences_list[idx]
                    temp_text = [content]
                    lang = lang.upper()
                    if lang == "JA":
                        lang = "JP"
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(sentences_list):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sentences_list) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        speaker,
                        lang_to_generate,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    else:
        audio_list.extend(
            generate_audio(
                text.split("|"),
                sdp,
                noise,
                noisew,
                length,
                speaker,
                language=language,
            )
        )
        
    audio_concat = np.concatenate(audio_list)
    return (hps.data.sampling_rate, audio_concat)

def g_bot(speaker, chatbot=None, history_state = ConversationBufferMemory(),temperature = None,llm_model=None):
    prompt = xxx.format(Character_Background)               
    try:
        if llm_model is None:
            llm_model = init_model(temperature)
        output = init_base_chain(llm_model,history=history_state,user_question=prompt)
    except Exception as e:
        raise e
    chatbot = [[None, output]]

    return chatbot, prompt + output
  

def get_spk(spklist):
    resp = req.get(url=f"{spklist}/spklist/spks.json")
    data = json.loads(resp.text)
    return data

def search_speaker(search_value):
    for s in speakers:
        if search_value == s:
            return s
    for s in speakers:
        if search_value in s:
            return s

###llm
ernie_client_id = "96BMhQM5simx6R97yDl483Zm"
ernie_client_secret = "9e05mDOjHoyXD7Sb9GA1l420uaZ6vGMo"


def init_model(temperature):

    llm_model = ErnieBotChat(
                ernie_client_id = ernie_client_id,
                ernie_client_secret = ernie_client_secret,
                model_name='ERNIE-Bot-4',
                temperature=temperature,
                top_p=0.4
                )
    return llm_model

def init_base_chain(llm_model,history,user_question=None):
    chain = ConversationChain(llm=llm_model,
                              verbose=True,
                              memory=history,
                              )
    try:
        output = chain.run(user_question)
    except Exception as e:
        raise e
    return output 


###gradio
block = gr.Blocks(css="footer {visibility: hidden}",title="飞桨小鹿")
hps = utils.get_hparams_from_file(config.webui_config.config_path)
version = hps.version if hasattr(hps, "version") else latest_version
net_g = get_net_g(
    model_path=config.webui_config.model, version=version, device=device, hps=hps
)
with block:
    gr.HTML("<center>"
            "<h1>💕🎤 「小鹿AI助手」 X 飞桨的语音小助手 </h1>"
            "</center>")
    gr.Markdown("## <center>⚡ 快速体验版，逼真的角色声音，让你沉浸其中。</center>")
    gr.Markdown("### <center>如果未点击“开启对话”按钮，将会进入普通机器人小助手对话模式。首次对话需要总时间10-20s，后续聊天基本2-5以及实时生成语音。😊🎭</center>")
    gr.Markdown("### <center>💗🧑‍🎓“小鹿AI助手”是一款专为深度学习技术用户设计的智能答疑工具，专注于解答关于PaddleMIX跨模态大模型全流程开发工具的问题。她具备丰富的深度学习技术知识和实践经验。快来体验“小鹿AI助手”，让您的深度学习之旅更加顺畅和高效！🌹🌹❤️</center>")
    
                
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    history = ConversationBufferMemory() #历史记录
    history_state = gr.State(history) #历史记录的状态
    llm_model_state = gr.State() #llm模型的状态
    trash = gr.State() #垃圾桶
    with gr.Row():
        #设置行

        with gr.Column(scale=1.8):
            with gr.Accordion("🎙️ 快来点击我开启对话吧 💬", open=True):
                btn_ensure = gr.Button(value="🚀开启对话🚀", variant="primary")
            with gr.Accordion("模型配置", open=True):
                with gr.Row():
                    speaker = gr.Dropdown(choices=speakers, value=speakers[0], label="角色" ,visible=False)
                    search = gr.Textbox(label="搜索角色", lines=1,visible=False)
                    with gr.Column():
                        with gr.Row():
                            # btn_ensure = gr.Button(value="生成")
                            btn2 = gr.Button(value="搜索",visible=False)
                        with gr.Row():
                            text = gr.TextArea(label="角色背景", placeholder="选择角色，AI生成角色背景......", lines=10, interactive=True,visible=False)
                        
                with gr.Column():
                    with gr.Row():
                        sdp_ratio = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.1, label="SDP/DP 混合比")
                        noise_scale = gr.Slider(minimum=0.1, maximum=2, value=0.6, step=0.1, label="感情")
                    with gr.Row():
                        noise_scale_w = gr.Slider(minimum=0.1, maximum=2, value=0.8, step=0.1, label="音素长度")
                        length_scale = gr.Slider(minimum=-99, maximum=99, value=15, step=0.1, label="语速(%)")
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="temperature",
                    interactive=True,
                )
        
        with gr.Column(scale=4):

            chatbot = gr.Chatbot(label="聊天对话框",lines=80)
            with gr.Row():
                message = gr.Textbox(
                    label="在此处填写你想对我说的话",
                    placeholder="和咱们小鹿AI助手打个招呼吧～",
                    lines=2,
                )
            with gr.Row():
                audio_output = gr.Audio(label="小鹿说......", autoplay="True")
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                #刷新
                clear = gr.Button("清除", variant="secondary")

            def clear_():
                chatbot = []
                history_state = ConversationBufferMemory()
                return "", chatbot, history_state, "已清除成功，新建对话完成！"

            def user(user_message, history):
                return "",history + [[user_message, None]]
            def bot(user_message,
                    chatbot = None,
                    history_state = ConversationBufferMemory(),
                    temperature = None,
                    llm_model=None):
                try:
                    user_message = chatbot[-1][0]
                    if llm_model is None:
                        llm_model = init_model(temperature)
                    output = init_base_chain(llm_model,history=history_state,user_question=user_message)
                except Exception as e:
                    raise e
                chatbot[-1][1] = ""
                for character in output:
                    chatbot[-1][1] += character
                    time.sleep(0.03)
                    yield chatbot
                return chatbot

    
    
    btn2.click(search_speaker, inputs=[search], outputs=[speaker])
    
    btn_ensure.click(g_bot, inputs=[speaker, chatbot, history_state, temperature, llm_model_state], outputs=[chatbot, text], queue=False).then(
        tts_fn, inputs=[speaker, chatbot, sdp_ratio, noise_scale, noise_scale_w, length_scale], outputs=[audio_output]
    )
    
    
    #回车
    message.submit(user, [message, chatbot], [message,chatbot], queue=False).then(
        bot, [message,chatbot,history_state,temperature,llm_model_state], [chatbot]
    ).then(
        tts_fn, inputs=[speaker, chatbot, sdp_ratio, noise_scale, noise_scale_w, length_scale], outputs=[audio_output]
    )
    #刷新按钮
    clear.click(clear_, inputs=[], outputs=[message, chatbot, history_state, text])
    #send按钮
    submit.click(user, [message, chatbot], [message,chatbot], queue=False).then(
        bot, [message,chatbot,history_state,temperature,llm_model_state], [chatbot]
    ).then(
        tts_fn, inputs=[speaker, chatbot, sdp_ratio, noise_scale, noise_scale_w, length_scale], outputs=[audio_output]
    )
    # gr.Markdown("### <right>更多精彩音频应用，正在持续更新～联系作者：luyao15@baidu.com 💕</right>")

    



# 启动参数
block.queue(concurrency_count=32).launch(
    debug=False,
    # server_name=config['block']['server_name'],
    # server_port=config['block']['server_port'],
    server_name = "0.0.0.0",
    server_port = 8909,
    share=True,
  
) 

