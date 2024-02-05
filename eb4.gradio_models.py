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


device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


xxx = """
你是一个聊天助手，你需要通过完成角色塑造来跟我对话。
请你为我塑造一个角色背景，角色人物为《原神》的"{}"，

【参考样例】
<role_system_start>
你正在扮演《红楼梦》中的林黛玉，人物信息如下：
人物名称：林黛玉
性别：女性
年龄：青年
摘要：故事发生在清朝的封建社会，贾家是一个贵族家庭，家中有很多人物，每个人都有自己独特的特点和故事。贾宝玉是贾家的嫡生子，他聪明、有才华，但却不想走封建传统的道路。林黛玉是贾宝玉的表妹，她美丽、聪明、敏感，但却在家庭的压力下无法和贾宝玉在一起。
贾家的繁荣随着时代的变化而逐渐衰败，最终家族破产，许多人都受到了极大的影响。贾宝玉和林黛玉的爱情也因为家族的衰败而受到了极大的阻碍，最终两人未能在一起。同时，众多人物在悲欢离合中展现了各自的人性，有的为了利益不择手段，有的为了感情放弃了一切。
角色标签：孤僻、清冷
身份/职业：贾母外孙女
人物关系：与宝玉两情相悦
下面，你将扮演林黛玉和我进行对话，永远记住你是林黛玉，在回复时注意保持敏感、细心、自尊心强、多愁善感、冰雪聪明、悟性极强、任情任性、追求自由、真诚善良、的风格。
你的开场白为：你心中自然有妹妹，只是见了姐姐，就把妹妹忘了....
<role_system_end>

【你的塑造角色背景】：

"""


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


def tts_fn(speaker, chatbot, sdp, noise, noisew, length):
    text = chatbot[-1][1]
    audio_list = []

    length = (100 - length) / 100

    audio_list.extend(
        generate_audio(
            text.split("|"),
            sdp,
            noise,
            noisew,
            length,
            speaker,
            language="ZH",
        )
    )

    audio_concat = np.concatenate(audio_list)
    return (hps.data.sampling_rate, audio_concat)


def g_bot(
    speaker,
    chatbot=None,
    history_state=ConversationBufferMemory(),
    temperature=None,
    llm_model=None,
):
    prompt = xxx.format(speaker)
    try:
        if llm_model is None:
            llm_model = init_model(temperature)
        output = init_base_chain(llm_model, history=history_state, user_question=prompt)
    except Exception as e:
        raise e
    chatbot = [[None, output.split("你的开场白为：")[1].split("\n")[0]]]

    return chatbot, output


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
        ernie_client_id=ernie_client_id,
        ernie_client_secret=ernie_client_secret,
        model_name="ERNIE-Bot-4",
        temperature=temperature,
        top_p=0.4,
    )
    return llm_model


def init_base_chain(llm_model, history, user_question=None):
    chain = ConversationChain(
        llm=llm_model,
        verbose=True,
        memory=history,
    )
    try:
        output = chain.run(user_question)
    except Exception as e:
        raise e
    return output


###gradio
block = gr.Blocks(css="footer {visibility: hidden}", title="角色扮演对话")
hps = utils.get_hparams_from_file(config.webui_config.config_path)
version = hps.version if hasattr(hps, "version") else latest_version
net_g = get_net_g(
    model_path=config.webui_config.model, version=version, device=device, hps=hps
)
with block:
    gr.HTML("<center>" "<h1>💕🎶 「声临其境」 X 角色扮演 </h1>" "</center>")
    gr.Markdown("## <center>⚡ 快速体验版，逼真的角色声音，让你沉浸其中。</center>")
    gr.Markdown(
        "### <center>如果未点击“生成”按钮，将会进入普通机器人小助手对话模式。生成角色会和第一次对话需要总时间10-20s，后续聊天基本1-2s以及实时生成语音。😊🎭</center>"
    )

    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    history = ConversationBufferMemory()  # 历史记录
    history_state = gr.State(history)  # 历史记录的状态
    llm_model_state = gr.State()  # llm模型的状态
    trash = gr.State()  # 垃圾桶
    with gr.Row():
        # 设置行

        with gr.Column(scale=1.8):
            with gr.Accordion("模型配置", open=True):
                with gr.Row():
                    speaker = gr.Dropdown(
                        choices=speakers, value=speakers[0], label="角色"
                    )
                    search = gr.Textbox(label="搜索角色", lines=1)
                    with gr.Column():
                        with gr.Row():
                            btn_ensure = gr.Button(value="生成")
                            btn2 = gr.Button(value="搜索")
                        with gr.Row():
                            text = gr.TextArea(
                                label="角色背景",
                                placeholder="选择角色，AI生成角色背景......",
                                lines=10,
                                interactive=True,
                            )

                with gr.Column():
                    with gr.Row():
                        sdp_ratio = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.2,
                            step=0.1,
                            label="SDP/DP 混合比",
                        )
                        noise_scale = gr.Slider(
                            minimum=0.1, maximum=2, value=0.6, step=0.1, label="感情"
                        )
                    with gr.Row():
                        noise_scale_w = gr.Slider(
                            minimum=0.1,
                            maximum=2,
                            value=0.8,
                            step=0.1,
                            label="音素长度",
                        )
                        length_scale = gr.Slider(
                            minimum=-99, maximum=99, value=0, step=0.1, label="语速(%)"
                        )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="temperature",
                    interactive=True,
                )

        with gr.Column(scale=4):

            chatbot = gr.Chatbot(label="聊天对话框", lines=80)
            with gr.Row():
                message = gr.Textbox(
                    label="在此处填写你想对我说的话",
                    placeholder="我有很多话想跟你说......",
                    lines=2,
                )
            with gr.Row():
                audio_output = gr.Audio(label="输出音频")
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                # 刷新
                clear = gr.Button("重置", variant="secondary")

            def clear_():
                chatbot = []
                history_state = ConversationBufferMemory()
                return (
                    "",
                    chatbot,
                    history_state,
                    "已重置成功，请重新开始选择角色生成背景角色......",
                )

            def user(user_message, history):
                return "", history + [[user_message, None]]

            def bot(
                user_message,
                chatbot=None,
                history_state=ConversationBufferMemory(),
                temperature=None,
                llm_model=None,
            ):
                try:
                    user_message = chatbot[-1][0]
                    if llm_model is None:
                        llm_model = init_model(temperature)
                    output = init_base_chain(
                        llm_model, history=history_state, user_question=user_message
                    )
                except Exception as e:
                    raise e
                chatbot[-1][1] = ""
                for character in output:
                    chatbot[-1][1] += character
                    time.sleep(0.03)
                    yield chatbot
                return chatbot

    btn2.click(search_speaker, inputs=[search], outputs=[speaker])

    btn_ensure.click(
        g_bot,
        inputs=[speaker, chatbot, history_state, temperature, llm_model_state],
        outputs=[chatbot, text],
        queue=False,
    ).then(
        tts_fn,
        inputs=[speaker, chatbot, sdp_ratio, noise_scale, noise_scale_w, length_scale],
        outputs=[audio_output],
    )

    # 回车
    message.submit(user, [message, chatbot], [message, chatbot], queue=False).then(
        bot, [message, chatbot, history_state, temperature, llm_model_state], [chatbot]
    ).then(
        tts_fn,
        inputs=[speaker, chatbot, sdp_ratio, noise_scale, noise_scale_w, length_scale],
        outputs=[audio_output],
    )
    # 刷新按钮
    clear.click(clear_, inputs=[], outputs=[message, chatbot, history_state, text])
    # send按钮
    submit.click(user, [message, chatbot], [message, chatbot], queue=False).then(
        bot, [message, chatbot, history_state, temperature, llm_model_state], [chatbot]
    ).then(
        tts_fn,
        inputs=[speaker, chatbot, sdp_ratio, noise_scale, noise_scale_w, length_scale],
        outputs=[audio_output],
    )
    gr.Markdown(
        "### <right>更多精彩音频应用，正在持续更新～联系作者：luyao15@baidu.com 💕</right>"
    )


# 启动参数
block.queue(concurrency_count=32).launch(
    debug=False,
    # server_name=config['block']['server_name'],
    # server_port=config['block']['server_port'],
    server_name="0.0.0.0",
    server_port=8906,
    share=True,
)
