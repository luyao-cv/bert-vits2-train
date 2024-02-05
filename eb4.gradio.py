import os 
# try:
#     import soundfile
# except:
#     os.system("pip install -r requirements.txt")

import gradio as gr
import json
import requests as req
import soundfile as sf
from pathlib import Path
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)
print(script_path)
from langchain.chat_models import ErnieBotChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import time
from langchain.embeddings import ErnieEmbeddings
import yaml
import random
import pandas as pd
from tqdm import tqdm
from langchain.vectorstores import Milvus
from langchain.prompts import PromptTemplate
import json
import requests as req
import soundfile as sf
from IPython.display import Audio



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

api = "https://ttsms.ai-lab.top"
spklist = "https://tirs.ai-lab.top"
token = "e51b77ca20bfe20ea0f631b2f68a0c8b"
def infer(speaker,chatbot,sdp,noise,noisew,length):

    text = chatbot[-1][1]
        
    speed = (100 - length) / 100
    headers = {'Content-Type': 'application/json'}
    infer_info = {'token': token,'speaker': speaker,'text': text,'sdp_ratio': sdp,'noise': noise,'noisew': noisew,'length': speed}
    resp = req.post(url=api, headers=headers, data=json.dumps(infer_info))
    data = json.loads(resp.text)
    os.system("wget -O temp.wav "+data["audio"]+"")

    data["message"] = data["message"].replace("\\n","\n").split("\n")[3:]

    return 'temp.wav'

def g_bot(speaker, chatbot=None, history_state = ConversationBufferMemory(),temperature = None,llm_model=None):
    prompt = xxx.format(speaker)                
    try:
        if llm_model is None:
            llm_model = init_model(temperature)
        output = init_base_chain(llm_model,history=history_state,user_question=prompt)
    except Exception as e:
        raise e
        
    chatbot = [[None, output.split("你的开场白为：")[1].split('\n')[0]]]
    
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
with open("new_cof.yaml", "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
ernie_client_id = config["API"]["ernie_client_id"]
ernie_client_secret = config["API"]["ernie_client_secret"] 


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
block = gr.Blocks(css="footer {visibility: hidden}",title="角色扮演对话")
with block:
    speakers = get_spk(spklist)
    history = ConversationBufferMemory() #历史记录
    history_state = gr.State(history) #历史记录的状态
    llm_model_state = gr.State() #llm模型的状态
    trash = gr.State() #垃圾桶
    with gr.Row():
        #设置行

        with gr.Column(scale=1):
            with gr.Accordion("模型配置", open=True):
                with gr.Row():
                    speaker = gr.Dropdown(choices=speakers, value=speakers[0], label="角色")
                    search = gr.Textbox(label="搜索角色", lines=1)
                    with gr.Row():
                        btn2 = gr.Button(value="搜索")
                        btn_ensure = gr.Button(value="生成")
                    
                    text = gr.TextArea(label="角色背景", placeholder="选择角色，AI生成角色背景......", lines=10, interactive=True)
                    
                with gr.Column():
                    with gr.Row():
                        sdp_ratio = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.1, label="SDP/DP 混合比")
                        noise_scale = gr.Slider(minimum=0.1, maximum=2, value=0.6, step=0.1, label="感情")
                    with gr.Row():
                        noise_scale_w = gr.Slider(minimum=0.1, maximum=2, value=0.8, step=0.1, label="音素长度")
                        length_scale = gr.Slider(minimum=-99, maximum=99, value=0, step=0.1, label="语速(%)")
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
                    placeholder="我有很多话想跟你说......",
                    lines=2,
                )
            with gr.Row():
                audio_output = gr.Audio(label="输出音频")
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                #刷新
                clear = gr.Button("刷新", variant="secondary")

            def clear_():
                chatbot = []
                history_state = ConversationBufferMemory()
                return "", chatbot, history_state

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
        infer, inputs=[speaker, chatbot, sdp_ratio, noise_scale, noise_scale_w, length_scale], outputs=[audio_output]
    )
    
    #刷新按钮
    clear.click(clear_, inputs=[], outputs=[message, chatbot, history_state])
    #send按钮
    submit.click(user, [message, chatbot], [message,chatbot], queue=False).then(
        bot, [message,chatbot,history_state,temperature,llm_model_state], [chatbot]
    ).then(
        infer, inputs=[speaker, chatbot, sdp_ratio, noise_scale, noise_scale_w, length_scale], outputs=[audio_output]
    )
    #回车
    message.submit(user, [message, chatbot], [message,chatbot], queue=False).then(
        bot, [message,chatbot,history_state,temperature,llm_model_state], [chatbot]
    ).then(
        infer, inputs=[speaker, chatbot, sdp_ratio, noise_scale, noise_scale_w, length_scale], outputs=[audio_output]
    )


# 启动参数
block.queue(concurrency_count=config['block']['concurrency_count']).launch(
    debug=config['block']['debug'],
    # server_name=config['block']['server_name'],
    # server_port=config['block']['server_port'],
    server_name = "0.0.0.0",
    server_port=8001,
    share=True,
  
) 

