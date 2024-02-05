import gradio as gr  
import io  
import os  
import requests
import json
import soundfile as sf
import numpy as np
import scipy
import requests
import base64

API_URL = "https://r6n8kck6t9q0tfke.aistudio-hub.baidu.com"

# 设置鉴权信息
headers = {
    # 请前往 https://aistudio.baidu.com/index/accessToken 查看 访问令牌 并替换
    "Authorization": "token b551128a38b4f10093168390aed3caf7e3a1d1e5",
    "Content-Type": "application/json"
}
text = "你好，我是你的专属人工智能小助手"



def text_to_speech(text):  
    # 使用gTTS将文本转换为语音  
    input_json = {
    "text":text
    }
    speech_file = requests.post("http://10.21.226.179:8920/tts", json=input_json)
    return speech_file  
  
# def on_submit(text, speech_file):  
#     # 播放语音文件  
#     os.system(f"mpg123 -q - {speech_file.name}")  
  
iface = gr.Interface(fn=text_to_speech, inputs="text", outputs="audio", title="Text to Speech", description="Enter text and listen to the speech output")  
iface.launch(share=True, server_name="10.21.226.179", server_port=8910)  