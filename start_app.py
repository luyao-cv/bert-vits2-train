import os 
import soundfile
import gradio as gr
import json
import requests as req

script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

import time
import json
import utils
# flake8: noqa: E402
import os
import logging
import io
import torch
import utils
from infer import infer, latest_version, get_net_g, infer_multilang
import numpy as np
from config import config

from flask import Flask, request, send_file, jsonify
import base64

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


hps = utils.get_hparams_from_file(config.webui_config.config_path)
version = hps.version if hasattr(hps, "version") else latest_version
net_g = get_net_g(
    model_path=config.webui_config.model, version=version, device=device, hps=hps
)

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


def tts_fn(speaker,text,sdp,noise,noisew,length):
    
    audio_list = []
    
    
    length = (100 - length) / 100
    print(text)
    split_text = [text[i: i + 500] for i in range(0, len(text), 500)]

    # for i in range(len(split_text)):
    #     if len(split_text[i]) > 500:
    #         split_text.append(split_text[i][-1:])     
    for text in split_text:
        audio_list.extend(
            generate_audio(
                text.split("|"
                ),
                sdp,
                noise,
                noisew,
                length,
                speaker,
                language='ZH',
            )
        )
        
    audio_concat = np.concatenate(audio_list)
    return (hps.data.sampling_rate, audio_concat)

sdp_ratio = 0.2
noise_scale = 0.6
noise_scale_w = 0.8
length_scale = 15
speaker = "三月七"

_text = """
迪拜，位于阿拉伯联合酋长国（阿联酋）的东部沿海地区，是阿联酋最大的城市，也是该国七个酋长国之一迪拜酋长国的首府。它位于波斯湾的南岸，是中东地区的重要港口城市和经济中心。

迪拜的历史可以追溯到公元10世纪左右，当时这里是阿拉伯商人聚集的地方。随着时间的推移，迪拜逐渐发展成为了一个重要的贸易中心和商业城市。20世纪70年代，迪拜发现了石油资源，这使得迪拜的经济迅速崛起。

现在，迪拜已经成为了一个现代化、国际化的大都市。它的经济发展非常迅速，特别是在旅游、金融、房地产和制造业等领域。迪拜的旅游业非常发达，吸引了来自世界各地的游客前来观光和旅游。此外，迪拜还是中东地区重要的金融中心之一，许多国际银行和金融机构都在这里设有分支机构。

在建筑方面，迪拜拥有世界上最高的人工建筑——哈利法塔，高度达到了828米。此外，迪拜还有世界上面积最大的人工岛项目——棕榈岛，由多个岛屿组成，每个岛屿都拥有不同的功能和设施。

除了经济和建筑方面的成就外，迪拜还在文化和艺术方面取得了很大的进展。它举办了许多国际性的文化活动和艺术展览，吸引了来自世界各地的艺术家和文化爱好者前来参与。

总的来说，迪拜是一个充满活力和魅力的城市，它的经济发展和文化繁荣都为世界所瞩目。

"""
tts_fn(speaker,_text,sdp_ratio,noise_scale,noise_scale_w,length_scale)

app = Flask(__name__)

@app.route('/tts', methods=["POST"])
def post_tts_results():
    input_json = request.get_json()
    text = input_json['text']

    sampling_rate, audio_data = tts_fn(speaker,text,sdp_ratio,noise_scale,noise_scale_w,length_scale)

    out_wav_path = io.BytesIO()  
    soundfile.write(out_wav_path, audio_data, sampling_rate, format="wav")
    out_wav_path.seek(0)

    #返回base64格式
    audio_base64 = base64.b64encode(out_wav_path.read()).decode('utf-8')  
    return audio_base64

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8920) 
