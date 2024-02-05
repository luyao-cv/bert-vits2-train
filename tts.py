import requests
import json
import soundfile as sf
import numpy as np
import scipy
import requests
import base64
import os
import io
import soundfile

API_URL = "https://u5z1sbm484l9ec07.aistudio-hub.baidu.com"

# 设置鉴权信息
headers = {
    # 请前往 https://aistudio.baidu.com/index/accessToken 查看 访问令牌 并替换
    "Authorization": "token 4ce50e3378f418d271c480c8ddfa818537071dbe",
    "Content-Type": "application/json"
}
text = """
迪拜是阿拉伯联合酋长国人口最多的城市，也是该国七个酋长国之一迪拜酋长国的首府。它位于中东地区的中央，面向波斯湾，是一片平坦的沙漠之地。面积约为4114平方公里，占阿联酋全国总面积的5.8%，继阿布扎比之后排名第二。人口约为3,392,408人（2020年6月），约占阿联酋全国人口的41.9%，为人口最多的城市。

迪拜是中东地区的经济金融中心，也是中东地区旅客和货物的主要运输枢纽。石油收入曾经促进了迪拜的早期发展，但由于储量有限，生产水平较低，2010年以后，石油产业只占到迪拜国民生产总值的5%以下。继石油之后，迪拜的经济主要依靠旅游业、航空业、房地产和金融服务。

迪拜也通过大型建筑项目和体育赛事吸引全世界的目光，拥有世界上最高的人工建筑哈利法塔，还有世界上面积最大的人工岛项目棕榈岛。2018年，被GaWC评为年度世界一线城市第九位。

以上是对迪拜的简单介绍，如果需要更多信息，可以阅读地理书籍或请教地理老师。

迪拜，位于阿拉伯联合酋长国（阿联酋）的东部沿海地区，是阿联酋最大的城市，也是该国七个酋长国之一迪拜酋长国的首府。它位于波斯湾的南岸，是中东地区的重要港口城市和经济中心。

迪拜的历史可以追溯到公元10世纪左右，当时这里是阿拉伯商人聚集的地方。随着时间的推移，迪拜逐渐发展成为了一个重要的贸易中心和商业城市。20世纪70年代，迪拜发现了石油资源，这使得迪拜的经济迅速崛起。

现在，迪拜已经成为了一个现代化、国际化的大都市。它的经济发展非常迅速，特别是在旅游、金融、房地产和制造业等领域。迪拜的旅游业非常发达，吸引了来自世界各地的游客前来观光和旅游。此外，迪拜还是中东地区重要的金融中心之一，许多国际银行和金融机构都在这里设有分支机构。

在建筑方面，迪拜拥有世界上最高的人工建筑——哈利法塔，高度达到了828米。此外，迪拜还有世界上面积最大的人工岛项目——棕榈岛，由多个岛屿组成，每个岛屿都拥有不同的功能和设施。

除了经济和建筑方面的成就外，迪拜还在文化和艺术方面取得了很大的进展。它举办了许多国际性的文化活动和艺术展览，吸引了来自世界各地的艺术家和文化爱好者前来参与。

总的来说，迪拜是一个充满活力和魅力的城市，它的经济发展和文化繁荣都为世界所瞩目。


"""
input_json = {
    "text":text
}
# 请求服务 访问AI Studio部署API服务
results = requests.post(API_URL+"/tts", headers=headers, json=input_json)
print(results)
# 请求本地服务 访问AI Studio部署API服务
# results = requests.post("http://10.21.226.179:8920/tts", json=input_json)

audio_base64 = results.content  
# 解码 Base64 data
audio_data = base64.b64decode(audio_base64)

with open("output1.wav", "wb") as wav_file:
    wav_file.write(audio_data)
