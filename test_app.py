import os
import os.path as osp
import shutil
import tempfile

import numpy as np
import gradio as gr
import torch
import utils

script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

from audio_utils import ndarray_to_wav

import re
from infer import infer, latest_version, get_net_g, infer_multilang
import numpy as np
from config import config

# from concurrent.futures import ProcessPoolExecutor

import time

datasets_dir = "./Data"
inference_datasets_dir = "./Data/inference"
train_datasets_dir = "./Data/models"

os.makedirs(datasets_dir, exist_ok=True)
os.makedirs(train_datasets_dir, exist_ok=True)
os.system("cp -r Data/config.json {}".format(train_datasets_dir))
from tools.sentence import split_by_language

torch.manual_seed(114514)
sdp_ratio = 0.2
noise_scale = 0.6
noise_scale_w = 0.8
length_scale = 15
from process_datasets import gen_filelist_process

# speaker = "三月七"

training_done_count = 0
inference_done_count = 0

# training_threadpool = ProcessPoolExecutor(max_workers=1)

DEFAULT_RESOURCE_MODEL_ID = "damo/speech_ptts_autolabel_16k"
BASE_MODEL_ID = "damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k"
HOT_MODELS = [
    "\N{fire}萝莉童声",
    "luyao2",
]


PRETRAINED_MODEL_ID = {
    "\N{fire}萝莉童声": "凝光_ZH",  # id 20
    "luyao2": "luyao2",
}


MY_MODELS = ["\N{bust in silhouette}测试模型"]


print("Numpy config: {}".format(np.__config__.show()), flush=True)


def NAIVE_LOG(text, user_id):
    print("User: {},  log: {}".format(user_id, text), flush=True)


def upload_records(record, index, cur_workspace: str):
    if record is None:
        return 0
    print("record: {}".format(record), flush=True)
    sr, data = record
    #  wavfile.write(osp.join(cur_workspace, "raw_" + str(index) + '.wav'), sr, data)

    # two channels
    if len(data.shape) > 1:
        single_data = data[:, 0]
    else:
        single_data = data

    target_sr = 44100
    wav = ndarray_to_wav(single_data, sr, target_sr)
    file = osp.join(cur_workspace, str(index) + ".wav")
    with open(file, "wb") as f:
        f.write(wav)
    return 1


device = config.webui_config.device
hps = utils.get_hparams_from_file(config.webui_config.config_path)
version = hps.version if hasattr(hps, "version") else latest_version

net_g = get_net_g(
    model_path=config.webui_config.model, version=version, device=device, hps=hps
)
default_model = config.webui_config.model


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


def tts_fn(speaker, text, length):

    audio_list = []
    length = (100 - length) / 100
    print(text)
    split_text = [text[i : i + 500] for i in range(0, len(text), 500)]
    for text in split_text:
        for idx, slice in enumerate(text.split("|")):
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
                        length,
                        speaker,
                        lang_to_generate,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1

    audio_concat = np.concatenate(audio_list)
    return (hps.data.sampling_rate, audio_concat)


def launch_infer_task(user_models, text, session):
    global net_g, default_model
    if session["model_list"][user_models] != "凝光_ZH":
        speaker = "派蒙_ZH"
        # import pdb
        # pdb.set_trace()
        if default_model != os.path.join(
            inference_datasets_dir, f"{session['model_list'][user_models]}.pth"
        ):
            default_model = os.path.join(
                inference_datasets_dir, f"{session['model_list'][user_models]}.pth"
            )
            net_g = get_net_g(
                model_path=default_model, version=version, device=device, hps=hps
            )
            NAIVE_LOG("切换模型：", session["model_list"][user_models])
    else:
        if default_model != f"Data/inference/G_0.pth":
            default_model = os.path.join(
                inference_datasets_dir, f"Data/inference/G_0.pth"
            )
            net_g = get_net_g(
                model_path=config.webui_config.model,
                version=version,
                device=device,
                hps=hps,
            )
            NAIVE_LOG("切换模型：", session["model_list"][user_models])
        speaker = session["model_list"][user_models]
    wav_res = tts_fn(speaker, text, length_scale)

    NAIVE_LOG("Inference task succeed!", user_models)

    yield ["合成完毕, 请点击播放按钮试听: ", wav_res]


def fetch_uuid(session):
    my_models = session["my_models"]
    return gr.Radio.update(my_models)


def training_task():
    os.system("python train_ms.py")


def launch_training_task(user_name, session, *audio_lst):
    # import pdb
    # pdb.set_trace()
    if not user_name:
        yield ["Error: 请给定制的声音模型取个名字吧", None, session]
        return ["Error: 请给定制的声音模型取个名字吧", None, session]

    cur_data_path = os.path.join(datasets_dir, user_name)
    os.makedirs(cur_data_path, exist_ok=True)

    valid_wav_entries = 0
    # save record to .cache dir
    NAIVE_LOG("Uploading Audio files", cur_data_path)
    for num in range(len(audio_lst)):
        res = upload_records(audio_lst[num], num, cur_data_path)
        valid_wav_entries += res

    if valid_wav_entries < 10:
        yield ["Error: 录音文件数量不足，请至少录制10个以上的音频文件", None, session]
        return ["Error: 录音文件数量不足，请至少录制10个以上的音频文件", None, session]

    os.system("cp -r Data/config.json {}".format(train_datasets_dir))

    NAIVE_LOG("Audio files prepared", cur_data_path)
    gen_filelist_process(cur_data_path)
    os.system("python process_datasets.py")
    yield ["数据处理完成，请继续等待......", None, session]
    os.system(
        "python resample.py --in_dir {} --out_dir {}".format(
            cur_data_path, cur_data_path
        )
    )
    yield ["数据处理完成，请继续等待......", None, session]
    yield [
        "采样 resample.py --in_dir {} --out_dir {}".format(
            cur_data_path, cur_data_path
        ),
        None,
        session,
    ]
    os.system("python bert_gen.py")
    yield ["开始训练，请继续等待......", None, session]
    NAIVE_LOG("加载底模......", user_name)
    os.system("rm -rf {}/*.pth".format(train_datasets_dir))

    shutil.copy(
        osp.join(inference_datasets_dir, "D_0.pth"),
        osp.join(train_datasets_dir, "D_0.pth"),
    )
    shutil.copy(
        osp.join(inference_datasets_dir, "G_0.pth"),
        osp.join(train_datasets_dir, "G_0.pth"),
    )
    shutil.copy(
        osp.join(inference_datasets_dir, "DUR_0.pth"),
        osp.join(train_datasets_dir, "DUR_0.pth"),
    )

    future = training_threadpool.submit(training_task)
    start_time = time.time()

    while not future.done():
        is_processing = future.running()
        if is_processing:
            passed_time = int(time.time() - start_time)
            yield [
                "训练中, 预计需要1200秒, 请耐心等待, 当前已等待{}秒...".format(
                    passed_time
                ),
                None,
                session,
            ]

        time.sleep(1)

    all_files = os.listdir(train_datasets_dir)
    # Filter files starting with 'G' and ending with '.pth'
    g_files = [f for f in all_files if f.startswith("G") and f.endswith(".pth")]
    # Sort the files based on the integer value in the filename
    sorted_g_files = sorted(g_files, key=lambda x: int(re.findall(r"\d+", x)[0]))
    shutil.move(
        osp.join(train_datasets_dir, sorted_g_files[-1]),
        osp.join(inference_datasets_dir, f"{user_name}.pth"),
    )

    os.system("rm -rf {}/*".format(train_datasets_dir))

    NAIVE_LOG("Training task completed", user_name)
    session["my_models"].append("\N{bust in silhouette}" + user_name)
    session["model_list"]["\N{bust in silhouette}" + user_name] = user_name
    yield [
        "训练完成，当前已等待{}秒...请点击模型体验来体验合成的效果吧......".format(
            passed_time
        ),
        gr.Radio.update(
            choices=HOT_MODELS + session["my_models"],
            value="\N{bust in silhouette}" + user_name,
        ),
        session,
    ]
    return [
        "训练完成，当前已等待{}秒...请点击模型体验来体验合成的效果吧......".format(
            passed_time
        ),
        gr.Radio.update(
            choices=HOT_MODELS + session["my_models"],
            value="\N{bust in silhouette}" + user_name,
        ),
        session,
    ]


def choice_user_voice(choice, session):
    session["voice"] = choice
    return session


with gr.Blocks(css="#warning {color: red} .feedback {font-size: 24px}") as demo:
    gr.Markdown("## 个性化声音定制，感受另外一个自己")
    voice: str = ""
    model_list = PRETRAINED_MODEL_ID
    my_models = []
    session = gr.State(
        {
            # 'modelscope_uuid': 'guest',
            # 'modelscope_request_id': 'test',
            "voice": voice,
            "model_list": model_list,
            "my_models": my_models,
        }
    )

    with gr.Tabs():
        with gr.TabItem("\N{rocket}模型定制") as tab_train:
            helper = gr.Markdown(
                """
            \N{glowing star}**定制步骤**\N{glowing star}

            Step 0. 输入您定制的模型名\N{white up pointing index}，<font color="red">未输入无法使用定制功能 </font>

            Step 1. 录制音频\N{microphone}，点击下方音频录制并朗读左上角文字, 请至少录制10句话

            Step 2. 点击 **[开始训练]** \N{hourglass with flowing sand}，启动模型训练，等待约10分钟

            Step 3. 切换至 **[模型体验]** \N{speaker with three sound waves}，选择训练好的模型，感受效果

            \N{electric light bulb}**友情提示**\N{electric light bulb}

            \N{face savouring delicious food}  已支持英文合成

            \N{speech balloon}  朗读时请保持语速、情感一致

            \N{speaker with cancellation stroke}  尽量保持周围环境安静，避免噪音干扰

            \N{headphone}  建议佩戴耳机，以获得更好的录制效果
            """
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(label="给声音模型取个名字吧", value="")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_lst1 = [
                        gr.Audio(
                            source="microphone",
                            label="1. 清晨，阳光洒满小路，花儿含笑迎接新一天。",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="2. 树梢上鸟儿歌唱，旋律悠扬。公园中，松鼠忙碌，秋日收获。",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="3. 花园里，蝴蝶轻舞，狗狗追逐着飘落的叶子，欢乐满溢。",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="4. 河边，小鸭戏水，水波荡漾，映出欢快身影。",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="5. 蓝天下，花儿开放，绽放生机，小猫追逐彩色的梦。",
                        ),
                    ]
                with gr.Column(scale=1):
                    audio_lst2 = [
                        gr.Audio(
                            source="microphone",
                            label="6. 林间小径，草地上，松鼠跳跃，觅得果实分享喜悦。",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="7. 海边，螃蟹踏着轻快步伐，留下一串足迹。",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="8. 夕阳西下，蝴蝶飞舞，小兔听见晚风诉说着故事。",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="9. 星空闪烁，小狐狸仰望，许下心中愿望。",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="10. 早市上，小猪噜噜噜，探寻新鲜蔬果的香气。",
                        ),
                    ]
                with gr.Column(scale=1):
                    audio_lst3 = [
                        gr.Audio(
                            source="microphone",
                            label="11. Dawn breaks, and the path is bathed in sunlight, flowers smiling at the day.",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="12. By the river, ducklings splash, rippling water reflects their joy.",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="13. On the hill, sheep graze leisurely, a cool breeze whispers.",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="14. Under the blue sky, a rainbow stretches, a kitten chases colorful dreams.",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="15. In the woods, squirrels leap, sharing the joy of found fruits.",
                        ),
                    ]
                with gr.Column(scale=1):
                    audio_lst4 = [
                        gr.Audio(
                            source="microphone",
                            label="16. At the beach, crabs walk briskly, leaving a trail of footprints.",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="17. As the sun sets, a bunny hears stories whispered by the evening wind.",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="18. Stars twinkle, a fox gazes up, making a secret wish.",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="19. In the morning market, a piglet sniffs, seeking the scent of fresh produce.",
                        ),
                        gr.Audio(
                            source="microphone",
                            label="20. On the meadow, butterflies flutter, flowers bloom, bursting with life.",
                        ),
                    ]
            # One task at a time
            train_progress = gr.Textbox(
                label="训练进度", value="当前无训练任务", interactive=False
            )
            with gr.Row():
                training_button = gr.Button("开始训练")

        with gr.TabItem("\N{party popper}模型体验") as tab_infer:
            uuid_txt = gr.Text(label="modelscope_uuid", visible=False)
            reqid_txt = gr.Text(label="modelscope_request_id", visible=False)
            dbg_output = gr.Textbox(visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="合成内容",
                        value="清晨，阳光洒满小路，花儿含笑迎接新一天。",
                        max_lines=3,
                    )
                    #  hot_models = gr.Radio(label="热门模型", choices=HOT_MODELS, type="value")
                    user_models = gr.Radio(
                        label="模型选择",
                        choices=HOT_MODELS,
                        type="value",
                        value=HOT_MODELS[0],
                    )
                    refresh_button = gr.Button("刷新模型列表")
                #  voice_name = gr.Label(label="当前选择模型", value='')
                with gr.Column(scale=1):
                    infer_progress = gr.Textbox(
                        label="合成进度", value="当前无合成任务", interactive=False
                    )
                    helper2 = gr.Markdown(
                        """
                    \N{bell}**温馨提示**:
                    点击 **[刷新模型列表]** 拉取已定制的模型, 首次合成会下载模型, 请耐心等待

                    \N{police cars revolving light}**注意**:
                    本录音由AI生成, 禁止用于非法用途
                    """
                    )
                    audio_output = gr.Audio(label="合成结果")
                    inference_button = gr.Button("合成")

        #     #  hot_models.change(fn=choice_hot_voice, inputs=[hot_models, session], outputs=[voice_name, session])
        user_models.change(
            fn=choice_user_voice, inputs=[user_models, session], outputs=session
        )

        refresh_button.click(fetch_uuid, inputs=[session], outputs=[user_models])

    audio_list = audio_lst1 + audio_lst2 + audio_lst3 + audio_lst4

    # training_button.click(launch_training_task,
    #                       inputs=[exp_dir1, session] + audio_list,
    #                       outputs=[train_progress, user_models, session])

    # inference_button.click(launch_infer_task,
    #                        inputs=[user_models, text_input, session],
    #                        outputs=[infer_progress, audio_output])


# demo.launch(enable_queue=True,server_name="0.0.0.0", share=True)
# demo.queue(concurrency_count=511, max_size=1022).launch(share=True)
#  demo.queue(concurrency_count=10).launch(share=True)
demo.queue(concurrency_count=511, max_size=1022).launch(
    server_name="0.0.0.0", share=True
)
# demo.queue(concurrency_count=20).launch()
# demo.launch()

# shutil.rmtree(ROOT_WORKSPACE, ignore_errors=True)
