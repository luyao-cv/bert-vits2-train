import os
import os.path as osp
import shutil
import tempfile
import uuid
from datetime import datetime
import threading
import time
import zipfile
import numpy as np
import gradio as gr
from scipy.io import wavfile
import torch

from audio_utils import ndarray_to_wav, wav_to_ndarray
from file_operation import (download_and_unzip_resousrce,
                            download_model,
                            download_model_from_oss,
                            get_all_model_name_from_oss, oss_auth_check,
                            zip_and_upload_to_oss)

from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)

training_threadpool = ProcessPoolExecutor(max_workers=1)
inference_threadpool = ProcessPoolExecutor(max_workers=5)

training_not_done = set()
inference_not_done = set()

training_done_count = 0
inference_done_count = 0

#  training_lock = threading.Lock()
#  inference_lock = threading.Lock()

DEFAULT_RESOURCE_MODEL_ID = 'damo/speech_ptts_autolabel_16k'
BASE_MODEL_ID = 'damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k'
HOT_MODELS = [
    "\N{fire}甜美女声",
    "\N{fire}知性女声",
    "\N{fire}萝莉童声",
    "\N{fire}顽皮童声",
    "\N{fire}诙谐男声",
    #  "\N{fire}财经男主播",
    #  "\N{fire}悬疑男解说",
    #  "\N{fire}青年男中音",
    "\N{fire}激昂男解说"
    ]

MY_MODELS = ["\N{bust in silhouette}测试模型"]

PRETRAINED_MODEL_ID = {
    '\N{fire}顽皮童声': 'jielidou',
    '\N{fire}萝莉童声': 'aibao',
    '\N{fire}知性女声': 'ruilin',
    '\N{fire}甜美女声': 'aimei',
    '\N{fire}财经男主播': 'aifei',
    '\N{fire}诙谐男声': 'aiming',
    '\N{fire}悬疑男解说': 'ailun',
    '\N{fire}激昂男解说': 'aifei',
    '\N{fire}青年男中音': 'kenny',
    '\N{bust in silhouette}测试模型': 'aiming',
    }

ROOT_WORKSPACE = tempfile.TemporaryDirectory().name
access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')
oss_bucket: object = None
bucket_name = 'isv-data'
bucket_folder = 'ics/MaaS/TTS/Data/'
bucket_hot_models_folder = 'ics/MaaS/TTS/Data/Hot/models/'
bucket_user_models_folder = 'ics/MaaS/TTS/Data/Custom/'
endpoint = 'oss-cn-hangzhou.aliyuncs.com'

print("Numpy config: {}".format(np.__config__.show()), flush=True)

os.makedirs(ROOT_WORKSPACE, exist_ok=True)
# oss_bucket = oss_auth_check(access_key_id, access_key_secret, endpoint,
#                             bucket_name)
if oss_bucket is None:
    print('oss auth failed.', flush=True)

BASE_MODEL_DIR = download_model(BASE_MODEL_ID, "v1.0.7")
print("Base model dir path: {}".format(BASE_MODEL_DIR), flush=True)

AUTO_LABEL_RESOURCE_DIR = download_and_unzip_resousrce(DEFAULT_RESOURCE_MODEL_ID, "v1.0.7")
print("AutoLabel resource dir path: {}".format(AUTO_LABEL_RESOURCE_DIR), flush=True)


def NAIVE_LOG(text, user_id):
    print("User: {},  log: {}".format(user_id, text), flush=True)

def increment_training_done_count():
    global training_done_count
    #  training_lock.acquire()
    training_done_count += 1
    #  training_lock.release()

def increment_inference_done_count():
    global inference_done_count
    #  inference_lock.acquire()
    inference_done_count += 1
    #  inference_lock.release()

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

    target_sr = 16000
    wav = ndarray_to_wav(single_data, sr, target_sr)
    file = osp.join(cur_workspace, str(index) + '.wav')
    with open(file, 'wb') as f:
        f.write(wav)
    return 1

def copy_importance_for_infer(src: str, dest: str):
    os.makedirs(dest, exist_ok=True)

    os.makedirs(osp.join(dest, 'tmp_am'), exist_ok=True)
    os.makedirs(osp.join(dest, 'tmp_am', 'ckpt'), exist_ok=True)
    # find the latest checkpoint
    checkpoints = os.listdir(osp.join(src, 'tmp_am', 'ckpt'))
    latest_ckpt = 0
    for checkpoint in checkpoints:
        if checkpoint.startswith('checkpoint'):
            steps = int(checkpoint.split('.')[0].split('_')[1])
            latest_ckpt = max(latest_ckpt, steps)

    lastest_ckpt_fn = 'checkpoint_{}.pth'.format(latest_ckpt)

    shutil.copy(osp.join(src, 'tmp_am', 'ckpt', lastest_ckpt_fn),
                    osp.join(dest, 'tmp_am', 'ckpt', lastest_ckpt_fn))
    shutil.copy(osp.join(src, 'tmp_am', 'config.yaml'),
                osp.join(dest, 'tmp_am', 'config.yaml'))

    os.makedirs(osp.join(dest, 'data'), exist_ok=True)
    os.makedirs(osp.join(dest, 'data', 'se'), exist_ok=True)
    shutil.copy(osp.join(src, 'data', 'se', 'se.npy'),
                osp.join(dest, 'data', 'se', 'se.npy'))

def infer_task(text, session):
    torch.cuda.empty_cache()
    try:
        print(session, flush=True)
        user_id = session.get('modelscope_uuid', 'guest')
        session["model_type"] = "hot" if session['voice'] in HOT_MODELS else "user"
        session = get_all_model_name_from_oss(session, oss_bucket, bucket_user_models_folder)
        model_dir, local_flag = download_model_from_oss(session, oss_bucket, ROOT_WORKSPACE,
                                                       bucket_user_models_folder,
                                                       bucket_hot_models_folder)
        if model_dir is None:
            NAIVE_LOG('model_dir cannot find', user_id)
            return False, None
        else:
            NAIVE_LOG('model_dir for inference: {}'.format(model_dir), user_id)

        output_file = tempfile.NamedTemporaryFile().name
        command = "python infer_script.py --text '{}' --model_dir {} --BASE_MODEL_DIR {} --local_flag {} --output_file {} --user_id {}".format(text, model_dir, BASE_MODEL_DIR, local_flag, output_file, user_id)
        print(command)
        os.system(command)

        wav = np.load(output_file + ".npy").tobytes()

        NAIVE_LOG('Inference done', user_id)
        increment_inference_done_count()
        return True, wav_to_ndarray(wav, 16000)
    except Exception as e:
        print("Inference Error: {}".format(e), flush=True)
        increment_inference_done_count()
        return False, None

def launch_infer_task(uuid_txt, user_model, text, session):
    session['modelscope_uuid'] = uuid_txt
    print("Infer task triggered fetch_uuid: {}".format(uuid_txt), flush=True)
    user_id = session['modelscope_uuid']
    before_queue_size = len(inference_not_done) - sum(1 for f in inference_not_done if f.done())
    before_done_count = inference_done_count

    user_id = session.get('modelscope_uuid', 'guest')
    session['voice'] = user_model
    NAIVE_LOG('Inference task vocie {}'.format(user_model), user_id)
    NAIVE_LOG('Inference task is launching', user_id)
    if text is not None and len(text) >= 300:
        text = text[:300]
        yield ["输入文本过长，单次合成限300字以内，已自动截断", None]

    future = inference_threadpool.submit(infer_task, text, session)
    inference_not_done.add(future)

    while not future.done():
        is_processing = future.running()
        if not is_processing:
            cur_done_count = inference_done_count
            to_wait = before_queue_size - (cur_done_count - before_done_count)
            yield ["排队等待资源中，前方还有{}个合成任务, 预计需要等待{}分钟...".format(to_wait, round(to_wait*0.3, 2)), None]
        else:
            yield ["合成中, 请耐心等待...", None]
        time.sleep(1)

    success, wav_res = future.result()
    if success:
        NAIVE_LOG('Inference task succeed!', user_id)
        yield ["合成完毕, 请点击播放按钮试听: ", wav_res]
    else:
        NAIVE_LOG('Inference task Error!', user_id)
        yield ["抱歉合成出错，请再试一次", None]

def training_task(name, cur_data_path, cur_model_path, session):
    torch.cuda.empty_cache()
    user_id = session.get('modelscope_uuid', 'guest')
    try:
        command = "python train_script.py --data_path {} --model_path {} --user_id {} --AUTO_LABEL_RESOURCE_DIR {}".format(cur_data_path, cur_model_path, user_id, AUTO_LABEL_RESOURCE_DIR)
        print(command)
        res = os.system(command)

        session = get_all_model_name_from_oss(session, oss_bucket, bucket_user_models_folder)
        if "\N{bust in silhouette}" + name not in session['my_models']:
            session['my_models'].append("\N{bust in silhouette}" + name)
        if "\N{bust in silhouette}" + name not in session['model_list']:
            session['model_list']["\N{bust in silhouette}" + name] = name
        NAIVE_LOG('session model_list: '.format(model_list), user_id)
        session['voice'] = name

        pretrain_work_dir = os.path.join(cur_model_path, 'pretrain')
        infer_work_dir = os.path.join(cur_model_path)

        # clean a dir for zip and upload
        copy_importance_for_infer(pretrain_work_dir, infer_work_dir)
        shutil.rmtree(pretrain_work_dir, ignore_errors=True)
        NAIVE_LOG('Moving trained model', user_id)
        # zip <workspace>/<uuid>/models/202303142248(cur_model_path) and upload to oss
        zip_and_upload_to_oss(session, cur_model_path, oss_bucket,
                              bucket_user_models_folder)
        NAIVE_LOG('Uploaded trained model', user_id)
        increment_training_done_count()
        NAIVE_LOG('Training task succeed', user_id)
        return True, '训练完成, 请切换至[模型体验]标签体验模型效果', session
    except Exception as e:
        increment_training_done_count()
        NAIVE_LOG('Training task failed!', user_id)
        return False, "训练失败: {}".format(e), session

def launch_training_task(uuid_txt, session, *audio_lst):
    session['modelscope_uuid'] = uuid_txt
    print("Training task triggered fetch_uuid: {}".format(uuid_txt), flush=True)

    cur_workspace: str = ROOT_WORKSPACE
    user_id = session.get('modelscope_uuid', 'guest')
    model_list = session['model_list']

    if user_id == "guest" or user_id == "":
        print('user_id is empty', flush=True)
        yield ["Error: 训练功能仅支持登录用户使用", None, session]
        return ["Error: 训练功能仅支持登录用户使用", None, session]

    name = datetime.now().strftime('%Y_%m%d_%H%M%S')
    user_path = osp.join(cur_workspace, user_id)  # your workspace
    user_data = osp.join(user_path, 'data')
    user_models = osp.join(user_path, 'models')
    cur_data_path = osp.join(user_data, name)
    cur_model_path = osp.join(user_models, name)
    os.makedirs(user_path, exist_ok=True)  # <workspace>/<uuid>
    os.makedirs(user_data, exist_ok=True)  # <workspace>/<uuid>/data
    os.makedirs(cur_data_path,
                exist_ok=True)  # <workspace>/<uuid>/data/202303142248
    os.makedirs(user_models, exist_ok=True)  # <workspace>/<uuid>/models
    os.makedirs(cur_model_path,
                exist_ok=True)  # <workspace>/<uuid>/models/202303142248

    valid_wav_entries = 0
    # save record to .cache dir
    NAIVE_LOG("Uploading Audio files", user_id)
    for num in range(len(audio_lst)):
        res = upload_records(audio_lst[num], num, cur_data_path)
        valid_wav_entries += res

    if valid_wav_entries < 10:
        yield ["Error: 录音文件数量不足，请至少录制10个以上的音频文件", None, session]
        return ["Error: 录音文件数量不足，请至少录制10个以上的音频文件", None, session]

    NAIVE_LOG("Audio files prepared", user_id)
    before_queue_size = len(training_not_done) - sum(1 for f in training_not_done if f.done())
    before_done_count = training_done_count

    NAIVE_LOG("Launching training task..., not done tasks: {}".format(before_queue_size), user_id)
    future = training_threadpool.submit(training_task, name, cur_data_path, cur_model_path, session)
    training_not_done.add(future)

    start_time = time.time()
    while not future.done():
        is_processing = future.running()
        if not is_processing:
            cur_done_count = training_done_count
            to_wait = before_queue_size - (cur_done_count - before_done_count)
            start_time = time.time()
            yield ["排队等待资源中，您前面还有{}个用户, 预计需要等待{}分钟...".format(to_wait, to_wait*6), None, session]
        else:
            passed_time = int(time.time() - start_time)
            yield ["训练中, 预计需要600秒, 请耐心等待, 当前已等待{}秒...".format(passed_time), None, session]
        time.sleep(1)

    success, msg, session = future.result()
    if success:
        NAIVE_LOG("Training task completed", user_id)
        yield [msg, gr.Radio.update(choices=HOT_MODELS + session['my_models'], value="\N{bust in silhouette}" + name), session]
        return [msg, gr.Radio.update(choices=HOT_MODELS + session['my_models'], value="\N{bust in silhouette}" + name), session]
    else:
        NAIVE_LOG("Training task failed", user_id)
        yield [msg, None, session]
        return [msg, None, session]


def choice_hot_voice(choice, session):
    session['voice'] = choice
    return [choice, session]

def choice_user_voice(choice, session):
    session['voice'] = choice
    return session

def fetch_uuid(uuid_txt, s):
    s['modelscope_uuid'] = uuid_txt
    print("Triggered fetch_uuid: {}".format(uuid_txt), flush=True)
    user_id = s['modelscope_uuid']
    s = get_all_model_name_from_oss(s, oss_bucket, bucket_user_models_folder)
    my_models = s['my_models']
    return user_id, s, gr.Radio.update(choices=HOT_MODELS + my_models)

def fetch_requestid(reqid_txt, s):
    s['modelscope_request_id'] = 'fakerequestid'
    return reqid_txt + s['modelscope_request_id'], s

with gr.Blocks(css="#warning {color: red} .feedback {font-size: 24px}") as demo:
    gr.Markdown("## 个性化声音定制，感受另外一个自己")
    voice: str = ''
    model_list = PRETRAINED_MODEL_ID
    my_models = []
    session = gr.State({
        'modelscope_uuid': 'guest',
        'modelscope_request_id': 'test',
        'voice': voice,
        'model_list': model_list,
        'my_models': my_models
    })

    with gr.Tabs():
        with gr.TabItem("\N{rocket}模型定制") as tab_train:
            helper = gr.Markdown(
            """
            \N{glowing star}**定制步骤**\N{glowing star}

            Step 0. 登陆ModelScope账号\N{white up pointing index}，<font color="red">未登陆无法使用定制功能 </font>

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
                with gr.Column(scale=1):
                    audio_lst1 = [
                        gr.Audio(source="microphone", label="1. 希望我们大家都能像他一样"),
                        gr.Audio(source="microphone", label="2. 不行, 他想了一下, 我不能这样对国王说, 这是在撒谎"),
                        gr.Audio(source="microphone", label="3. 但他们非常和气地问她说, 你叫什么名字"),
                        gr.Audio(source="microphone", label="4. 鸭子心想, 我必须去拿回我的软糖豆"),
                        gr.Audio(source="microphone", label="5. 小朋友, 你们不要再欺负它了"),
                    ]
                with gr.Column(scale=1):
                    audio_lst2 = [
                        gr.Audio(source="microphone", label="6. 可是, 小黄鸭并不怕他们"),
                        gr.Audio(source="microphone", label="7. 然后, 他们一起走了很长一段时间"),
                        gr.Audio(source="microphone", label="8. 突然, 墙壁后面传来一阵声音"),
                        gr.Audio(source="microphone", label="9. 结果盘子掉在地上, 打得粉碎"),
                        gr.Audio(source="microphone", label="10. 四个小伙伴很开心, 一起感谢小松鼠的帮助"),
                    ]
                with gr.Column(scale=1):
                    audio_lst3 = [
                        gr.Audio(source="microphone", label="11. 不过, 当他看到拇指姑娘的时候, 他马上就变得高兴起来"),
                        gr.Audio(source="microphone", label="12. 从此以后, 他过上了幸福的生活"),
                        gr.Audio(source="microphone", label="13. 老山羊最后伤心地, 哭着走了出去"),
                        gr.Audio(source="microphone", label="14. 而且准备一直找下去, 直到他走不动为止"),
                        gr.Audio(source="microphone", label="15. 海马先生轻轻游过大海"),
                    ]
                with gr.Column(scale=1):
                    audio_lst4 = [
                        gr.Audio(source="microphone", label="16. 一起高高兴兴地, 回到了他们的爸爸妈妈身边"),
                        gr.Audio(source="microphone", label="17. 艾丽莎很小不能去上学, 但她有一个非常贵重精美的画册"),
                        gr.Audio(source="microphone", label="18. 狮子还是够不着, 它叫来了狐狸"),
                        gr.Audio(source="microphone", label="19. 姑娘坐到国王的马车上, 和国王一起回到宫中"),
                        gr.Audio(source="microphone", label="20. 温妮大叫了起来, 现在我们该怎么回家呀"),
                    ]
            # One task at a time
            train_progress = gr.Textbox(label="训练进度", value="当前无训练任务", interactive=False)
            with gr.Row():
                training_button = gr.Button("开始训练")

        with gr.TabItem("\N{party popper}模型体验") as tab_infer:
            uuid_txt = gr.Text(label="modelscope_uuid", visible=False)
            reqid_txt = gr.Text(label="modelscope_request_id", visible=False)
            dbg_output = gr.Textbox(visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(label="合成内容", value="这周天气真不错, 叫上朋友一起爬山吧", max_lines=3)
                    #  hot_models = gr.Radio(label="热门模型", choices=HOT_MODELS, type="value")
                    user_models = gr.Radio(label="模型选择", choices=HOT_MODELS, type="value", value=HOT_MODELS[0])
                    refresh_button = gr.Button("刷新模型列表")
                #  voice_name = gr.Label(label="当前选择模型", value='')
                with gr.Column(scale=1):
                    infer_progress = gr.Textbox(label="合成进度", value="当前无合成任务", interactive=False)
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

        #  hot_models.change(fn=choice_hot_voice, inputs=[hot_models, session], outputs=[voice_name, session])
        user_models.change(fn=choice_user_voice, inputs=[user_models, session], outputs=session)

    #  tab_infer.select(fetch_uuid, inputs=[uuid_txt, session], outputs=[dbg_output, session, user_models])
    refresh_button.click(fetch_uuid, inputs=[uuid_txt, session], outputs=[dbg_output, session, user_models])
    #  audio_lst1[0].change(fetch_uuid, inputs=[uuid_txt, session], outputs=[dbg_output, session, user_models])

    audio_list = audio_lst1 + audio_lst2 + audio_lst3 + audio_lst4
    training_button.click(launch_training_task,
                          inputs=[uuid_txt, session] + audio_list,
                          outputs=[train_progress, user_models, session])

    inference_button.click(launch_infer_task,
                           inputs=[uuid_txt, user_models, text_input, session],
                           outputs=[infer_progress, audio_output])


#  demo.queue(concurrency_count=10).launch(share=True)
#  demo.queue(concurrency_count=2).launch(server_name="0.0.0.0")
demo.queue(concurrency_count=20).launch()
#  demo.launch()

shutil.rmtree(ROOT_WORKSPACE, ignore_errors=True)