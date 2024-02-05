import os
import os.path as osp
import sys
import zipfile

import oss2
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.check_model import check_local_model_is_latest
from modelscope.utils.constant import ThirdParty


def oss_auth_check(key_id: str, secret: str, endpoint, bucket_name):
    oss_auth = oss2.Auth(key_id, secret)
    return oss2.Bucket(oss_auth, endpoint, bucket_name)


def download_model_from_oss(session, oss_bucket, root_workspace, bucket_user_models_folder,
                            bucket_hot_models_folder):
    zip_file: str
    model_basename: str
    bucket_folder: str
    bucket_file: str
    birth: str = session['model_type']
    cur_workspace: str = root_workspace
    user_id = session['modelscope_uuid']
    model: str = session['voice']
    model_list = session['model_list']
    print('download_model_from_oss model_list: ', model_list)
    print('download_model_from_oss model: ', model)
    print('download_model_from_oss birth: ', birth)

    if birth == 'user':
        # download custom model from OSS
        if model in model_list:
            model_basename = model_list[model]
        else:
            print('cannot find custom_model_id', flush=True)
            return None, False

        bucket_folder = bucket_user_models_folder + user_id + '/models/'
        cur_workspace = osp.join(cur_workspace, user_id)

    elif birth == 'hot':
        if model in model_list:
            model_basename = model_list[model]
        else:
            print('cannot find hot_model_id', flush=True)
            return None, False

        bucket_folder = bucket_hot_models_folder
        cur_workspace = osp.join(cur_workspace, 'hot')
    else:
        return None, False

    print('download_model_from_oss model_basename: ', model_basename)

    local_flag = True
    if osp.exists(model_basename) and osp.isdir(model_basename):
        local_flag = True
        local_model_dir = model_basename
    else:
        local_flag = True
        os.makedirs(cur_workspace, exist_ok=True)
        cur_model_workspace = osp.join(cur_workspace, 'models')
        os.makedirs(cur_model_workspace, exist_ok=True)
        print('download_model_from_oss cur_model_workspace: ',
              cur_model_workspace)

        zip_file = model_basename + '.zip'
        bucket_file = bucket_folder + zip_file
        local_model_file = osp.join(cur_model_workspace, zip_file)
        local_model_dir = osp.join(cur_model_workspace, model_basename)

        if not osp.exists(local_model_dir) or not osp.exists(local_model_file):
            os.makedirs(local_model_dir, exist_ok=True)

            if not osp.exists(local_model_file):
                exist = oss_bucket.object_exists(bucket_file)
                if exist:
                    oss_bucket.get_object_to_file(bucket_file,
                                                  local_model_file)
                else:
                    print('cannot find ' + bucket_file)
                    return None, local_flag

            with zipfile.ZipFile(local_model_file, 'r') as zipf:
                zipf.extractall(cur_model_workspace)
            print('download_model_from_oss cur_model_workspace DONE')

    return local_model_dir, local_flag


def download_and_unzip_resousrce(model, model_revision=None):
    if osp.exists(model):
        model_cache_dir = model if osp.isdir(model) else osp.dirname(model)
        check_local_model_is_latest(
            model_cache_dir,
            user_agent={ThirdParty.KEY: 'speech_tts_autolabel'})
    else:
        model_cache_dir = snapshot_download(
            model,
            revision=model_revision,
            user_agent={ThirdParty.KEY: 'speech_tts_autolabel'})
    if not osp.exists(model_cache_dir):
        raise ValueError(f'mdoel_cache_dir: {model_cache_dir} not exists')
    zip_file = osp.join(model_cache_dir, 'model.zip')
    if not osp.exists(zip_file):
        raise ValueError(f'zip_file: {zip_file} not exists')
    z = zipfile.ZipFile(zip_file)
    z.extractall(model_cache_dir)
    target_resource = osp.join(model_cache_dir, 'model')
    return target_resource


def download_model(model, model_revision=None):
    if osp.exists(model):
        model_cache_dir = model if osp.isdir(model) else osp.dirname(model)
        check_local_model_is_latest(
            model_cache_dir,
            user_agent={ThirdParty.KEY: 'speech_tts_autolabel'})
    else:
        model_cache_dir = snapshot_download(
            model,
            revision=model_revision,
            user_agent={ThirdParty.KEY: 'speech_tts_autolabel'})
        with zipfile.ZipFile(os.path.join(model_cache_dir, "resource.zip"), 'r') as zipf:
            zipf.extractall(model_cache_dir)
    return model_cache_dir


def upload_percentage(consumed_bytes, total_bytes):
    if total_bytes:
        rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
        print('\r{0}% '.format(rate), end='')
        sys.stdout.flush()


def zip_and_upload_to_oss(session, source, oss_bucket,
                          bucket_user_models_folder):
    pardir = osp.abspath(osp.dirname(source))
    dir_basename = osp.basename(source)
    target_zip_path = osp.join(pardir, dir_basename + '.zip')

    z = zipfile.ZipFile(file=osp.abspath(target_zip_path),
                        mode='w',
                        compression=zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(source):
        for filename in filenames:
            # 文件路径
            res_path = str(osp.join(dirpath, filename))
            # arcname的归档路径
            dirname = osp.basename(source)
            print('dirname: ', dirname)
            print('len(source)+1: ', len(source) + 1)
            rename_path = osp.join(dirname, res_path[len(source) + 1:])
            print('source: ', source)
            print('dirpath: ', dirpath)
            print('filename: ', filename)
            print('res_path:', res_path, ' rename_path: ', rename_path)
            z.write(res_path, arcname=rename_path)
    z.close()

    user_id = session['modelscope_uuid']
    bucket_cur_folder = bucket_user_models_folder + user_id + '/models/' + dir_basename + '.zip'
    print('upload to ', bucket_cur_folder)
    oss2.resumable_upload(oss_bucket,
                          bucket_cur_folder,
                          target_zip_path,
                          progress_callback=upload_percentage)
    print('upload zip to oss finish.')


def get_all_model_name_from_oss(session, oss_bucket,
                                bucket_user_models_folder):
    my_models = session['my_models']
    model_list = session['model_list']
    user_id = session['modelscope_uuid']
    bucket_folder = bucket_user_models_folder + user_id + '/models/'
    print('get_all_model_name_from_oss bucket_folder: ', bucket_folder)
    for filename in oss2.ObjectIteratorV2(oss_bucket, prefix=bucket_folder):
        print('filename: ', filename.key)
        file_basename = osp.basename(filename.key).split('.')[0]
        if file_basename is not None and len(file_basename) > 0:
            print('file_basename: ', file_basename)
            if "\N{bust in silhouette}" + file_basename not in my_models:
                my_models.append("\N{bust in silhouette}" + file_basename)
            model_list["\N{bust in silhouette}" + file_basename] = file_basename
    session['my_models'] = my_models
    session['model_list'] = model_list
    return session
