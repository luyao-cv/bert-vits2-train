
echo $PWD
export pwd_dir=$PWD
cd $pwd_dir
cp -r nltk_data ../

cd /opt/conda/envs/python35-paddle120-env/lib/python3.10/
rm -rf site-packages
wget https://bj.bcebos.com/v1/paddlenlp/models/community/luyao15/Bert-VITS/site-packages.tar.gz
tar -xzvf site-packages.tar.gz -C ./
rm -rf site-packages.tar.gz

cd $pwd_dir

pip install modelscope
pip uninstall gradio -y
pip install gradio==3.50.1 

wget https://bj.bcebos.com/v1/paddlenlp/models/community/luyao15/Bert-VITS/bert.tar.gz
tar -zxvf bert.tar.gz
rm -rf bert.tar.gz

wget https://bj.bcebos.com/v1/paddlenlp/models/community/luyao15/Bert-VITS/Data.tar.gz
tar -zxvf Data.tar.gz
rm -rf Data.tar.gz