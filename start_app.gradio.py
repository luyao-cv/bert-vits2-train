import os
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)
print(script_path)
os.system("pwd")

try:
    import torch
except:
    os.system('sh prepare_env.sh')
    print("downloads env list")

try:
    os.system("python test_app.py")

except: 
    os.system("python test_app.py")



