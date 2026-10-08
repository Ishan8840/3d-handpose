"""Record hardware; installation is explicit in README to isolate model environments."""
import json
import platform
import shutil
import subprocess

def main():
    info={'python':platform.python_version(),'platform':platform.platform()}
    try:
        import torch
        info.update(torch=torch.__version__,cuda=torch.cuda.is_available())
        if torch.cuda.is_available():
            info.update(gpu=torch.cuda.get_device_name(),gpu_free_total_bytes=torch.cuda.mem_get_info())
    except ImportError:
        info['torch']='not installed'
    if shutil.which('nvidia-smi'):
        info['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True)
    print(json.dumps(info,indent=2))
if __name__=='__main__': main()
