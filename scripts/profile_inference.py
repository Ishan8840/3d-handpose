"""Profile one explicitly supplied command; sample device-wide memory, not allocator peak."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();command=a.command
if command and command[0]=='--':command=command[1:]
if not command:raise ValueError('Supply a command after --')
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);memory=[];start=time.perf_counter()
with out.with_suffix('.log').open('w') as log:
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    while process.poll() is None:
        try:
            value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=3)
            memory.append([int(line) for line in value.strip().splitlines()])
        except (FileNotFoundError,subprocess.SubprocessError,ValueError):pass
        time.sleep(.5)
    code=process.returncode
out.write_text(json.dumps({'command':command,'elapsed_seconds':time.perf_counter()-start,'exit_code':code,'peak_sampled_device_memory_MiB':[max(row[i] for row in memory) for i in range(len(memory[0]))] if memory else None,'memory_scope':'device-wide nvidia-smi sampled every ~0.5 seconds; includes other CUDA allocations, not exact process peak'},indent=2))
if code:raise SystemExit(code)
