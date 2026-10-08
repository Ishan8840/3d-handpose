from huggingface_hub import HfApi
api=HfApi()
for repo in ['acerobotics2025/ACE-Ego-Hand','alibaba-pai/Wan2.2-Fun-5B-Control']:
    try:
        info=api.model_info(repo,files_metadata=True)
        print(repo,info.sha,[(f.rfilename,round((f.size or 0)/1e9,3)) for f in info.siblings],flush=True)
    except Exception as e: print(repo,type(e).__name__,str(e)[:300])
