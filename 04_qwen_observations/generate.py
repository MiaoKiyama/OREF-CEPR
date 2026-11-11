"""Generate image-only Q attributes with the archived prompt and parser."""
import argparse
import json
import hashlib
from pathlib import Path
from PIL import Image
from generator_reference import normalize

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='Qwen/Qwen3.5-9B');p.add_argument('--device',default='cuda:0')
    p.add_argument('--limit',type=int);p.add_argument('--check-inputs',action='store_true')
    a=p.parse_args();rows=json.loads(a.requests.read_text())
    if a.limit:rows=rows[:a.limit]
    for row in rows:
        for key,size in [('context',512),('target',256)]:
            if key+'_sha256' in row and hashlib.sha256((a.requests.parent/row[key]).read_bytes()).hexdigest()!=row[key+'_sha256']:
                raise ValueError('Prepared view checksum mismatch')
            with Image.open(a.requests.parent/row[key]) as im:
                if im.size!=(size,size) or im.mode!='RGB':raise ValueError('Invalid prepared view')
    if a.check_inputs:
        print(json.dumps(dict(requests=len(rows),status='passed')));return
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    torch.manual_seed(20261012)
    processor=AutoProcessor.from_pretrained(a.model)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').to(a.device).eval()
    prompt=json.loads(Path(__file__).with_name('prompt.json').read_text())
    prompt_sha=hashlib.sha256(json.dumps(prompt,sort_keys=True).encode()).hexdigest()
    a.output.mkdir(parents=True,exist_ok=True)
    for row in rows:
        destination=a.output/f'{row["scene"]}__grid{row["roi"]:02d}.json'
        if destination.exists():
            old=json.loads(destination.read_text())
            if old['request']!=row or old['model']!=a.model or old.get('prompt_sha256')!=prompt_sha:
                raise ValueError('Existing observation belongs to different input/model/prompt')
            continue
        images=[Image.open(a.requests.parent/row[key]).convert('RGB') for key in ('context','target')]
        try:
            content=[{'type':'image','image':im} for im in images]+[{'type':'text','text':prompt['user']}]
            messages=[{'role':'system','content':[{'type':'text','text':prompt['system']}]},
                      {'role':'user','content':content}]
            rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=processor(text=[rendered],images=images,return_tensors='pt').to(a.device)
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=384,do_sample=False,
                                          pad_token_id=model.config.text_config.eos_token_id)
            tokens=generated[:,inputs['input_ids'].shape[1]:]
            eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
            ended=bool(tokens.shape[1] and int(tokens[0,-1]) in eos)
            raw=processor.batch_decode(tokens,skip_special_tokens=True)[0]
            value=dict(request=row,raw_text=raw,model=a.model,prompt_sha256=prompt_sha,
                       **normalize(raw,ended,tokens.shape[1]>=384 and not ended))
            tmp=destination.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(destination)
            print(destination.name,flush=True)
        finally:
            for im in images:im.close()

if __name__=='__main__':main()
