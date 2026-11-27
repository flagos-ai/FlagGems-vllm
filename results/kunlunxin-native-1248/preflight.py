"""Native vLLM0.32 compatibility/oracle preflight; never a timing benchmark."""
import hashlib
import importlib
import importlib.metadata as md
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

os.environ['VLLM_PLUGINS']=''
os.environ['USE_FLAGGEMS']='0'
op=sys.argv[1]
assert op in ['grammar','penalties','bincount']
report={'op':op,'native_only':True,'timing_run':False,'cases':[]}
def save():
    Path('/results/preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

try:
    import torch
    import triton
    assert torch.cuda.device_count()==1,torch.cuda.device_count()
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_properties(0).pci_bus_id==132
    assert str(Path(triton.__file__)).startswith('/opt/flagtree/triton/')
    report.update({'torch':torch.__version__,'triton':triton.__version__,'pci_bus_id':132,'triton_origin':triton.__file__})
    packages={}
    for name in ['flagcx','flagtree','vllm','torch','flag-gems','flaggems_vllm']:
        try:packages[name]=md.version(name)
        except md.PackageNotFoundError:packages[name]=None
    report['package_metadata']=packages
    compiler=Path(triton.__file__).parent/'_C/libtriton.so'
    report['compiler_sha256']=hashlib.sha256(compiler.read_bytes()).hexdigest()
    name='vllm.v1.worker.gpu.structured_outputs' if op=='grammar' else 'vllm.v1.worker.gpu.sample.penalties'
    native=importlib.import_module(name)
    assert not getattr(native,'_vllm_fl_kunlunxin_penalties_provider',False)
    assert not any(n.startswith(('flaggems_vllm','vllm_fl')) for n in sys.modules)
    file=Path(native.__file__)
    expected={'grammar':'9e212ac51c603e9b95bbc8dceec4510d96b764e49eafa7c10a581ac1e6535297','penalties':'a5e99c71e6ac50ef025b8a7b0b903a1d2679738d8319af9c98bb200389388b29','bincount':'a5e99c71e6ac50ef025b8a7b0b903a1d2679738d8319af9c98bb200389388b29'}[op]
    assert hashlib.sha256(file.read_bytes()).hexdigest()==expected
    original=getattr(native,{'grammar':'_apply_grammar_bitmask_kernel','penalties':'apply_penalties','bincount':'bincount'}[op])
    fn=original.fn if op=='grammar' else original
    assert fn.__module__.startswith('vllm.'),fn.__module__
    report['native_callable']={'module':fn.__module__,'source':str(file),'source_sha256':expected,'name':fn.__name__}
    save()
    device='cuda:0'
    if op=='grammar':
        rows,vocab=1,167
        bitmask=torch.randint(-(1<<31),1<<31,(rows,(vocab+31)//32),dtype=torch.int32,generator=torch.Generator().manual_seed(873))
        tokens=torch.arange(vocab)
        expected_out=torch.zeros((rows,vocab))
        expected_out[(bitmask[:,tokens//32]>>(tokens%32))&1==0]=-torch.inf
        mask=bitmask.to(device)
        mapping=torch.arange(rows,dtype=torch.int32,device=device)
        offsets=torch.arange(rows+1,dtype=torch.int32,device=device)
        actual=torch.zeros(rows,vocab,device=device)
        original[(rows,triton.cdiv(vocab,8192))](actual,actual.stride(0),mapping,offsets,mask,mask.stride(0),vocab,MASK_STRIDE=1,BLOCK_SIZE=8192)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(),expected_out,rtol=0,atol=0)
        report['cases'].append({'rows':rows,'vocab':vocab,'seed':873,'BLOCK_SIZE':8192,'oracle':'original exact CPU mask','passed':True})
    elif op=='penalties':
        rows,vocab=1,1031
        cpu=torch.randn(rows,vocab,generator=torch.Generator().manual_seed(873))
        expected_out=torch.where(cpu>0,cpu/1.001,cpu*1.001)-.01-.01
        indices=torch.arange(rows,dtype=torch.int32,device=device)
        zero=torch.zeros_like(indices)
        repetition=torch.full((rows,),1.001,device=device)
        frequency=torch.full((rows,),.01,device=device)
        packed=torch.zeros(rows,triton.cdiv(vocab,32),dtype=torch.int32,device=device)
        counts=torch.ones(rows,vocab,dtype=torch.int32,device=device)
        actual=cpu.to(device)
        original(actual,indices,zero,zero,repetition,frequency,frequency.clone(),packed,counts)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(),expected_out,rtol=1e-5,atol=1e-5)
        report['cases'].append({'rows':rows,'vocab':vocab,'seed':873,'native_BLOCK_SIZE':8192,'rtol':1e-5,'atol':1e-5,'passed':True})
    else:
        # The four original PR1248 UT fixture contracts, unchanged. No retries.
        for vocab,max_prefill_len in [(167,185),(151939,185),(151939,2048),(167,0)]:
            ids=torch.randint(vocab,(4,max(190,max_prefill_len)),dtype=torch.int32,generator=torch.Generator().manual_seed(147))
            ids[:,:4]=torch.tensor([31,31,12,vocab-1])
            mapping=torch.tensor([3,1],dtype=torch.int64)
            prompt_len=torch.tensor([20,135,0,72],dtype=torch.int32)
            prefill_len=torch.tensor([44,185,0,175],dtype=torch.int32)
            if max_prefill_len==2048:
                prompt_len[:]=513
                prefill_len[:]=max_prefill_len
                ids[:,520::133]=31
                ids[:,670::217]=vocab-1
            elif max_prefill_len==0:
                prompt_len.zero_();prefill_len.zero_()
            words=triton.cdiv(vocab,32)
            packed=torch.full((4,words+2),7,dtype=torch.int32)
            counts=torch.full((4,vocab+5),7,dtype=torch.int32)
            expected_packed=packed.clone();expected_counts=counts.clone()
            for row in mapping.tolist():
                expected_packed[row,:words]=0;expected_counts[row,:vocab]=0
                for position,token in enumerate(ids[row,:prefill_len[row]].tolist()):
                    if position<prompt_len[row]:
                        bit=1<<(token%32)
                        expected_packed[row,token//32]|=bit if bit<1<<31 else -(1<<31)
                    else:expected_counts[row,token]+=1
            actual_packed=packed.to(device);actual_counts=counts.to(device)
            report['executing_contract']={'vocab':vocab,'max_prefill_len':max_prefill_len,'seed':147,'native_BLOCK_SIZE':1024}
            save()
            original(mapping.to(device),ids.to(device),prompt_len.to(device),prefill_len.to(device),actual_packed[:,:words],actual_counts[:,:vocab],max_prefill_len)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual_packed.cpu(),expected_packed,rtol=0,atol=0)
            torch.testing.assert_close(actual_counts.cpu(),expected_counts,rtol=0,atol=0)
            report['cases'].append({**report['executing_contract'],'passed':True,'rtol':0,'atol':0})
    assert not any(n.startswith(('flaggems_vllm','vllm_fl')) for n in sys.modules)
    report['passed']=True
except Exception as exc:
    report['passed']=False
    report['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    traceback.print_exc()
finally:save()
sys.exit(0 if report['passed'] else 1)
