#!/usr/bin/env python3
"""CPU sample() spectrum, preregistered consumer-v2; never modifies generator."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import stbn_spectral_check as raw

REGISTRATION = "ddbe3d5"
PROTOCOL = "tools/testdata/stbn/protocol-consumer-v2.json"
RAW_RESULT = "docs/results/STBN-cpu-spectrum-M5Max-2026-10-07.json"


def hash32(value):
    value &= 0xffffffff
    value ^= value >> 16; value = value*0x7feb352d & 0xffffffff
    value ^= value >> 15; value = value*0x846ca68b & 0xffffffff
    value ^= value >> 16
    return value


def sample_float32(ranks,seed,block):
    n = int(np.prod(ranks.shape[1:]))
    base = ((ranks.astype(np.float64)+0.5)/n).astype(np.float32)
    out = np.empty_like(base)
    for d in range(ranks.shape[0]):
        bits = hash32((seed^0xa511e9b3)^hash32(0)^hash32(0x9e3779b9)^
                      hash32(block+0x632be5ab)^hash32(d+0x85157af5))
        rotation = np.float32(bits>>8)*np.float32(1.0/16777216.0)
        total = np.add(base[d],rotation,dtype=np.float32)
        out[d] = np.subtract(total,np.floor(total),dtype=np.float32)
    return out


def controls(seed,block,shape):
    dims,frames,height,width=shape;n=frames*height*width
    permutation,white=np.empty(shape),np.empty(shape)
    for d in range(dims):
        a=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,1398030926,d,0,block])))
        b=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,1398030926,d,1,block])))
        permutation[d]=((a.permutation(n)+0.5)/n).reshape(frames,height,width)
        white[d]=b.random((frames,height,width))
    return {"uniform_permutation":permutation,"iid_uniform":white}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks',type=Path,required=True)
    ap.add_argument('--consumer',type=Path,required=True)
    ap.add_argument('--snapshot',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('result must be fresh; earlier FAIL evidence is never overwritten')
    repo=Path(__file__).resolve().parents[1]
    protocol_bytes=(repo/PROTOCOL).read_bytes()
    if protocol_bytes!=subprocess.check_output(['git','show',f'{REGISTRATION}:{PROTOCOL}'],cwd=repo):
        raise RuntimeError('consumer protocol changed after preregistration')
    protocol=json.loads(protocol_bytes)
    raw_bytes=(repo/RAW_RESULT).read_bytes()
    if raw.sha256(raw_bytes)!=protocol['rank_result_sha256']:raise RuntimeError('raw S1 result changed')
    previous=json.loads(raw_bytes)
    source_hashes={}
    for file in raw.SOURCE_FILES:
        expected=subprocess.check_output(['git','show',protocol['source_commit']+':'+file],cwd=repo)
        actual=(args.snapshot/file).read_bytes()
        if expected!=actual:raise RuntimeError('consumer snapshot mismatch: '+file)
        source_hashes[file]=raw.sha256(actual)
    old_source=subprocess.check_output(['git','show',protocol['rank_source_commit']+':'+raw.SOURCE_FILES[0]],cwd=repo)
    new_source=(args.snapshot/raw.SOURCE_FILES[0]).read_bytes()
    if old_source.split(b'StbnMask generateStbn(',1)[1]!=new_source.split(b'StbnMask generateStbn(',1)[1]:
        raise RuntimeError('generator body changed; original ranks cannot be reused')
    metadata_bytes=(args.consumer/'consumer.json').read_bytes()
    metadata=json.loads(metadata_bytes)
    if metadata['generator_version']!=protocol['generator_version']:raise RuntimeError('consumer version mismatch')
    entries={(e['seed'],e['block']):e for e in metadata['runs']}
    if set(entries)!={(s,b) for s in protocol['seeds'] for b in protocol['blocks']}:raise RuntimeError('consumer corpus mismatch')
    cfg=protocol['config'];shape=(cfg['dimensions'],cfg['frames'],cfg['height'],cfg['width']);n=int(np.prod(shape[1:]))
    runs=[]; sequence_diagnostics=[]
    for seed in protocol['seeds']:
        old=next(r for r in previous['runs'] if r['seed']==seed)
        rank_bytes=(args.ranks/old['file']).read_bytes()
        if raw.sha256(rank_bytes)!=old['ranks_sha256']:raise RuntimeError('raw rank table changed')
        ranks=np.frombuffer(rank_bytes,dtype='<u4').reshape(shape)
        sequences={name:[] for name in ('stbn',*protocol['controls'])}
        for block in protocol['blocks']:
            entry=entries[seed,block];sample_bytes=(args.consumer/entry['file']).read_bytes()
            actual=np.frombuffer(sample_bytes,dtype='<f4').reshape(shape)
            independently_computed=sample_float32(ranks,seed,block)
            mismatches=int(np.count_nonzero(actual.view(np.uint32)!=independently_computed.view(np.uint32)))
            in_range=bool(np.all(np.isfinite(actual)) and np.all(actual>=0) and np.all(actual<1))
            values={'stbn':actual.astype(np.float64),**controls(seed,block,shape)}
            for name,v in values.items():sequences[name].append(v)
            uniform=raw.distribution(np.floor(actual.astype(np.float64)*n).astype(np.uint32)) if in_range else {
                'rank_permutation':False,'exact_32_bin_histogram':False}
            result={**entry,'samples_sha256':raw.sha256(sample_bytes),'raw_ranks_sha256':old['ranks_sha256'],
                    'cpp_float32_mismatches':mismatches,'samples_compared':int(actual.size),'range_passed':in_range,
                    'relaxation_converged':old['relaxation_converged'],'generation_ms':old['generation_ms'],
                    'distribution':uniform,'correlations':{name:raw.correlations(v) for name,v in values.items()},
                    'raw_rank_spectra':{name:{'spatial':raw.spectrum(v),'temporal':raw.spectrum(v,True)} for name,v in values.items()},
                    'thresholds':{}}
            for threshold in protocol['thresholds']:
                binary={name:v<threshold for name,v in values.items()}
                result['thresholds'][str(threshold)]={axis:{name:raw.spectrum(v,axis=='temporal') for name,v in binary.items()}
                                                     for axis in ('spatial','temporal')}
            runs.append(result)
        sequences={name:np.concatenate(v,axis=1) for name,v in sequences.items()}
        sequence_diagnostics.append({'seed':seed,'thresholds':{str(q):{
            name:raw.spectrum(v<q,True) for name,v in sequences.items()} for q in protocol['thresholds']}})
    blocks={str(b):raw.summarize([r for r in runs if r['block']==b],protocol) for b in protocol['blocks']}
    bit_exact=all(r['cpp_float32_mismatches']==0 for r in runs)
    valid_range=all(r['range_passed'] for r in runs)
    overall=bit_exact and valid_range and all(b['passed'] for b in blocks.values())
    output={'scope':'actual CPU sample() values, blocks0..3, no GPU or rendered-quality claim',
            'preregistration_commit':REGISTRATION,'protocol_sha256':raw.sha256(protocol_bytes),'protocol':protocol,
            'raw_result_unchanged_sha256':raw.sha256(raw_bytes),'raw_original_result_passed':previous['summary']['passed'],
            'source_sha256':source_hashes,'metadata_sha256':raw.sha256(metadata_bytes),'compiler':metadata['compiler'],
            'analyzer_sha256':raw.sha256(Path(__file__).read_bytes()),'fft_helper_sha256':raw.sha256(Path(raw.__file__).read_bytes()),
            'driver_sha256':raw.sha256((repo/'tools/stbn_consumer_generate.cpp').read_bytes()),
            'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),
            'summary':{'passed':bool(overall),'cpp_float32_bit_exact':bit_exact,'range_passed':valid_range,
                       'samples_compared':sum(r['samples_compared'] for r in runs),'blocks':blocks},
            'runs':runs,'sequence64_temporal_diagnostic':sequence_diagnostics}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output['summary'],indent=2))
    return 0 if overall else 1

if __name__=='__main__':raise SystemExit(main())
