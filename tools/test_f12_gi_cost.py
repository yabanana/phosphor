#!/usr/bin/env python3
"""Small synthetic report checks; this test never runs a renderer."""
import contextlib, copy, datetime, io, json, pathlib, tempfile, types, unittest
import f12_gi_cost as cost

ROOT=pathlib.Path(__file__).resolve().parents[1]

class CostProtocolTest(unittest.TestCase):
    def test_aba_provenance_o7_and_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=pathlib.Path(temporary).resolve();build=folder/'build';(build/'shaders').mkdir(parents=True)
            app=build/'phosphor';app.write_bytes(b'NOT AN EXECUTABLE; SYNTHETIC TEST ONLY')
            (build/'shaders/phosphor.metallib').write_bytes(b'SYNTHETIC')
            (build/'CMakeCache.txt').write_text('CMAKE_BUILD_TYPE:STRING=Release\nPHOSPHOR_BUILD_APP:BOOL=ON\nPHOSPHOR_TRACY:BOOL=OFF\n')
            asset=folder/'Sponza.gltf';asset.write_text('{"buffers":[],"images":[]}')
            out=folder/'runs';args=types.SimpleNamespace(protocol=ROOT/'tools/testdata/f12/cost-protocol-v1.json',project=ROOT,app=app,sponza=asset,output=out,python='/usr/bin/python3')
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(cost.stage(args),0)
            plan=json.loads((out/'plan.json').read_text());self.assertEqual(len(plan['jobs']),18)
            values=lambda x:{'mean':x,'p50':x,'p95':x+1,'p99':x+2,'min':x-.1,'max':x+3}
            for i,job in enumerate(plan['jobs']):
                self.assertNotIn('--gi-grid',job['command']) if job['member']!='B' else self.assertIn('--gi-grid',job['command'])
                status={'passed':True,'returncode':0,'exit_marker':0,'command':job['command'],'commit':plan['commit'],'tracked_patch_sha256':plan['tracked_patch_sha256'],'artifacts':plan['artifacts'],'binary_sha256':plan['artifacts'][str(app)],'validation_env':{},'started_utc':(datetime.datetime(2026,10,7,tzinfo=datetime.timezone.utc)+datetime.timedelta(seconds=i*2)).isoformat(),'elapsed_seconds':1}
                report={'frames':512,'width':1920,'height':1080,'gpu_allocations':0,'vsync':False,'ui':False,'gpu_timing':True,'gpu_ms':values(12 if job['member']=='B' else 10),'cpu_ms':values(1),'frame_ms':values(12),'wait_ms':values(.1),'rendering':{'input_width_last':1920,'input_height_last':1080,'gpu_failures':0,'post':True,'upscaler_effective':'native','offscreen':False,'auto_exposure':False,'edr':False,'asset':'Sponza.gltf' if job['scene']=='sponza' else 'procedural-lighting-validation-v1','engine_resource_bytes_last':200 if job['member']=='B' else 100,'device_allocated_bytes_last':300,'parent_physical_footprint_last':400},'hardware':{'effective_capabilities':'apple10','physical_device':'Apple M5 Max'},'lighting':{'gi':'ddgi' if job['member']=='B' else 'off','direct':'restir','shadows':'off','reflections':'off','ao':'off','denoise_effective':'custom','denoise_requested':'custom','checks':0,'failures':0,'gi_visibility_disabled':False},'pipelines':{'failures':0,'reloadFailures':0},'passes':[{'name':'sample','frames':1,'gpu_ms':values(1)}]}
                path=pathlib.Path(job['folder']);(path/'report.json').write_text(json.dumps(report));(path/'run.log.status.json').write_text(json.dumps(status))
            analysis=types.SimpleNamespace(plan=out/'plan.json',output=out/'analysis.json')
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(cost.analyze(analysis),0)
            result=json.loads(analysis.output.read_text());self.assertEqual(result['scene_summaries']['cornell']['paired_gpu_delta_summary']['p50']['median_ms'],2)
            self.assertFalse(result['runs']['cornell-r1-A1']['pass_coverage']['steady_attribution_available'])
            # O7 is an actual failure, never waived as instrumentation noise.
            target=out/'cornell-r1-B/report.json';good=json.loads(target.read_text());bad=copy.deepcopy(good);bad['gpu_allocations']=1;target.write_text(json.dumps(bad))
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(cost.analyze(analysis),1)
            target.write_text(json.dumps(good))
            # A drifted trailing baseline invalidates its whole triple.
            target=out/'cornell-r1-A2/report.json';good=json.loads(target.read_text());bad=copy.deepcopy(good);bad['gpu_ms']=values(14);target.write_text(json.dumps(bad))
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(cost.analyze(analysis),1)
            target.write_text(json.dumps(good))
            # Metadata order cannot silently become A/A/B or concurrent GPU work.
            target=out/'cornell-r1-B/run.log.status.json';bad=json.loads(target.read_text());bad['started_utc']='2026-10-07T00:00:00+00:00';target.write_text(json.dumps(bad))
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(cost.analyze(analysis),1)

if __name__=='__main__':unittest.main()
