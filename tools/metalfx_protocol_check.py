#!/usr/bin/env python3
"""Exercise the inherited-FD worker ABI and an idle period beyond 30 seconds."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import time


def exact(sock, count):
    result = b''
    while len(result) < count:
        part = sock.recv(count-len(result))
        if not part:
            raise RuntimeError('Worker closed the channel')
        result += part
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app',type=Path,default=Path('build/release/phosphor'))
    p.add_argument('--out',type=Path,default=Path('build/metalfx-protocol-check'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    w,h=640,360;rows=[];offsets=[];size=0
    for i,bpp in enumerate((8,4,4,1,2,8)):
        offsets.append(size);row=(((1 if i==4 else w)*bpp+255)//256)*256;rows.append(row);size+=row*(1 if i==4 else h)
    slot=((size+16383)//16384)*16384;total=slot*3
    results=[]
    for name,version,idle in [('invalid-version',2,0),('idle',1,35)]:
        with tempfile.TemporaryFile() as shared,(a.out/(name+'.log')).open('wb') as log:
            shared.truncate(total)
            parent,child=socket.socketpair();parent.settimeout(30)
            high=[fcntl.fcntl(fd,fcntl.F_DUPFD_CLOEXEC,10) for fd in (child.fileno(),shared.fileno(),log.fileno())]
            actions=[(os.POSIX_SPAWN_DUP2,high[0],3),(os.POSIX_SPAWN_DUP2,high[1],4),
                     (os.POSIX_SPAWN_DUP2,high[2],1),(os.POSIX_SPAWN_DUP2,high[2],2)]
            actions += [(os.POSIX_SPAWN_CLOSE,fd) for fd in high]
            pid=os.posix_spawn(str(a.app.resolve()),[str(a.app.resolve()),'--metalfx-worker'],os.environ,file_actions=actions)
            for fd in high:os.close(fd)
            child.close()
            parent.sendall(struct.pack('<4I14Q',0x46585434,version,w,h,*offsets,*rows,slot,total))
            if idle:
                reply=struct.unpack('<4I4Q',exact(parent,48))
                assert reply[:4]==(0x46585434,1,0,0) and reply[4]==0
                time.sleep(idle)
                assert os.waitpid(pid,os.WNOHANG)==(0,0),'Idle worker exited prematurely'
                # Closing the private channel must stop the idle process.
                parent.close()
            else:
                assert parent.recv(1)==b''
                parent.close()
            deadline=time.monotonic()+10;status=None
            while time.monotonic()<deadline:
                found,value=os.waitpid(pid,os.WNOHANG)
                if found:status=value;break
                time.sleep(.02)
            if status is None:
                os.kill(pid,9);os.waitpid(pid,0);raise RuntimeError('Worker failed to stop')
            code=os.waitstatus_to_exitcode(status)
            assert code==(0 if idle else 2),(name,code)
            results.append({'case':name,'idle_seconds':idle,'exit':code,'reaped':True})
            print(name,code,'PASS',flush=True)
    (a.out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
