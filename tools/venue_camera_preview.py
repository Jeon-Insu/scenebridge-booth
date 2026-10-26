"""Four independent live tiles for venue alignment; no pose estimation or calibration writes."""
import argparse
import json
from pathlib import Path
import threading
import time
import cv2
import numpy as np
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--alternating',action='store_true',help='Alignment-only fallback before USB bandwidth correction')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/venue_finalize_20260918/preview'
OUT.mkdir(parents=True,exist_ok=True)
sources=sorted(p for p in Path('/dev/v4l/by-path').glob('*-usb-*:*-video-index0'))
import os
# Labels come from the install day's setup (VENUE_DATE=YYYYMMDD, as in the calibration tools).
setup_path=ROOT/f"smartphone_mocap/calibration/venue_{os.environ.get('VENUE_DATE','20260920')}/setup.json"
setup=json.loads(setup_path.read_text()) if setup_path.exists() else {}
# Labels are trusted only while every camera is still on the port recorded for it;
# after re-cabling the same path can belong to a different physical camera.
recorded={row['source'] for row in setup.get('cameras',[])}
identities={row['source']:row for row in setup.get('cameras',[])} if recorded=={str(p) for p in sources} else {}
controllers=[str(p).split('NVDA8000:')[1][:2] for p in sources]
SHARED={c for c in controllers if controllers.count(c)>1}
def is_shared(path): return str(path).split('NVDA8000:')[1][:2] in SHARED
frames={}; failures={}; stop=threading.Event()

def capture(index,path):
 cap=cv2.VideoCapture(str(path),cv2.CAP_V4L2)
 shared=is_shared(path)
 cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*('YUYV' if shared else 'MJPG')))
 cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
 cap.set(cv2.CAP_PROP_FPS,10 if shared else 15);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
 try:
  while not stop.is_set():
   ok,frame=cap.read()
   if ok:frames[index]=(frame,time.monotonic())
   else:failures[index]=failures.get(index,0)+1;time.sleep(.05)
 finally:cap.release()

(OUT/'ports.json').write_text(json.dumps({f'VIEW {i+1}':str(p) for i,p in enumerate(sources)},indent=2))
def shared_preview():
 # Alignment-only fallback: alternating streams avoids shared USB reservation.
 # These frames are NOT a synchronized calibration/triangulation capture.
 while not stop.is_set():
  for i,p in enumerate(sources):
   if not is_shared(p):continue
   cap=cv2.VideoCapture(str(p),cv2.CAP_V4L2)
   cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
   cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
   for _ in range(2):
    ok,frame=cap.read()
    if ok:frames[i]=(frame,time.monotonic())
   cap.release()
for i,p in enumerate(sources):
 if not args.alternating or not is_shared(p):threading.Thread(target=capture,args=(i,p),daemon=True).start()
if args.alternating:threading.Thread(target=shared_preview,daemon=True).start()
name='4 CAMERA LIVE - ALIGNMENT ONLY - Q to close'
cv2.namedWindow(name,cv2.WINDOW_NORMAL);cv2.resizeWindow(name,1600,940);cv2.moveWindow(name,40,40)
sys.path.insert(0,str(ROOT))
from smartphone_mocap.calibration import board_from_spec,board_min_corners,detect_charuco
dictionary,board=board_from_spec(setup.get('board'))
minimum=board_min_corners(setup.get('board'))
total=(board.getChessboardSize()[0]-1)*(board.getChessboardSize()[1]-1)
last=0
try:
 while True:
  tiles=[];now=time.monotonic()
  for i,p in enumerate(sources[:4]):
   entry=frames.get(i);tile=np.zeros((450,800,3),np.uint8)
   if entry:
    frame,stamp=entry
    corners,ids,_,markers=detect_charuco(frame,dictionary,board)
    board_view=frame.copy()
    if ids is not None:
     cv2.aruco.drawDetectedCornersCharuco(board_view,np.asarray(corners,dtype=np.float32).reshape(-1,1,2),np.asarray(ids,dtype=np.int32).reshape(-1,1))
    tile=cv2.resize(board_view,(800,450));age=now-stamp
    count=0 if ids is None else len(ids)
    cv2.putText(tile,f'ChArUco corners: {count}/{total}',(400,432),0,.6,(0,255,0) if count>=minimum else (0,0,255),2)
    if age>1:cv2.putText(tile,f'STALE {age:.1f}s',(240,225),0,1.2,(0,0,255),3)
   else:cv2.putText(tile,'WAITING FOR CAMERA',(200,225),0,1,(0,0,255),2)
   cv2.line(tile,(400,0),(400,449),(0,255,255),1);cv2.line(tile,(0,225),(799,225),(0,255,255),1)
   cv2.putText(tile,('ALTERNATING PREVIEW' if args.alternating and is_shared(p) else 'LIVE'),(12,432),0,.6,(0,255,255),2)
   cv2.rectangle(tile,(0,0),(799,48),(0,0,0),-1)
   identity=identities.get(str(p),{})
   letter=identity.get('id','UNASSIGNED').replace('cam_','')
   height=identity.get('height_m_user_reported')
   port=str(p).split('NVDA8000:')[1][:12]
   label=f'VIEW {i+1} = {letter}' + (f' | height {height*100:.0f}cm' if height is not None else '') + f' | {port}'
   cv2.putText(tile,label,(12,30),0,.8,(255,255,255),2)
   tiles.append(tile)
   if entry and now-last>2:cv2.imwrite(str(OUT/f'view_{i+1}.jpg'),entry[0])
  while len(tiles)<4:tiles.append(np.zeros((450,800,3),np.uint8))
  mosaic=np.vstack((np.hstack(tiles[:2]),np.hstack(tiles[2:])))
  cv2.imshow(name,mosaic)
  if now-last>2:cv2.imwrite(str(OUT/'mosaic.jpg'),mosaic);last=now
  if cv2.waitKey(25)&255 in [ord('q'),27]:break
finally:stop.set();cv2.destroyAllWindows()
