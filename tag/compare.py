from tag_word import dotagwordw
from tag_span import dotagwordspan
from tag_seq import dotagwordseq
from tqdm import tqdm
import time
datatype='label'
outputPosFile=open('./data_label/{}.set.pos'.format(datatype),'r',encoding='utf-8')
outputTagFile=open('./data_label/{}.set.tag'.format(datatype),'r',encoding='utf-8')
outputWordFile=open('./data_label/{}.set.word'.format(datatype),'r',encoding='utf-8')
buggycodeFile=open('./data_label/{}.set.src'.format(datatype),'r',encoding='utf-8')
fixedcodeFile=open('./data_label/{}.set.tgt'.format(datatype),'r',encoding='utf-8')

buggycodes=buggycodeFile.readlines()
fixedcodes=fixedcodeFile.readlines()
label_w=outputWordFile.readlines()
label_t=outputTagFile.readlines()
label_p=outputPosFile.readlines()

time_word=0
time_span=0
time_seq=0
acc_word=0
acc_span=0
acc_seq=0
workw=dotagwordw('label')
workspan=dotagwordspan('label')
workseq=dotagwordseq('label',5)
for index in tqdm(range(len(buggycodes))):
  lw=label_w[index].strip().split('<<<<>>>>')
  lt=label_t[index].strip().split('<<<<>>>>')
  lp=[int(p) for p in label_p[index].strip().split('<<<<>>>>')]
  bug=buggycodes[index].strip()
  fix=fixedcodes[index].strip()
  start=time.perf_counter()
  w,t,p=workw.doOne(bug,fix)
  time_word+=(time.perf_counter()-start)
  if lp==p:
    acc_word+=1
  start=time.perf_counter()
  w,t,p=workspan.doOne(bug,fix)
  time_span+=(time.perf_counter()-start)
  if lp==p:
    acc_span+=1
  start=time.perf_counter()
  w,t,p=workseq.doOne(bug,fix)
  time_seq+=(time.perf_counter()-start)
  if lp==p:
    acc_seq+=1



print('基于词')
print('准确率:',acc_word/350)
print('耗时:',time_word)
print('基于区间')
print('准确率:',acc_span/350)
print('耗时:',time_span)
print('基于语句')
print('准确率:',acc_seq/350)
print('耗时:',time_seq)
