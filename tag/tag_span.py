#-*- coding: utf-8 -*-
import errant
import re
from tqdm import tqdm
import time
import argparse
import os 
from transformers import RobertaTokenizer

class editForCode():
    def __init__ (self):
        self.start=''
        self.end=''
        self.type=''
        self.seq=''
class wordPack():
    def __init__ (self,word,opIndex=-1,replaceSeq='self',insertSeq='self',type='self'):
        self.opIndex=opIndex
        self.tokens=[]
        self.delete=False
        self.keep=True
        self.insert=False
        self.replace=False    
        self.replaceSeq=replaceSeq  
        self.insertSeq=insertSeq 
        self.type=type
        
        if(word.startswith(' ')):
            self.word=word
        else:
            self.word=' '+word
    

class getEditBetweenSourAndTgt():
    def __init__(self,tokenizerName=None):
        self.annotator = errant.load('en')
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    
    def doEdit(self,source,target):
        source=source.strip()
        source= ' '.join(source.split())
        target=target.strip()
        target= ' '.join(target.split())
        source_tokens=[self.tokenizer.tokenize(token) for token in source.split()]
        target_tokens=[self.tokenizer.tokenize(token) for token in target.split()]
        orig = self.annotator.parse(source)
        cor = self.annotator.parse(target)
        alignment = self.annotator.align(orig, cor)
        edits = self.annotator.merge(alignment)
        wordPacks=[]
        for eIndex in range(len(edits)):
            e=edits[eIndex]
            if e.o_str and e.c_str:
                packt=(e.c_str,'replace',e.o_start,e.o_end,source_tokens,target_tokens)
                wordPacks.append(packt)
            elif e.o_str:
                packt=('self','delete',e.o_start,e.o_end,source_tokens,target_tokens)
                wordPacks.append(packt)
            elif e.c_str:
                packt=(e.c_str,'insert',e.o_start,e.o_end,source_tokens,target_tokens)
                wordPacks.append(packt)
            else:
                print('found an unknown edit type!')
        return wordPacks

class dotagwordspan():
  def __init__(self,datetype):
    self.genEdit=getEditBetweenSourAndTgt()
    self.type=datetype
    
  def startwork (self):
    if not os.path.exists(r'.\data_span'):
      os.makedirs(r'.\data_span')
    if not os.path.exists(r'.\data_span\{}'.format(self.type)):
      os.makedirs(r'.\data_span/{}'.format(self.type))
    buggycodeFile=open('{}.buggy-fixed.buggy'.format(self.type),'r',encoding='utf-8')
    fixedcodeFile=open('{}.buggy-fixed.fixed'.format(self.type),'r',encoding='utf-8')
    outputPosFile=open('./data_span/{}/{}.set.pos'.format(self.type,self.type),'w',encoding='utf-8')
    outputTagFile=open('./data_span/{}/{}.set.tag'.format(self.type,self.type),'w',encoding='utf-8')
    outputWordFile=open('./data_span/{}/{}.set.word'.format(self.type,self.type),'w',encoding='utf-8')
    outputSrcFile=open('./data_span/{}/{}.set.src'.format(self.type,self.type),'w',encoding='utf-8')
    outputTgtFile=open('./data_span/{}/{}.set.tgt'.format(self.type,self.type),'w',encoding='utf-8')

    buggycodes=buggycodeFile.readlines()
    fixedcodes=fixedcodeFile.readlines()
  
    for index in tqdm(range(len(buggycodes))):
      w,t,p=self.doOne(buggycodes[index].strip(),fixedcodes[index].strip())
      outputPosFile.write('<<<<>>>>'.join('%s' %id for id in p)+'\n')
      outputWordFile.write('<<<<>>>>'.join(w)+'\n')
      outputTagFile.write('<<<<>>>>'.join(t)+'\n')
      outputSrcFile.write(buggycodes[index].strip()+'\n')
      outputTgtFile.write(fixedcodes[index].strip()+'\n')

    buggycodeFile.close()
    fixedcodeFile.close()
    outputPosFile.close()
    outputTagFile.close()
    outputWordFile.close()
    outputSrcFile.close()
    outputTgtFile.close()

  def doOne(self,buggyCode,fixedCode):
    for uselessI in range(1):
      wordPacks=self.genEdit.doEdit(buggyCode,fixedCode)
      curi=0
      w=[]
      t=[]
      p=[]
      for pack in wordPacks:
        if pack[2]>curi:
          w.append('self')
          t.append('self')
          p.append(pack[2])
        w.append(pack[0])
        t.append(pack[1])
        p.append(pack[3])
        curi=pack[3]
      ls=len(buggyCode.strip().split())
      if(curi<ls):
        w.append('self')
        t.append('self')
        p.append(ls)
      return w,t,p

if __name__ =='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("type", type=str,choices=['train','dev','test'])
  args = parser.parse_args()
  c=dotagwordspan(args.type)
  c.startwork()