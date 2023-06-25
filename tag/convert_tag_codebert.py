from transformers import AutoTokenizer
from tag_seq import dotagwordseq
from tqdm import tqdm
import argparse
import os 
class convertor():
  def __init__ (self):
    self.tokenizer=AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
  
  def doConveror(self,bug,w,t,p):
    bugs=bug.split()
    ret=['KEEP']+['KEEP' for _ in bugs]
    retw=['']+['' for _ in bugs]
    lasti=0
    for index in range(len(t)):
      if t[index]=='self':
        lasti=p[index]
        continue
      elif t[index]=='replace':
        ws=w[index].split()
        for i in range(lasti+1,p[index]+1):
          ret[i]='replace'
          if i-lasti-1 < len(ws):
            ret[i]='replace'
            retw[i]=ws[i-lasti-1]
          else:
            ret[i]='delete'
        if len(ws)>p[index]-lasti:
          retw[p[index]]= retw[p[index]]+('' if retw[p[index]]=='' else ' ')+' '.join(ws[p[index]-lasti:])
        lasti=p[index]
        continue
      elif t[index]=='insert':
        ret[p[index]]='insert'
        retw[p[index]]=w[index]
        lasti=p[index]
        continue
      elif t[index]=='delete':
        for i in range(lasti+1,p[index]+1):
          ret[i]='delete'
        lasti=p[index]
        continue
      else:
        raise ValueError('error tag')
        exit()
    for ri in range(len(ret)):
      if ri==0:
        if ret[ri]=='KEEP':
          ret[ri]='$STARTSEPL|||SEPR$KEEP'
        elif ret[ri]=='insert':
          ret[ri]='$STARTSEPL|||'
          tokens=self.tokenizer.tokenize(retw[ri])
          for token in tokens[:-1]:
            ret[ri]=ret[ri]+'SEPR$APPEND_'+token+'SEPL__'
          ret[ri]=ret[ri]+'SEPR$APPEND_'+tokens[-1]
        else:
          raise ValueError('error tag when ri is 0')
          exit()
      else:
        if ret[ri]=='KEEP':
          ret[ri]=''
          if ri-1==0:
            tokens=self.tokenizer.tokenize(bugs[ri-1])
          else:
            tokens=self.tokenizer.tokenize(' '+bugs[ri-1])
          for token in tokens:
            ret[ri]=ret[ri]+token+ 'SEPL|||SEPR$KEEP '
          ret[ri]=ret[ri][:-1]
          
        elif ret[ri]=='insert':
          ret[ri]=''
          if ri-1==0:
            tokens=self.tokenizer.tokenize(bugs[ri-1])
          else:
            tokens=self.tokenizer.tokenize(' '+bugs[ri-1])
          for token in tokens[:-1]:
            ret[ri]=ret[ri]+token+ 'SEPL|||SEPR$KEEP '
          ret[ri]=ret[ri]+tokens[-1]+ 'SEPL|||'
          tokens=self.tokenizer.tokenize(' '+retw[ri])
          for token in tokens[:-1]:
            ret[ri]=ret[ri]+'SEPR$APPEND_'+token+'SEPL__'
          ret[ri]=ret[ri]+'SEPR$APPEND_'+tokens[-1]
       
        elif ret[ri]=='delete':
          ret[ri]=''
          if ri-1==0:
            tokens=self.tokenizer.tokenize(bugs[ri-1])
          else:
            tokens=self.tokenizer.tokenize(' '+bugs[ri-1])
          for token in tokens:
            ret[ri]=ret[ri]+token+ 'SEPL|||SEPR$DELETE '
          ret[ri]=ret[ri][:-1]
          
        elif ret[ri]=='replace':
          ret[ri]=''
          if ri-1==0:
            tokens1=self.tokenizer.tokenize(bugs[ri-1])
            tokens2=self.tokenizer.tokenize(retw[ri])
          else:
            tokens1=self.tokenizer.tokenize(' '+bugs[ri-1])
            tokens2=self.tokenizer.tokenize(' '+retw[ri])
          ci=0
          for i in range(min(len(tokens1),len(tokens2))):
            ret[ri]=ret[ri]+tokens1[i]+'SEPL|||SEPR$REPLACE_'+tokens2[i]+' '
            ci=i
          if len(tokens1)>len(tokens2):
            for i in range(ci+1,len(tokens1)):
              ret[ri]=ret[ri]+tokens1[i]+'SEPL|||SEPR$DELETE '
          elif len(tokens1)<len(tokens2):
            ret[ri]=ret[ri][:-1]
            for i in range(ci+1,len(tokens2)):
              ret[ri]=ret[ri]+'SEPL__SEPR$APPEND_'+tokens2[i]
            ret[ri]=ret[ri]+' '
          ret[ri]=ret[ri][:-1]
        else:
          raise ValueError('unknown tag')
          exit()
      
    return ret
    

if __name__ =='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("datafold", type=str,choices=['data_seq','data_span','data_word'])
  args = parser.parse_args()
  datatypes=['train','dev','test']
  c=convertor()
  if not os.path.exists(r'.\codebert_label'):
    os.makedirs(r'.\codebert_label')
  labels=[]
  for typem in datatypes:
    outputPosFile=open('./{}/{}/{}.set.pos'.format(args.datafold,typem,typem),'r',encoding='utf-8')
    outputTagFile=open('./{}/{}/{}.set.tag'.format(args.datafold,typem,typem),'r',encoding='utf-8')
    outputWordFile=open('./{}/{}/{}.set.word'.format(args.datafold,typem,typem),'r',encoding='utf-8')
    outputSrcFile=open('./{}/{}/{}.set.src'.format(args.datafold,typem,typem),'r',encoding='utf-8')

    ss=outputSrcFile.readlines()
    ws=outputWordFile.readlines()
    ts=outputTagFile.readlines()
    ps=outputPosFile.readlines()
    
    outputLabelFile=open('./codebert_label/{}.{}.label'.format(args.datafold,typem),'w',encoding='utf-8')
    
    for index in tqdm(range(len(ss))):
      s=ss[index].strip()
      w=ws[index].strip().split('<<<<>>>>')
      t=ts[index].strip().split('<<<<>>>>')
      p=ps[index].strip().split('<<<<>>>>')
      p=[int(i) for i in p]
      ret=c.doConveror(s,w,t,p)
      outputLabelFile.write(' '.join(ret)+'\n')
    outputPosFile.close()
    outputTagFile.close()
    outputWordFile.close()

  
