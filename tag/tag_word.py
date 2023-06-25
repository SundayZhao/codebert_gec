import os
from tqdm import tqdm
import argparse


class tagging(object):
  def __init__(self, tag):
    if '|' in tag:
      pos_pipe = tag.index('|')
      tag_type, added_phrase = tag[:pos_pipe], tag[pos_pipe + 1:]
    else:
      tag_type, added_phrase = tag, ''
    try:
      self.tag_type = tag_type
    except KeyError:
      raise ValueError(
          'TagType should be self, DELETE or SWAP, not {}'.format(tag_type))
    self.added_phrase = added_phrase

  def __str__(self):
    if not self.added_phrase:
      return self.tag_type
    else:
      return '{}|{}'.format(self.tag_type, self.added_phrase)
      

class dotagwordw():
  def __init__(self,datetype):
    self.type=datetype
    self.ap=[]
    
  def startwork (self):
    if not os.path.exists(r'.\data_word'):
      os.makedirs(r'.\data_word')
    if not os.path.exists(r'.\data_word\{}'.format(self.type)):
      os.makedirs(r'.\data_word/{}'.format(self.type))
    buggycodeFile=open('{}.buggy-fixed.buggy'.format(self.type),'r',encoding='utf-8')
    fixedcodeFile=open('{}.buggy-fixed.fixed'.format(self.type),'r',encoding='utf-8')
    outputPosFile=open('./data_word/{}/{}.set.pos'.format(self.type,self.type),'w',encoding='utf-8')
    outputTagFile=open('./data_word/{}/{}.set.tag'.format(self.type,self.type),'w',encoding='utf-8')
    outputWordFile=open('./data_word/{}/{}.set.word'.format(self.type,self.type),'w',encoding='utf-8')
    outputSrcFile=open('./data_word/{}/{}.set.src'.format(self.type,self.type),'w',encoding='utf-8')
    outputTgtFile=open('./data_word/{}/{}.set.tgt'.format(self.type,self.type),'w',encoding='utf-8')

    buggycodes=buggycodeFile.readlines()
    fixedcodes=fixedcodeFile.readlines()

    for index in tqdm(range(len(buggycodes))):
      b=buggycodes[index].strip()
      f=fixedcodes[index].strip()
      w,t,p=self.doOne(b,f)
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
      source_tokens=buggyCode.split()
      target_tokens=fixedCode.split()
      tags=self._compute_tags_fixed_order(source_tokens,target_tokens)
      w=[]
      t=[]
      p=[]
      
      for ti in range(len(tags)):
        ct=tags[ti]
        crealt='self'
        if ct.added_phrase!='':
          w.append(ct.added_phrase)
          if ct.tag_type=='self':
            t.append('insert')
          else :
            t.append('replace')
          p.append(ti+1)
        if len(w)==0 or t[-1]!=ct.tag_type:
          w.append('self')
          t.append(ct.tag_type)
          p.append(ti+1)
        else:
          p[-1]=ti+1
      return w,t,p

      
  def _compute_tags_fixed_order(self, source_tokens, target_tokens):
    tags = [tagging('delete') for _ in source_tokens]
    source_token_idx = 0
    target_token_idx = 0
    while target_token_idx < len(target_tokens):
      self.ap=self._get_added_phrases(source_tokens,target_tokens)
      tags[source_token_idx], target_token_idx = self._compute_single_tag(source_token_idx,source_tokens, target_token_idx, target_tokens)
      if tags[source_token_idx].added_phrase:
        first_deletion_idx = self._find_first_deletion_idx(source_token_idx, tags)
        if first_deletion_idx != source_token_idx:
          tags[first_deletion_idx].added_phrase = (tags[source_token_idx].added_phrase)
          tags[source_token_idx].added_phrase = ''
      source_token_idx += 1
      if source_token_idx >= len(tags):
        break
    if target_token_idx >= len(target_tokens):
      return tags
    end_phrase=' '.join(target_tokens[target_token_idx:])
    tags[-1].added_phrase=tags[-1].added_phrase+end_phrase
    return tags
    
  def _find_first_deletion_idx(self, source_token_idx, tags):
      for idx in range(source_token_idx, 0, -1):
        if tags[idx - 1].tag_type != 'delete':
          return idx
      return 0
      

  def _compute_single_tag(self, source_token_idx,source_tokens, target_token_idx,target_tokens):
    source_token = source_tokens[source_token_idx]
    target_token = target_tokens[target_token_idx]
    if source_token == target_token:
      return tagging('self'), target_token_idx + 1
    added_phrase = ''
    for num_added_tokens in range(1, 10000):
      added_phrase += (' ' if added_phrase else '') + target_token
      next_target_token_idx = target_token_idx + num_added_tokens
      if next_target_token_idx >= len(target_tokens):
        if added_phrase not in self.ap:
          res = tagging('delete')
          res.added_phrase = added_phrase
          return res, next_target_token_idx
        else:
          self.ap.append(added_phrase)
          res = tagging('delete')
          res.added_phrase = added_phrase
          return res, next_target_token_idx
      target_token = target_tokens[next_target_token_idx]
      if source_token == target_token :
        return tagging('self|' + added_phrase), next_target_token_idx + 1
    return tagging('delete'), target_token_idx
    
    
  def _get_added_phrases(self,source_tokens, target_tokens):
    kept_tokens=[]
    kept_tokens = self._compute_lcs(source_tokens, target_tokens)
    kept_tokens = self._compute_lcs(source_tokens, target_tokens)
    added_phrases = []
    kept_idx = 0
    phrase = []
    for token in target_tokens:
      if kept_idx < len(kept_tokens) and token == kept_tokens[kept_idx]:
        kept_idx += 1
        if phrase:
          added_phrases.append(' '.join(phrase))
          phrase = []
      else:
        phrase.append(token)
    if phrase:
      added_phrases.append(' '.join(phrase))
    return added_phrases
    
    
  def _compute_lcs(self,source, target):
    table = self._lcs_table(source, target)
    return self._backtrack(table, source, target, len(source), len(target))
    
  def _lcs_table(self,source, target):
    rows = len(source)
    cols = len(target)
    lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
      for j in range(1, cols + 1):
        if source[i - 1] == target[j - 1]:
          lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
        else:
          lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    return lcs_table

  def _backtrack(self,table, source, target, i, j):
    if i == 0 or j == 0:
      return []
    if source[i - 1] == target[j - 1]:
      return self._backtrack(table, source, target, i - 1, j - 1) + [target[j - 1]]
    if table[i][j - 1] > table[i - 1][j]:
      return self._backtrack(table, source, target, i, j - 1)
    else:
      return self._backtrack(table, source, target, i - 1, j)


if __name__ =='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("type", type=str,choices=['train','dev','test'])
  args = parser.parse_args()
  c=dotagwordw(args.type)
  c.startwork()