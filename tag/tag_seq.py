import argparse
import os
from tqdm import tqdm
import copy
import argparse
import time as _te
from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
import re
import platform
from rapidfuzz.distance import Indel
import spacy.parts_of_speech as POS
import errant

Match = _namedtuple('Match', 'a b size')
Prep = _te.perf_counter
def _calculate_ratio(matches, length):
    if length:
        return 2.0 * matches / length
    return 1.0

class editTagGen:
    _open_pos = {POS.ADJ, POS.ADV, POS.NOUN, POS.VERB}
    annotator = errant.load('en')
    
    def __init__(self, a='', b='', isjunk=None,autojunk=True):
        self.isjunk = isjunk
        self.a = self.b = None
        self.autojunk = autojunk
        self.set_seqs(a, b)



    def do_CodeTagGen(self):
        res_tags=[]
        res_addingwords=[]
        res_position=[]
        res_position_start=[]
        for tag, i1, i2, j1, j2 in self.get_opcodes():
            if tag == 'delete':
                res_tags.append('delete')
                res_position_start.append(i1)
                res_position.append(i2)
                res_addingwords.append('self')
            elif tag == 'equal':
                res_tags.append('self')
                res_position_start.append(i1)
                res_position.append(i2)
                res_addingwords.append('self')
            elif tag == 'insert':
                res_tags.append('insert')
                res_position_start.append(i1)
                res_position.append(i2)
                res_addingwords.append(' '.join(self.b[j1:j2]))
            elif tag == 'replace':
                res_tags.append('replace')
                res_position_start.append(i1)
                res_position.append(i2)
                res_addingwords.append(' '.join(self.b[j1:j2]))
        return res_tags,res_addingwords,res_position,res_position_start

    def set_seqs(self, a, b):
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, a):
        if a is self.a:
            return
        self.a = a
        self.matching_blocks = self.opcodes = None

    def set_seq2(self, b):
        if b is self.b:
            return
        self.b = b
        self.matching_blocks = self.opcodes = None
        self.fullbcount = None
        self.__chain_b()
    def __chain_b(self):
        b = self.b
        self.b2j = b2j = {}

        for i, elt in enumerate(b):
            indices = b2j.setdefault(elt, [])
            indices.append(i)

        self.bjunk = junk = set()
        isjunk = self.isjunk
        if isjunk:
            for elt in b2j.keys():
                if isjunk(elt):
                    junk.add(elt)
            for elt in junk: 
                del b2j[elt]

        self.bpopular = popular = set()
        n = len(b)
        if self.autojunk and n >= 200:
            ntest = n // 100 + 1
            for elt, idxs in b2j.items():
                if len(idxs) > ntest:
                    popular.add(elt)
            for elt in popular:
                del b2j[elt]

    def find_longest_match(self, alo=0, ahi=None, blo=0, bhi=None):
        a, b, b2j, isbjunk = self.a, self.b, self.b2j, self.bjunk.__contains__
        if ahi is None:
            ahi = len(a)
        if bhi is None:
            bhi = len(b)
        besti, bestj, bestsize = alo, blo, 0

        j2len = {}
        nothing = []
        for i in range(alo, ahi):
            j2lenget = j2len.get
            newj2len = {}
            for j in b2j.get(a[i], nothing):
                if j < blo:
                    continue
                if j >= bhi:
                    break
                k = newj2len[j] = j2lenget(j-1, 0) + 1
                if k > bestsize:
                    besti, bestj, bestsize = i-k+1, j-k+1, k
            j2len = newj2len

        while besti > alo and bestj > blo and \
              not isbjunk(b[bestj-1]) and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              not isbjunk(b[bestj+bestsize]) and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize += 1

        while besti > alo and bestj > blo and \
              isbjunk(b[bestj-1]) and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              isbjunk(b[bestj+bestsize]) and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize = bestsize + 1

        return Match(besti, bestj, bestsize)


    def get_matching_blocks(self):
        if self.matching_blocks is not None:
            return self.matching_blocks
        la, lb = len(self.a), len(self.b)
        queue = [(0, la, 0, lb)]
        matching_blocks = []
        while queue:
            alo, ahi, blo, bhi = queue.pop()
            i, j, k = x = self.find_longest_match(alo, ahi, blo, bhi)
            if k:   
                matching_blocks.append(x)
                if alo < i and blo < j:
                    queue.append((alo, i, blo, j))
                if i+k < ahi and j+k < bhi:
                    queue.append((i+k, ahi, j+k, bhi))
        matching_blocks.sort()
        i1 = j1 = k1 = 0
        non_adjacent = []
        for i2, j2, k2 in matching_blocks:
            if i1 + k1 == i2 and j1 + k1 == j2:
                k1 += k2
            else:
                if k1:
                    non_adjacent.append((i1, j1, k1))
                i1, j1, k1 = i2, j2, k2
        if k1:
            non_adjacent.append((i1, j1, k1))

        non_adjacent.append( (la, lb, 0) )
        self.matching_blocks = list(map(Match._make, non_adjacent))
        return self.matching_blocks

    def get_opcodes(self):
        if self.opcodes is not None:
            return self.opcodes
        i = j = 0
        self.opcodes = answer = []
        for ai, bj, size in self.get_matching_blocks():
            tag = ''
            if i < ai and j < bj:
                tag = 'replace'
            elif i < ai:
                tag = 'delete'
            elif j < bj:
                tag = 'insert'
            if tag:
                answer.append( (tag, i, ai, j, bj) )
            i, j = ai+size, bj+size
            if size:
                answer.append( ('equal', ai, i, bj, j) )
        return answer
    def get_lscodes(self):
        table = self.get_grouped_table(self.a.strip().split(), self.b.strip().split())
        return self.get_backtrack(table, self.a.strip().split(), self.b.strip().split(), len(self.a.strip().split()), len(self.b.strip().split()))
      
    def get_grouped_table(self,a, b):
        rows = len(a)
        cols = len(b)
        grouped_table = [[0] * (cols + 1) for _ in range(rows + 1)]
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if a[i - 1] == b[j - 1]:
                  grouped_table[i][j] = grouped_table[i - 1][j - 1] + 1
                else:
                  grouped_table[i][j] = max(grouped_table[i - 1][j], grouped_table[i][j - 1])
        return grouped_table

    def get_backtrack(self,table, a, b, i, j):
        if i == 0 or j == 0:
            return []

        if a[i - 1] == b[j - 1]:
            return self.get_backtrack(table, a, b, i - 1, j - 1) + [b[j - 1]]
        if table[i][j - 1] > table[i - 1][j]:
            return self.get_backtrack(table, a, b, i, j - 1)
        else:
            return self.get_backtrack(table, a, b, i - 1, j)
      
    def get_grouped_opcodes(self, n=3):
        codes = self.get_opcodes()
        if not codes:
            codes = [("equal", 0, 1, 0, 1)]
        if codes[0][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[0]
            codes[0] = tag, max(i1, i2-n), i2, max(j1, j2-n), j2
        if codes[-1][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[-1]
            codes[-1] = tag, i1, min(i2, i1+n), j1, min(j2, j1+n)

        nn = n + n
        group = []
        for tag, i1, i2, j1, j2 in codes:
            if tag == 'equal' and i2-i1 > nn:
                group.append((tag, i1, min(i2, i1+n), j1, min(j2, j1+n)))
                yield group
                group = []
                i1, j1 = max(i1, i2-n), max(j1, j2-n)
            group.append((tag, i1, i2, j1 ,j2))
        if group and not (len(group)==1 and group[0][0] == 'equal'):
            yield group

    def ratio(self):
        matches = self.get_lscodes()
        matches = sum(triple[-1] for triple in self.get_matching_blocks())
        return _calculate_ratio(matches, len(self.a) + len(self.b))

    def quick_ratio(self):
        if self.fullbcount is None:
            self.fullbcount = fullbcount = {}
            for elt in self.b:
                fullbcount[elt] = fullbcount.get(elt, 0) + 1
        fullbcount = self.fullbcount
        avail = {}
        availhas, matches = avail.__contains__, 0
        for elt in self.a:
            if availhas(elt):
                numb = avail[elt]
            else:
                numb = fullbcount.get(elt, 0)
            avail[elt] = numb - 1
            if numb > 0:
                matches = matches + 1
        return _calculate_ratio(matches, len(self.a) + len(self.b))

    def real_quick_ratio(self):
        la, lb = len(self.a), len(self.b)
        return _calculate_ratio(min(la, lb), la + lb)

    def get_Levenshtein_dis(self,normal=False):
        cost_matrix_a = self.align(normal)
        matches = self.get_lscodes()
        matches = sum(triple[-1] for triple in self.get_matching_blocks())
        cost_matrix_b= _calculate_ratio(matches, len(self.a) + len(self.b))
        return -cost_matrix_a[-1][-1] if cost_matrix_b>2 else cost_matrix_b
        
    def align(self, lev):
        orig = self.annotator.parse(self.a)
        cor = self.annotator.parse(self.b)
        orig=[o.orth_ for o in orig]
        cor=[c.orth_ for c in cor]
        o_len = len(orig)
        c_len = len(cor)
        cost_matrix = [[0.0 for j in range(c_len+1)] for i in range(o_len+1)]
        op_matrix = [["O" for j in range(c_len+1)] for i in range(o_len+1)]
        for i in range(1, o_len+1):
            cost_matrix[i][0] = cost_matrix[i-1][0] + 1
        for j in range(1, c_len+1):
            cost_matrix[0][j] = cost_matrix[0][j-1] + 1
        for i in range(o_len):
            for j in range(c_len):
                if orig[i] == cor[j]:
                    cost_matrix[i+1][j+1] = cost_matrix[i][j]
                else:
                    del_cost = cost_matrix[i][j+1] + 1
                    ins_cost = cost_matrix[i+1][j] + 1
                    trans_cost = float("inf")
                    if lev: sub_cost = cost_matrix[i][j] + 1
                    else:
                        sub_cost = cost_matrix[i][j] + self.get_sub_cost(orig[i], cor[j])
                        k = 1
                        while i-k >= 0 and j-k >= 0 and \
                                cost_matrix[i-k+1][j-k+1] != cost_matrix[i-k][j-k]:
                            if sorted(orig[i-k:i+1]) == sorted(cor[j-k:j+1]):
                                trans_cost = cost_matrix[i-k][j-k] + k
                                break
                            k += 1
                    costs = [trans_cost, sub_cost, ins_cost, del_cost]
                    l = costs.index(min(costs))
                    cost_matrix[i+1][j+1] = costs[l]
        return cost_matrix

    def get_sub_cost(self, o, c):
        if o == c: return 0
        if o.lower() == c.lower(): lemma_cost = 0
        else: lemma_cost = 0.499
        pos_cost=0
        char_cost = Indel.normalized_distance(o, c)
        return lemma_cost + pos_cost + char_cost





class ali_solution():
  def __init__(self,bugglen,fixedlen):
    self.sub_solutions=[]
    self.buggy_ali=[-1]*bugglen
    self.fixed_ali=[-1]*fixedlen
    self.s=0
  
  def getAliedBuggyIndex(self):
    r=[]
    for index in range(len(self.buggy_ali)):
      if self.buggy_ali[index]>=0:
        r.append(index)
    return r
  
  def getAliedFixedIndex(self):
    r=[]
    for index in range(len(self.fixed_ali)):
      if self.fixed_ali[index]>=0:
        r.append(index)
    return r

  def is_legal(self,buggy_index,fixed_index):
    tof=self.buggy_ali[buggy_index]
    ulimit=next((self.buggy_ali[x] for x in range(buggy_index+1,len(self.buggy_ali)) if self.buggy_ali[x]>-1),len(self.fixed_ali))
    llimit=next((self.buggy_ali[x] for x in reversed(range(buggy_index)) if self.buggy_ali[x]>-1),-1)
    if fixed_index<=llimit or fixed_index>=ulimit:
      return False
    return True
  
  def add_sub_solution(self,s,buggy_index,fixed_index):
    self.s=self.s+s
    self.buggy_ali[buggy_index]=fixed_index
    if fixed_index >=0:
        self.fixed_ali[fixed_index]=buggy_index

class dotagwordseq():
  def __init__(self,datatype,kv):
    self.type=datatype
    self.kvalue=kv
    
  def getFormatCode(self,originCode):
    originCode=originCode.strip()
    javaFormaterInput=open('input.java','w',encoding='utf-8')
    javaFormaterInput.write('public class test {'+originCode+'}')
    javaFormaterInput.close()
    os_name = platform.system()
    if os_name == "Windows":
      os.system('java -jar .\javaf.jar --replace input.java  > NUL 2>&1')
    else:
      os.system('java -jar javaf.jar --replace input.java  > /dev/null 2>&1')
    javaFormaterOutput=open('input.java','r',encoding='utf-8')
    formatedCode=javaFormaterOutput.readlines()
    formatedCode=[c.strip() for c in formatedCode[1:-1]]
    formatedCode_withzip=[]
    index=-1
    for code in formatedCode:
      if code =='{' or code =='}':
        formatedCode_withzip[index]=formatedCode_withzip[index]+' '+code
      else:
        index+=1
        formatedCode_withzip.append(code)
    formatedCode_withzip_withspace=[]
    originCode=originCode.split()
    starti=0
    for formatedCode_withzip_item in formatedCode_withzip:
      nospace=formatedCode_withzip_item.replace(' ','')
      for i in range(starti,len(originCode)+1):
        if ''.join(originCode[starti:i]) == nospace:
          formatedCode_withzip_withspace.append(' '.join(originCode[starti:i]))
          starti=i
          break
    assert len(formatedCode_withzip_withspace)==len(formatedCode_withzip)
    return formatedCode_withzip_withspace


  def getFormatCode_nocmd(self,originCode):
    split_regex = r"\s*([\{\}\;])\s*"
    code_blocks = re.split(split_regex, originCode)
    indent_level = 0
    indent_size = 4
    output_str = ""
    for block in code_blocks:
      if block == '{':
        output_str += f"{block}\n" + " " * indent_level * indent_size
        indent_level += 1
      elif block == '}':
        indent_level -= 1
        output_str = output_str.rstrip() + f"\n" + " " * indent_level * indent_size + f"{block}\n" + " " * indent_level * indent_size
      elif block == ';':
        output_str += f"{block}\n" + " " * indent_level * indent_size
      elif block.isspace() or block == '':
        continue
      else:
        output_str += f"{block} "
    formatedCode=output_str.split('\n')
    index=-1
    formatedCode_withzip=[]
    for code in formatedCode:
      code=code.strip()
      if code=='':
        continue
      if code =='{' or code =='}':
        formatedCode_withzip[index]=formatedCode_withzip[index]+' '+code
      else:
        index+=1
        formatedCode_withzip.append(code)
    formatedCode_withzip_withspace=[]
    originCode=originCode.split()
    starti=0
    for formatedCode_withzip_item in formatedCode_withzip:
      nospace=formatedCode_withzip_item.replace(' ','')
      for i in range(starti,len(originCode)+1):
        if ''.join(originCode[starti:i]) == nospace:
          formatedCode_withzip_withspace.append(' '.join(originCode[starti:i]))
          starti=i
          break
    assert len(formatedCode_withzip_withspace)==len(formatedCode_withzip)
    return formatedCode_withzip_withspace

  def startwork(self):
    if not os.path.exists(r'.\data_seq'):
      os.makedirs(r'.\data_seq')
    if not os.path.exists(r'.\data_seq\{}'.format(self.type)):
      os.makedirs(r'.\data_seq/{}'.format(self.type))
    buggycodeFile=open('{}.buggy-fixed.buggy'.format(self.type),'r',encoding='utf-8')
    fixedcodeFile=open('{}.buggy-fixed.fixed'.format(self.type),'r',encoding='utf-8')
    outputPosFile=open('./data_seq/{}/{}.set.pos'.format(self.type,self.type),'w',encoding='utf-8')
    outputTagFile=open('./data_seq/{}/{}.set.tag'.format(self.type,self.type),'w',encoding='utf-8')
    outputWordFile=open('./data_seq/{}/{}.set.word'.format(self.type,self.type),'w',encoding='utf-8')
    outputSrcFile=open('./data_seq/{}/{}.set.src'.format(self.type,self.type),'w',encoding='utf-8')
    outputTgtFile=open('./data_seq/{}/{}.set.tgt'.format(self.type,self.type),'w',encoding='utf-8')

    buggycodes=buggycodeFile.readlines()
    fixedcodes=fixedcodeFile.readlines()
    for index_file in tqdm(range(len(buggycodes))):
      mw,mt,mp=self.doOne(buggycodes[index_file].strip(),fixedcodes[index_file].strip())
      outputPosFile.write('<<<<>>>>'.join('%s' %id for id in mp)+'\n')
      outputWordFile.write('<<<<>>>>'.join(mw)+'\n')
      outputTagFile.write('<<<<>>>>'.join(mt)+'\n')
      outputSrcFile.write(buggycodes[index_file].strip()+'\n')
      outputTgtFile.write(fixedcodes[index_file].strip()+'\n')

    buggycodeFile.close()
    fixedcodeFile.close()
    outputPosFile.close()
    outputTagFile.close()
    outputWordFile.close()
    outputSrcFile.close()
    outputTgtFile.close()
    
  def doOne(self,_buggyCode,_fixedCode):
    #buggyCode=self.getFormatCode(_buggyCode)
    buggyCode=self.getFormatCode_nocmd(_buggyCode)
    #fixedCode=self.getFormatCode(_fixedCode)
    fixedCode=self.getFormatCode_nocmd(_fixedCode)
    for uselessI in range(1):
      solutions=[]
      for index_buggycode in range(len(buggyCode)):
        sub_solutions=[]
        for index_fixedcode in range(len(fixedCode)):
          sub_solutions.append((editTagGen(buggyCode[index_buggycode], fixedCode[index_fixedcode]).get_Levenshtein_dis(),index_buggycode,index_fixedcode))
        sub_solutions=sorted(sub_solutions,key=lambda x: x[0],reverse=True)
        solutions.extend(sub_solutions[:5])
      solutions=sorted(solutions,key=lambda x: x[0],reverse=True)
      len_bug=len(buggyCode)
      len_fix=len(fixedCode)
      ali_solutions=[ali_solution(len_bug,len_fix) for x in range(self.kvalue)]
      for si in range(min(len(solutions),self.kvalue)):
        ali_solutions[si].add_sub_solution(solutions[si][0],solutions[si][1],solutions[si][2])
      
      for epoch in range(len_bug-1):
        ali_solutions_t=[]
        t=Prep()
        for solution in ali_solutions:
            bugis=solution.getAliedBuggyIndex()
            fixis=solution.getAliedFixedIndex()
            for k in range(min(len_bug,self.kvalue)):
              found=None
              for fsi in range(len(solutions)):
                if solutions[fsi][1] not in bugis and solutions[fsi][2] not in fixis:
                  if solution.is_legal(solutions[fsi][1],solutions[fsi][2]):
                    found=solutions[fsi]
                    break

              if found == None and len(bugis)+k<len_bug:
                c=set(range(len_bug)) - set(bugis)
                found=(-2,list(c)[k],-2)
              if found != None:
                st=copy.deepcopy(solution)
                st.add_sub_solution(found[0],found[1],found[2])
                ali_solutions_t.append(st)
        
        ali_solutions=sorted(ali_solutions_t,key=lambda x: x.s,reverse=True)[:self.kvalue]
      start_fixed_index=0
      offset=0
      buggy_ali=ali_solutions[0].buggy_ali
      rw=[]
      rp=[]
      rt=[]
      for rbi in range(len(buggy_ali)):
        fi=buggy_ali[rbi]
        if start_fixed_index<fi and fi>=0 :
          af=' '.join(fixedCode[start_fixed_index:fi])
          rw.append(af)
          rp.append(offset)
          offset=offset
          rt.append('insert')
        elif fi<0:
           rw.append('self')
           rt.append('delete')
           offset=offset + len(buggyCode[rbi].strip().split())
           rp.append(offset)
           continue
        start_fixed_index=fi+1
        genTool=editTagGen(buggyCode[rbi].split(),fixedCode[fi].split())
        res_tags,res_addingwords,res_position_end,res_position_start=genTool.do_CodeTagGen()
        for w,t,pe,ps in zip(res_addingwords,res_tags,res_position_end,res_position_start):
          rw.append(w)
          rt.append(t)
          rp.append(offset+pe)
        offset=offset + len(buggyCode[rbi].split())
      mw=[]
      mp=[]
      mt=[]
      for w,t,p in zip(rw,rt,rp):
        if len(mw)>0 and t==mt[-1]:
          mp[-1]=p
        else:
          mw.append(w)
          mt.append(t)
          mp.append(p)
      return mw,mt,mp



if __name__ =='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("type", type=str,choices=['train','dev','test','label'])
  parser.add_argument("kvalue", type=int)
  args = parser.parse_args()
  c=dotagwordseq(args.type,args.kvalue)
  c.startwork()

