# -*- coding: utf-8
import os
from transformers import BertModel
import torch
import tokenization
import argparse
from tools.gec_model import GecBERTModel
import re
from opencc import OpenCC

cc = OpenCC("t2s")

def split_sentence(document: str, flag: str = "all", limit: int = 510):
    sent_list = []
    try:
        if flag == "zh":
            document = re.sub('(?P<quotation_mark>([。？！](?![”’"\'])))', r'\g<quotation_mark>\n', document)  
            document = re.sub('(?P<quotation_mark>([。？！])[”’"\'])', r'\g<quotation_mark>\n', document) 
        elif flag == "en":
            document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)  
            document = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', document)  
        else:
            document = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)
            document = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n', document)  

        sent_list_ori = document.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except:
        sent_list.clear()
        sent_list.append(document)
    return sent_list


def predict_for_file(input_file, output_file, model, batch_size, log=True, seg=False):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sents = [s.strip().replace("  "," ") for s in lines]
    subsents = []
    s_map = []
    for i, sent in enumerate(sents):  
        if seg:
            subsent_list = split_sentence(sent, flag="zh")
        else:
            subsent_list = [sent]
        s_map.extend([i for _ in range(len(subsent_list))])
        subsents.extend(subsent_list)
    assert len(subsents) == len(s_map)
    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in subsents:
        batch.append(sent.split())
        if len(batch) == batch_size: 
            preds, cnt = model.handle_batch(batch)
            assert len(preds) == len(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            if log:
                for z in zip(batch, preds):
                    print("source： " + "".join(z[0]))
                    print("target： " + "".join(z[1]))
                    print()
            batch = []

    if batch:
        preds, cnt = model.handle_batch(batch)
        assert len(preds) == len(batch)
        predictions.extend(preds)
        cnt_corrections += cnt
        if log:
            for z in zip(batch, preds):
                print("source： " + "".join(z[0]))
                print("target： " + "".join(z[1]))
                print()

    assert len(subsents) == len(predictions)
    print("prediction number:",len(predictions))
    if output_file:
        with open(output_file, 'w',encoding="utf-8") as f1:
            results = ["" for _ in range(len(sents))]
            for i, ret in enumerate(predictions):
                ret_new = [tok.lstrip("##") for tok in ret]
                ret = cc.convert(" ".join(ret_new))
                results[s_map[i]] += cc.convert(ret)
                #tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=False)
            print("results:",len(results))
            for ret in results:
                f1.write(ret + "\n")
            f1.close()
    return cnt_corrections


def main(args):
    model = GecBERTModel(vocab_path=args.vocab_path,model_paths=args.model_path.split(','),weights_names=args.weights_name.split(','),max_len=args.max_len, min_len=args.min_len,iterations=args.iteration_count,min_error_probability=args.min_error_probability,min_probability=args.min_error_probability,log=False,confidence=args.additional_confidence,is_ensemble=args.is_ensemble,weigths=args.weights,cuda_device=args.cuda_device)
    cnt_corrections = predict_for_file(args.input_file, args.output_file, model,
                                                   batch_size=args.batch_size, log=args.log, seg=args.seg)
    print(f"Produced overall corrections: {cnt_corrections}")


if __name__ == '__main__':
    #参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file',
                        required=True)  
    parser.add_argument('--weights_name',
                        help='Path to the bert encoder model',
                        default='bert',
                        required=True)  
    parser.add_argument('--vocab_path',
                        help='Path to the vocab file',
                        default='',
                        ) 
    parser.add_argument('--input_file',
                        required=True)  
    parser.add_argument('--output_file',
                        required=True)  
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length',
                        default=200) 
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length',
                        default=0) 
    parser.add_argument('--batch_size',
                        type=int,
                        default=128) 
    parser.add_argument('--iteration_count',
                        type=int,
                        default=5) 
    parser.add_argument('--additional_confidence',
                        type=float,
                        default=0.0) 
    parser.add_argument('--min_probability',
                        type=float,
                        default=0) 
    parser.add_argument('--min_error_probability',
                        type=float,
                        default=0.0)  
    parser.add_argument('--is_ensemble', 
                        type=int,
                        default=0)  
    parser.add_argument('--weights',
                        nargs='+',
                        default=None) 
    parser.add_argument('--cuda_device',  
                        help='The number of GPU',
                        default=0)
    parser.add_argument('--log',  
                        action='store_true')  
    parser.add_argument('--seg',  
                        action='store_true')  
    args = parser.parse_args()
    main(args)
