### 必要的库
```
pip install allennlp==1.3.0
pip install ltp==4.1.3.post1
pip install OpenCC==1.1.2
pip install pyarrow
pip install python-Levenshtein==0.12.1
pip install numpy==1.21.1
pip install transformers
```

### 预训练权重
codebert预训练模型。存放到./data/codebert/内。  命名为pytorch_model.bin
```
链接：https://pan.baidu.com/s/1iUYfKKiqty3vxSQW9fIAmg 
提取码：lpc3
```

codegec预训练模型。将model文件夹存放到本脚本的根目录下内
```
链接：https://pan.baidu.com/s/1LTt9YIJucoxkk4iReJcaMA 
提取码：wvgj
```

### 训练，预测模型
见Untitled.ipynb内，首先复制需要的模型到指定目录，然后运行预测或者训练


### .\tag文件夹内包含所有标签方法
tag_word.py 词
tag_span.py 区间
tag_seq.py  语句
compare.py 人工标注对比
convert_tag_codebert.py 标签转换