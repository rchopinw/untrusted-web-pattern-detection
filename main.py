# Code by B.X. Weinstein Xiao
# Contact: bangxi_xiao@brown.edu


import numpy as np
import pandas as pd
from lxml import etree
import os
import chardet
import urllib.request, urllib.error
from gensim import models, corpora
from bs4 import BeautifulSoup, Comment
import re
import jieba
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


bd = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+，。！？“”《》：、． [’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
special_characters = ['#', '@', '_', '-', '&', '/', '=']
tags = ['<iframe>', '<meta>', '<image>', '<script>', '<object>', '<embed>']
jsc_funtags = ['setTimeout', 'setInterval', 'window.location', 'window.open', '.src', 'setAttribute',
               'innerHTML', 'document.location', 'document.cookie', 'document.write', 'navigator']
comb_funtags = [['unescape', 'escape'], ['split', 'replace']]
sus_tags = ['.exe', '.ini', '.dll', '.tmp']
special_change = ['\\', "\'", "\"", "\a", "\b", "\000", "\n", "\v", "\t", "\r", "\f"]
# 读取停止词
with open('stop_words.txt', 'r', encoding='utf_8_sig') as f:
    stop_words = f.readlines()
stop_words = list(map(lambda x: x.replace('\n', ''), stop_words))
# 读取url的file
with open('file_list.txt', 'r', encoding='utf_8_sig') as f:
    urls = f.readlines()
urls = list(map(lambda x: x.replace('\n', '').split(','), urls))
urls_df = pd.DataFrame(urls, columns=['num_index', 'label', 'id', 'url'])
# 设置html文件的储存路径
html_path = 'D:\\web analysis project\\file1'
# 读取d数据集
d_class_df = pd.read_csv('dataset_large.csv', encoding='utf_8_sig', engine='python')


# 下载html数据
def url_html_downloader(u):
    try:
        h = urllib.request.urlopen(u, timeout=2).read()
    except:
        return 'empty_'
    return h


d_class_htmls = []
for i in range(6):
    d_class_htmls += list(map(lambda x: ('999999',
                                         'd',
                                         str(np.random.rand(1)[0]).replace('.', ''),
                                         x,
                                         url_html_downloader(x)), d_class_df['url'].tolist()[i*20000:(i+1)*20000]))  # 大概需要运行三日
    d_class_htmls = list(filter(lambda x: x[-1] != 'empty_', d_class_htmls))  # 删掉empty的html
    d_class_htmls = list(filter(lambda x: len(x[-1]) != 0, d_class_htmls))  # 删掉长度为0的html
save_obj({'a': d_class_htmls}, 'html_intermid')  # 储存下载下来的html（暂存，以防后面出bug）
d_class_htmls = load_obj('html_intermid')['a']
ddf = pd.DataFrame(d_class_htmls, columns=['num_index', 'label', 'id', 'url', 'html_info'])
# 合并d类url的数据框和原始提取的数据框
urls_df = pd.concat([urls_df, ddf.drop(columns=['html_info'])], axis=0)
urls_df.to_csv('join_acc.csv', encoding='utf_8_sig', index=False)

# 写入本地
for i, j in zip(ddf['id'].tolist(), ddf['html_info'].tolist()):
    with open(html_path + '\\' + i, 'wb') as f:
        f.write(j)
# 获取html_path路径下所有html文件的文件名
html_files = os.listdir(html_path)


def auto_dropper(df, threshold=0.5):
    l = df.shape[0]
    for i, j in enumerate(df.columns):
        mp = df[j].isnull().sum() / l
        if mp >= threshold:
            df = df.drop(columns=[j])
        else:
            pass
    return df


def read_file(n):
    # 该函数用于读取html文档，并处理所有解码异常情况。
    with open(n, 'rb') as f:
        fr = f.readlines()

    def decoder(s):
        try:
            ds = str(s, 'utf_8_sig')
        except UnicodeDecodeError:
            try:
                ds = str(s, 'gbk')
            except UnicodeDecodeError:
                try:
                    ds = str(s, 'GB18030')
                except UnicodeDecodeError:
                    ds = str(s, 'gb2312', 'ignore')
        return ds

    fr = list(map(decoder, fr))
    return fr


def data_scaler(data, method='min-max'):
    # 数据标准化处理，四种方法可选['min-max', 'z-score', 'rank-min-max', 'none']，默认为'min-max'。
    # 其中，rank-min-max 为先进行升序排序，再进行 min-max 标准化
    # 标准化公式参考 https://blog.csdn.net/bbbeoy/article/details/70185798
    data = pd.DataFrame(data)

    if method == 'min-max':
        for i in data.columns:
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
    elif method == 'z-score':
        for i in data.columns:
            data[i] = (data[i] - data[i].mean()) / data[i].std()
    elif method == 'rank-min-max':
        for i in data.columns:
            data[i] = data[i].rank()
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
    elif method == 'none':
        data = data

    return data


def clean_file(x):
    x = x.replace('\n', '')
    x = x.replace('&nbsp', '')
    return x


def process_url(url):
    # URL长度处理
    u1 = len(url)
    # URL中点的个数
    u2 = url.count('.')
    # URL中数字的个数
    u3 = len(list(filter(str.isdigit, url)))
    # URL中特殊字符的个数
    u4 = sum(url.count(x) for x in special_characters)
    # URL中子域名的个数
    num_sub = url.count('http')
    u5 = num_sub - 1 if num_sub else 0
    # URL路径深度（层数）
    num_depth = url.replace('//', '').count('/')
    u6 = num_depth if num_depth else 0
    # 检测ip地址
    result = re.findall(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", url)
    u7 = int(1) if result else int(0)
    return [u1, u2, u3, u4, u5, u6, u7]


def process_html(html: list):
    # 接受参数为一个用html语言构成的list。
    j = ''.join(html)
    # HTML长度
    h1 = len(j)
    # 网页文本
    h2 = 0 if h1==0 else len(extract_pure_text(j)) / h1
    # URL出现次数
    h3 = j.count('http:')
    # 隐藏标签的数量
    h4 = j.count('none') + j.count('hidden')
    # h5-h10
    h5, h6, h7, h8, h9, h10 = [j.count(x) for x in tags]
    return [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]


def process_javascript(h: list):
    # 接受参数为一个由javascript构成的list。
    f = ''.join(h)
    soup = BeautifulSoup(f, 'lxml')
    pd = soup.find_all('script')
    pd = list(map(lambda x: x.text, pd))
    hh = ''.join(pd)
    # 计算j1-j18特征
    j1 = hh.count('eval')
    j2 = len(list(filter(lambda x: len(x) > 30, re.findall("\"([^\"]*)\"", hh))))
    j3 = len(hh)
    j4 = sum([hh.count(x) for x in sus_tags])
    j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j18 = [hh.count(x) for x in jsc_funtags]
    j15, j16 = [hh.count(x[0]) + hh.count(x[1]) for x in comb_funtags]
    j17 = 0 if len(f)==0 else j3 / len(f)

    return [j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15, j16, j17, j18]


def extract_pure_text(h):
    j = ''.join(h)
    # 去除style内容
    soup = BeautifulSoup(j, 'lxml')
    _ = [s.extract() for s in soup('style')]
    # 去除comments
    comments = soup.findAll(text=lambda text: isinstance(text, Comment))
    _ = [comment.extract() for comment in comments]
    # 去除<>的内容
    st = re.sub('<.+?>', '', soup.text)
    # 去除{}的内容
    st = re.sub(u"\\(.*?\\)|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", st)
    # 删除空格
    st = st.replace(' ', '')
    return st


def process_text(t: str,
                 n: int):
    # 剔除停止词
    for w in stop_words:
        t = t.replace(w, '')
    # 删除中英文标点
    for w in bd:
        t = t.replace(w, '')
    # 分词
    l = jieba.lcut(t)
    cl = Counter(l)
    # 删除转义符号
    for sc in special_change:
        cl.pop(sc)
    cl_tup = list(zip(cl.keys(), cl.values()))
    cl_tup = sorted(cl_tup, key=lambda x: x[1], reverse=True)
    words_all = list(set(cl.keys()))
    return cl_tup[:n], words_all


# # 处理url
# url_features = pd.DataFrame([process_url(x) for x in urls_df.url.tolist()],
#                             columns=['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7'])
# url_features['id'] = urls_df['id'].tolist()
# url_features.to_csv('url_features.csv', encoding='utf_8_sig', index=False)
#
# # 处理html
# html_features = []
# for i in html_files:
#     h = read_file(html_path + '\\' + i)
#     h_c = list(map(clean_file, h))
#     html_features.append(process_html(h_c))
# html_df = pd.DataFrame(html_features, columns=['h1', 'h2', 'h3', 'h4', "h5", 'h6', "h7", 'h8', 'h9', "h10"])
# html_df['id'] = html_files
# html_df.to_csv('html_features.csv', encoding='utf_8_sig', index=False)
#
# # 处理javascript
# jsc_features = []
# for i in html_files:
#     h = read_file(html_path + '\\' + i)
#     h_c = list(map(clean_file, h))
#     jsc_features.append(process_javascript(h_c))
# jsc_df = pd.DataFrame(jsc_features, columns=['j%d' % d for d in range(1, 19)])
# jsc_df['id'] = html_files
# jsc_df.to_csv('jsc_features.csv', encoding='utf_8_sig', index=False)

# 处理文本
# LSA MODEL - Vectorization of Documents
docs = []
for i in html_files:
    h = read_file(html_path + '\\' + i)
    h_c = list(map(clean_file, h))
    docs.append(extract_pure_text(h_c))


def LSA(dim, documents):
    DIM_CURRENT = dim
    documents_full = list(documents)
    texts_full = list(map(lambda x: jieba.lcut(x), documents_full))
    dictionary_full = corpora.Dictionary(texts_full)
    corpus_full = [dictionary_full.doc2bow(text) for text in texts_full]
    tfidf_model = models.TfidfModel(corpus_full)
    corpus_tfidf = tfidf_model[corpus_full]
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary_full, num_topics=DIM_CURRENT)
    corpus_lsi = lsi_model[corpus_tfidf]
    characteristic_vector = list(map(lambda x: list(map(lambda y: y[1], x)), corpus_lsi))
    return characteristic_vector


cvs = LSA(150, docs)  # 250 = 设置向量化后的维度
text_df = pd.DataFrame(cvs, columns=['w%d' % d for d in range(1, 151)])
text_df['id'] = html_files
text_df.to_csv('text_features.csv', index=False, encoding='utf_8_sig')

# 导入数据
url_df = pd.read_csv('join_acc.csv', encoding='utf_8_sig', engine='python')
url_feature = pd.read_csv('url_features.csv', encoding='utf_8_sig', engine='python')
html_feature = pd.read_csv('html_features.csv', encoding='utf_8_sig', engine='python')
jsc_feature = pd.read_csv('jsc_features.csv', encoding='utf_8_sig', engine='python')
text_feature = pd.read_csv('text_features.csv', encoding='utf_8_sig', engine='python')

# 按照id合并数据
full_df = pd.merge(url_df, url_feature, on='id', how='inner')
full_df = pd.merge(full_df, html_feature, on='id', how='inner')
full_df = pd.merge(full_df, jsc_feature, on='id', how='inner')
full_df = pd.merge(full_df, text_feature, on='id', how='inner')
full_df = full_df.drop(columns=['num_index', 'id', 'url']).dropna()
x_full = data_scaler(full_df.drop(columns=['label']), method='z-score').reset_index(drop=True)
y_full = full_df[['label']].reset_index(drop=True)
xy_full = pd.concat([x_full, y_full], axis=1)
xy_full = auto_dropper(xy_full)
x_full = xy_full.drop(columns=['label']).reset_index(drop=True)
y_full = xy_full[['label']].reset_index(drop=True)

# 训练模型
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.25, stratify=y_full)
m = SVC()
m.fit(x_train, y_train)
y_pred = m.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(precision_recall_fscore_support(y_test, y_pred))
