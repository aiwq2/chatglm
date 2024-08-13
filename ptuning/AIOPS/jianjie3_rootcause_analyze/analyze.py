

import os
import json
import re
import math

# 分别找出dev_jianjie3_2.json和train_jianjie3.json中的异常数据，便于添加根因分析的标签

# dev_jianjie3_2_new=[]
# train_jianjie3_new=[]
# with open('dev_jianjie3_2.json','r',encoding='utf-8') as f:
#     for line in f.readlines():
#         line=line.strip()
#         ob=json.loads(line)
#         if '1' in ob['summary']:
#             dev_jianjie3_2_new.append(ob)
# with open('train_jianjie3.json','r',encoding='utf-8') as f:
#     for line in f.readlines():
#         line=line.strip()
#         ob=json.loads(line)
#         if '1' in ob['summary']:
#             train_jianjie3_new.append(ob)
# with open('dev_jianjie3_2_rca.json','w',encoding='utf-8') as f:
#     for dev in dev_jianjie3_2_new:
#         f.write(json.dumps(dev,ensure_ascii=False)+'\n')
# with open('train_jianjie3_rca.json','w',encoding='utf-8') as f:
#     for tr in train_jianjie3_new:
#         f.write(json.dumps(tr,ensure_ascii=False)+'\n')

# ------------------------------------------------------------------------------------------------------------------
# 根据指标中的异常偏差值来生成rca的可能标签并写入一个新的文件中

new_content=[]
with open('train_jianjie3_rca.json','r',encoding='utf-8') as f:
    for index,line in enumerate(f.readlines()):
        metrics_value_list=[]
        line=line.strip()
        ob=json.loads(line)
        content=ob['content']
        if '1' in ob['summary']:
            content_list=re.split(r'[,。]',content)
            for ct in content_list:
                ct=ct.strip()
                if ct:
                    match=re.search(r'([a-zA-Z_]+)(-?\d+\.\d+|-?\d+)$',ct)
                    if match:
                        metrics_value_list.append((match.group(2),match.group(1)))
            if metrics_value_list:
                metrics_value_list_sorted=sorted(metrics_value_list,key=lambda x:math.fabs(float(x[0])),reverse=True)
                ob['summary']=ob['summary']+f',可能的原因为指标{metrics_value_list_sorted[0][1]}的大幅波动'
        new_content.append(ob)
with open('train_jianjie3_rca_final.json','w',encoding='utf-8') as f:
    for ob in new_content:
        f.write(json.dumps(ob,ensure_ascii=False)+'\n')

