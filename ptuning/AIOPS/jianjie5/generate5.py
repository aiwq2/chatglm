import os
import json
import re
import math
import numpy as np

# 分别找出dev_jianjie3_2.json和train_jianjie3.json中的异常数据，便于添加根因分析的标签
# 借用的jianjie3_rootcause_analyze里面的generate.py，有些变量名没修改

# dev_jianjie3_2_new=[]
# train_jianjie3_new=[]
# with open('dev_jianjie5.json','r',encoding='utf-8') as f:
#     for line in f.readlines():
#         line=line.strip()
#         ob=json.loads(line)
#         if '1' in ob['summary']:
#             dev_jianjie3_2_new.append(ob)
# with open('train_jianjie5.json','r',encoding='utf-8') as f:
#     for line in f.readlines():
#         line=line.strip()
#         ob=json.loads(line)
#         if '1' in ob['summary']:
#             train_jianjie3_new.append(ob)
# with open('dev_jianjie5_rca.json','w',encoding='utf-8') as f:
#     for dev in dev_jianjie3_2_new:
#         f.write(json.dumps(dev,ensure_ascii=False)+'\n')
# with open('train_jianjie5_rca.json','w',encoding='utf-8') as f:
#     for tr in train_jianjie3_new:
#         f.write(json.dumps(tr,ensure_ascii=False)+'\n')

# ------------------------------------------------------------------------------------------------------------------
# 根据指标中的异常偏差值来生成rca的可能标签并写入一个新的文件中,rca_final,rca,rewrite这些文件都可以通过这部分代码来生成

new_content=[]
with open('dev_jianjie5.json','r',encoding='utf-8') as f:
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
                max_change_metrics=metrics_value_list_sorted[0][1]
                max_chage_value=metrics_value_list_sorted[0][0]
                value_index=content.find(max_chage_value)
                node_pattern=r"为([a-zA-Z0-9_]+)时"
                match = re.findall(node_pattern, content[:value_index])
                node=''
                if match:
                    node = match[-1]
                    print(f"找到的node为: {node}")
                else:
                    print("没有找到 '为' 和 '时' 之间的单词")
                ob['summary']=ob['summary']+f',异常节点为{node},可能的原因为指标{max_change_metrics}的大幅波动'
        if content=='各节点变化极小':
            ob['summary']='该节点为正常状态(0)'
        new_content.append(ob)

# 这是在训练集中的rewrite中进行构造的代码，因为现在训练集中没有这部分代码，验证集不需要
new_content=new_content*3
# ob_gen=dict()
# ob_gen['content']='各节点变化极小'
# ob_gen['summary']='该节点为正常状态(0)'
# new_content=new_content+[ob_gen]*50
# np.random.seed(42)
# np.random.shuffle(new_content)


with open('dev_jianjie5_final_rewrite.json','w',encoding='utf-8') as f:
    for ob in new_content:
        f.write(json.dumps(ob,ensure_ascii=False)+'\n')

# ------------------------------------------------------------------------------------------------------------------
# new=[]
# with open('dev_jianjie3_2_final.json','r') as f:
#     for line in f.readlines():
#         line=line.strip()
#         ob=json.loads(line)
#         if '1' in ob['summary']:
#             new.append(ob)

# with open('temp.json','w') as f:
#     for ob in new:
#         f.write(json.dumps(ob,ensure_ascii=False)+'\n')