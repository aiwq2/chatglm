from cProfile import label
from cgi import print_arguments
from doctest import master
import enum
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix

# count_1=0
# count_0=0
# label_1_list=[]
# with open('train_jianjie3.json','r') as f:
#     for tt in f.readlines():
#         tt=tt.strip()
#         tt_dict=json.loads(tt)
#         label=tt_dict['summary']
#         tt_dict['content']=''.join(tt_dict['content'].split(':')[1:])
#         if '1' in label:
#             count_1+=1
#             label_1_list.append(tt_dict)
#         elif '0' in label:
#             count_0+=1
#         else:
#             print('label error')
# print('count1:',count_1)
# print('count0:',count_0)
# with open('temp.json','w') as f:
#     for t in label_1_list:
#         f.write(json.dumps(t,ensure_ascii=False)+'\n')

# ---------------------------------------------------------
# 看一下result.txt中那些有问题的样本，这里用代码先做一下筛选
# labels=[]
# preds=[]
# with open('result.txt','r') as f:
#     for line in f.readlines():
#         line=line.strip()
#         if line.startswith('labels'):
#             labels.extend(line.split(':')[1].split(','))
#         if line.startswith('pred'):
#             preds.extend(line.split(':')[1].split(','))
# for index,(lb,pre) in enumerate(zip(labels,preds)):
#     if lb!=pre:
#         print(f'line{index+1},label:{lb},prediction:{pre}')

# ---------------------------------------------------------

# ---------------------------------------------------------
# 根据人工审查的label对于原来本身的label进行修改,并且计算单节点和多节点聚合的共同的指标
# labels=[]
# preds=[]
# labels_0to1_index=[]
# labels_1to0_index=[]
# with open('result.txt','r') as f:
#     for line in f.readlines():
#         line=line.strip()
#         if line.startswith('labels'):
#             labels.extend(line.split(':')[1].split(','))
#         if line.startswith('pred'):
#             preds.extend(line.split(':')[1].split(','))
#         if line.startswith('0to1'):
#             labels_0to1_index.extend(line.split(':')[1].split(','))
#         if line.startswith('1to0'):
#             labels_1to0_index.extend(line.split(':')[1].split(','))
# labels_1to0_index=list(map(int,labels_1to0_index))
# labels_0to1_index=list(map(int,labels_0to1_index))
# labels=list(map(int,labels))
# preds=list(map(int,preds))
# for index,(lb,pre) in enumerate(zip(labels,preds)):
#     if index+1 in labels_0to1_index:
#         if labels[index]==1:
#             print('label already 1')
#         labels[index]=1
#     if index+1 in labels_1to0_index:
#         if labels[index]==0:
#             print('label already 0')
#         labels[index]=0

# accuracy=accuracy_score(labels,preds)
# precision=precision_score(labels,preds)
# recall=recall_score(labels,preds)
# report=classification_report(labels,preds)
# matrix=confusion_matrix(labels,preds)

# print('single node:')
# print(f'accuracy:{accuracy:.2f}')
# print(f'precision:{precision:.2f}')
# print(f'recall:{recall:.2f}')
# print('report:')
# print(report)
# print(matrix.T)

# labels_group=[]
# preds_group=[]
# nodes_len=17
# for index,(lb,pre) in enumerate(zip(labels,preds)):
#     if index+nodes_len>=len(labels):
#         break
#     if 1 in labels[index:index+nodes_len]:
#         labels_group.append(1)
#     else:
#         labels_group.append(0)
#     if 1 in preds[index:index+nodes_len]:
#         preds_group.append(1)
#     else:
#         preds_group.append(0)


# accuracy=accuracy_score(labels_group,preds_group)
# precision=precision_score(labels_group,preds_group)
# recall=recall_score(labels_group,preds_group)
# report=classification_report(labels_group,preds_group)

# print('group node:')
# print(f'accuracy:{accuracy:.2f}')
# print(f'precision:{precision:.2f}')
# print(f'recall:{recall:.2f}')
# print('report:')
# print(report)

# ---------------------------------------------------------

# ---------------------------------------------------------
# 用单个节点的label和pred的值来计算聚合后的所有节点的label和pred的值
labels=[]
preds=[]
with open('result.txt','r') as f:
    for line in f.readlines():
        line=line.strip()
        if line.startswith('bert_labels'):
            labels.extend(line.split(':')[1].split(','))
        if line.startswith('bert_pred'):
            preds.extend(line.split(':')[1].split(','))

labels_group=[]
preds_group=[]
nodes_len=17
for index,(lb,pre) in enumerate(zip(labels,preds)):
    if index+nodes_len>=len(labels):
        break
    if '1' in labels[index:index+nodes_len]:
        labels_group.append(1)
    else:
        labels_group.append(0)
    if '1' in preds[index:index+nodes_len]:
        preds_group.append(1)
    else:
        preds_group.append(0)


accuracy=accuracy_score(labels_group,preds_group)
precision=precision_score(labels_group,preds_group)
recall=recall_score(labels_group,preds_group)
report=classification_report(labels_group,preds_group)
matrix=confusion_matrix(labels_group,preds_group)

print(f'accuracy:{accuracy:.2f}')
print(f'precision:{precision:.2f}')
print(f'recall:{recall:.2f}')
print('report:')
print(report)
print(matrix.T)

# ---------------------------------------------------------

