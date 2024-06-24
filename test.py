from transformers import AutoTokenizer, AutoModel,AutoConfig
import os
import torch
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,roc_auc_score

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", config=config,trust_remote_code=True)
CHECKPOINT_PATH='ptuning/output/aiops-chatglm-6b-pt-128-2e-2/checkpoint-1000'

prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)


model=model.quantize(4).half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()
labels_01=[]
pred=[]
with open(r'D:\计算所研究生学习\毕业设计\renrui code\my_code\ChatGLM-6B\ptuning\AIOPS\jianjie3\dev_jianjie3.json','w') as f:
    for line in tqdm(f.readlines()):
        line=line.strip()
        chat_input_dict=json.loads(line)
        label=chat_input_dict['summary']
        if '0' in label:
            labels_01.append(0)
        elif '1' in label:
            labels_01.append(1)
        else:
            print('labels存在问题,不属于0,1的分类')
        # chat_input='微服务中一个节点在某一个timestamp时刻,会存在一共124个metrics(指标)状态,例如CPU_Used_Pct,CPU_free_pct等。接下来我将给出每个节点在三个等间隔timestamp所形成的时间窗口下,所对应的每个timestamp相比于上一个timestamp存在变化的metrics变化值。第1408个时间窗口,共包含3个时刻:节点为os_018时,第2时刻相较于上一时刻,上涨指标及其对应的值为:Cache_used:1.22,Disk_svctm:11.21,Memory_free:1.07,下降指标及其对应的值为:Received_packets:-1.13,Sent_packets:-1.42。第3时刻相较于上一时刻,上涨指标及其对应的值为:CPU_iowait_time:7.47,Disk_avgqu_sz:4.64,Disk_await:1.33,Disk_io_util:1.01,Disk_wr_ios:12.50,下降指标及其对应的值为:Cache_used:-1.22,Disk_svctm:-6.11,Memory_free:-1.07。'
        chat_input=chat_input_dict['content']
        response, history = model.chat(tokenizer, chat_input, history=[])
        if '1' not in response and '0' not in response:
            print('response error,没有获得正确的答案')
        elif '1' in response and '0' in response:
            print('1和0都存在于答案当中,因此答案存在一定的问题')
        elif '0' in response:
            pred.append(0)
        else:
            pred.append(1)


accuracy=accuracy_score(labels_01,pred)
precision=precision_score(labels_01,pred)
recall=recall_score(labels_01,pred)
report=classification_report(labels_01,pred)

with open('result.txt','r') as f:
    f.write(f'accuracy:{accuracy:.2f}\n')
    f.write(f'precision:{precision:.2f}\n')
    f.write(f'recall:{recall:.2f}\n')

print(f'accuracy:{accuracy:.2f}')
print(f'precision:{precision:.2f}')
print(f'recall:{recall:.2f}')
print('report:')
print(report)




# 原始输入
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# chat_input='{"content": "微服务中一个节点在某一个timestamp时刻,会存在一共124个metrics(指标)状态,例如CPU_Used_Pct,CPU_free_pct等。接下来我将给出每个节点在三个等间隔timestamp所形成的时间窗口下,所对应的每个timestamp相比于上一个timestamp存在变化的metrics变化值。第624个时间窗口,共包含3个时刻:节点为os_021时,第2时刻相较于上一时刻,上涨指标及其对应的值为:CPU_iowait_time:3.51,CPU_util_pct:1.02,Disk_avgqu_sz:2.19,Disk_await:4.76,cost:1.79,下降指标及其对应的值为:无较大变化。第3时刻相较于上一时刻,上涨指标及其对应的值为:CPU_iowait_time:15.74,Disk_avgqu_sz:16.56,Disk_io_util:1.50,Disk_svctm:131.00,下降指标及其对应的值为:CPU_user_time:-1.09,Disk_await:-4.76。'


# model=model.quantize(4).half().cuda()
# model = model.eval()
# response, history = model.chat(tokenizer, chat_input, history=[])
# print(response)