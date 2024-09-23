from transformers import AutoTokenizer, AutoModel,AutoConfig
import os
import torch
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,roc_auc_score
import re


# 验证accury等指标
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", config=config,trust_remote_code=True)
CHECKPOINT_PATH='ptuning/output/aiops-chatglm-6b-pt-128-2e-2-withcause_0909/checkpoint-1000'

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
rootcause_label=[]
rootcause_pred=[]
rootcuase_right_count=0
rootcuase_right_count2=0
rootcuase_count=0


# 按单独节点来计算指标
with open('ptuning/AIOPS/jianjie3_rootcause_analyze/dev_jianjie3_2_final.json','r') as f:
    for index,line in enumerate(tqdm(f.readlines())):
        
        line=line.strip()
        chat_input_dict=json.loads(line)
        label=chat_input_dict['summary']
        if '0' in label or ('1' in label and ',' not in label):
            labels_01.append(0)
            rootcause_label.append('normal')
        elif '1' in label:
            labels_01.append(1)
            rootcuase_count+=1
            if ',' in label:
                match=re.search(r'([a-zA-Z_]+)',label)
                if match:
                    metrics_label=match.group(1)
                    rootcause_label.append(metrics_label)
                else:
                    rootcause_label.append('yy')
            else:
                rootcause_label.append('xx')
        else:
            print('labels存在问题,不属于0,1的分类')
            labels_01.append(0)
        chat_input=chat_input_dict['content']
        response, history = model.chat(tokenizer, chat_input, history=[])
        # print(response[:15])
        if '1' not in response and '0' not in response:
            print(f'wrong index:{index},response error,没有获得正确的答案')
            pred.append(0)
            rootcause_pred.append('error1')
        elif '1' in response and '0' in response:
            print(f'wrong index:{index},1和0都存在于答案当中,因此答案存在一定的问题')
            pred.append(0)
            rootcause_pred.append('error2')
        # 将只有1的标签认定为0
        elif '0' in response or ('1' in response and '，' not in response):
            pred.append(0)
            rootcause_pred.append('normal')
        else:
            pred.append(1)
            match_response=re.search(r'([a-zA-Z_]+)',response)
            if match_response:
                metrics_response=match_response.group(1)
                rootcause_pred.append(metrics_response)
            else:
                rootcause_pred.append('norootcuase')
        if labels_01[-1]==1 and pred[-1]==1 and metrics_response==metrics_label:
            # 这个是来计算异常数据条件下的根因定位准确性
            rootcuase_right_count+=1
        if rootcause_pred[-1]==rootcause_label[-1]:
            # 这个是来计算所有数据下的根因定位准确性
            rootcuase_right_count2+=1
        print(f'response:{response}')
        if pred[-1]==0 and labels_01[-1]==1:
            print(f'index:{index+1}')
        if index>1000:
            break


# 按聚合节点来计算指标,感觉这里不需要，利用单独节点在result中记录的值可以直接来计算聚合的指标
# nodes_len=17
# label_group=0
# pred_group=0
# with open('ptuning/AIOPS/jianjie3/dev_jianjie3_2.json','r') as f:
#     for index,line in enumerate(tqdm(f.readlines())):
        
#         line=line.strip()
#         chat_input_dict=json.loads(line)
#         label=chat_input_dict['summary']
#         if '1' in label:
#             label_group=1
#         else:
#             if '0' not in label:
#                 print('labels存在问题,不属于0,1的分类')
# # chat_input='微服务中一个节点在某一个timestamp时刻,会存在一共124个metrics(指标)状态,例如CPU_Used_Pct,CPU_free_pct等。接下来我将给出每个节点在三个等间隔timestamp所形成的时间窗口下,所对应的每个timestamp相比于上一个timestamp存在变化的metrics变化值。第1408个时间窗口,共包含3个时刻:节点为os_018时,第2时刻相较于上一时刻,上涨指标及其对应的值为:Cache_used:1.22,Disk_svctm:11.21,Memory_free:1.07,下降指标及其对应的值为:Received_packets:-1.13,Sent_packets:-1.42。第3时刻相较于上一时刻,上涨指标及其对应的值为:CPU_iowait_time:7.47,Disk_avgqu_sz:4.64,Disk_await:1.33,Disk_io_util:1.01,Disk_wr_ios:12.50,下降指标及其对应的值为:Cache_used:-1.22,Disk_svctm:-6.11,Memory_free:-1.07。'
#         chat_input=chat_input_dict['content']
#         response, history = model.chat(tokenizer, chat_input, history=[])
#         print(response[:15])
#         if '1' not in response and '0' not in response:
#             print(f'wrong index:{index},response error,没有获得正确的答案')
#             pred.append(0)
#         elif '1' in response and '0' in response:
#             print(f'wrong index:{index},1和0都存在于答案当中,因此答案存在一定的问题')
#             pred.append(0)
#         elif '1' in response:
#             pred_group=1
#         print(f'response:{response}')
#         if index%nodes_len==0:
#             labels_01.append(label_group)
#             pred.append(pred_group)
#             label_group=0
#             pred_group=0
#         if index>34:
#             break



accuracy=accuracy_score(labels_01,pred)
precision=precision_score(labels_01,pred)
recall=recall_score(labels_01,pred)
report=classification_report(labels_01,pred)
rootcause_acc=f'{rootcuase_right_count/rootcuase_count:.2f}'
rootcause_acc2=f'{rootcuase_right_count2/len(rootcause_label):.2f}'

with open('result.txt','w') as f:
    f.write(f'accuracy:{accuracy:.2f}\n')
    f.write(f'precision:{precision:.2f}\n')
    f.write(f'recall:{recall:.2f}\n')
    f.write(f'labels:{",".join(list(map(str,labels_01)))}\n')
    f.write(f'pred:{",".join(list(map(str,pred)))}\n')
    f.write(f'rootcuase_label:{",".join(rootcause_label)}\n')
    f.write(f'rootcuase_pred:{",".join(rootcause_pred)}\n')
    f.write(f'rootcause_acc:{rootcause_acc}\n')
    f.write(f'rootcause_acc2:{rootcause_acc2}')

print(f'accuracy:{accuracy:.2f}')
print(f'precision:{precision:.2f}')
print(f'recall:{recall:.2f}')
print('report:')
print(report)
print(f'rootcuase_label:{rootcause_label}')
print(f'rootcuase_pred:{rootcause_pred}')
print(f'rootcause_acc:{rootcause_acc}')
print(f'rootcuase_right_count:{rootcuase_right_count}')
print(f'rootcuase_count:{rootcuase_count}')
print(f'rootcause_acc2:{rootcause_acc2}')



# ------------------------
# 原始输入
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# chat_input='{"content": "微服务中一个节点在某一个timestamp时刻,会存在一共124个metrics(指标)状态,例如CPU_Used_Pct,CPU_free_pct等。接下来我将给出每个节点在三个等间隔timestamp所形成的时间窗口下,所对应的每个timestamp相比于上一个timestamp存在变化的metrics变化值。第624个时间窗口,共包含3个时刻:节点为os_021时,第2时刻相较于上一时刻,上涨指标及其对应的值为:CPU_iowait_time:3.51,CPU_util_pct:1.02,Disk_avgqu_sz:2.19,Disk_await:4.76,cost:1.79,下降指标及其对应的值为:无较大变化。第3时刻相较于上一时刻,上涨指标及其对应的值为:CPU_iowait_time:15.74,Disk_avgqu_sz:16.56,Disk_io_util:1.50,Disk_svctm:131.00,下降指标及其对应的值为:CPU_user_time:-1.09,Disk_await:-4.76。'


# model=model.quantize(4).half().cuda()
# model = model.eval()
# response, history = model.chat(tokenizer, chat_input, history=[])
# print(response)
# ------------------------


# ------------------------
# 推理
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, pre_seq_len=128)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", config=config,trust_remote_code=True)
# CHECKPOINT_PATH='ptuning/output/aiops-chatglm-6b-pt-128-2e-2-RCA_0909/checkpoint-1000'

# prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
# new_prefix_state_dict = {}
# for k, v in prefix_state_dict.items():
#     if k.startswith("transformer.prefix_encoder."):
#         new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
# model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)


# model=model.quantize(4).half().cuda()
# model.transformer.prefix_encoder.float()
# model = model.eval()
# # chat_input='微服务中一个节点在某一个timestamp时刻,会存在一共124个metrics(指标)状态,例如CPU_Used_Pct,CPU_free_pct等。接下来我将给出某个节点在6个等间隔timestamp所形成的时间窗口下,所对应的每个timestamp相比于上一个timestamp存在变化的metrics变化值。在时间窗口1591452600,共包含6个时刻:节点为db_003时,第2时刻相较于上一时刻,上涨指标及其对应的值为:New_Tbs_Free_Gb:1.06,下降指标及其对应的值为:PGA_used_total:-1.05。第3时刻相较于上一时刻,上涨指标及其对应的值为:PGA_used_total:1.05,Physical_Read_Per_Sec:4.08,下降指标及其对应的值为:New_Tbs_Free_Gb:-1.06。第4时刻相较于上一时刻,上涨指标及其对应的值为:无较大变化,下降指标及其对应的值为:PGA_used_total:-1.05,Physical_Read_Per_Sec:-3.78。第5时刻相较于上一时刻,上涨指标及其对应的值为:PGA_used_total:1.05,下降指标及其对应的值为:无较大变化。第6时刻相较于上一时刻,上涨指标及其对应的值为:Sess_Connect:1.24,tnsping_result_time:9999.90,下降指标及其对应的值为:DFParaWrite_Per_Sec:-1.05,LFParaWrite_Per_Sec:-1.10,LFSync_Per_Sec:-1.28,On_Off_State:-10.00,PGA_used_total:-1.05,Physical_Read_Per_Sec:-30.61,SctRead_Per_Sec:-5.99,SeqRead_Per_Sec:-1.15,Session_pct:-1.11。'
# chat_input='节点为db_003时,第2时刻相较于上一时刻,上涨指标及其对应的值为Physical_Read_Per_Sec30.66,下降指标及其对应的值为Sess_Connect-3.10。第3时刻相较于上一时刻,上涨指标及其对应的值为Login_Per_Sec5.84,下降指标及其对应的值为无较大变化。第4时刻相较于上一时刻,上涨指标及其对应的值为无较大变化,下降指标及其对应的值为Login_Per_Sec-5.00。第5时刻相较于上一时刻,上涨指标及其对应的值为无较大变化,下降指标及其对应的值为count-2.46。第6时刻相较于上一时刻,上涨指标及其对应的值为Login_Per_Sec2.17,下降指标及其对应的值为无较大变化。'
# response, history = model.chat(tokenizer, chat_input, history=[])
# print(response)
# chat_input='节点为db_007时,第2时刻相较于上一时刻,无较大变化。第3时刻相较于上一时刻,无较大变化。第4时刻相较于上一时刻,无较大变化。第5时刻相较于上一时刻,上涨指标及其对应的值为Redo_Per_Sec150.95,User_Commit3.86,下降指标及其对应的值为无较大变化。第6时刻相较于上一时刻,上涨指标及其对应的值为无较大变化,下降指标及其对应的值为Redo_Per_Sec-145.62,User_Commit-3.10。'
# response, history = model.chat(tokenizer, chat_input, history=[])
# print(response)

# ------------------------