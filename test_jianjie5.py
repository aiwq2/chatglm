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
CHECKPOINT_PATH='ptuning/output/aiops-chatglm-6b-pt-128-2e-2-jianjie5_0921/checkpoint-500'

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

abnoramlNode_label=[]
abnoramlNode_pred=[]
abnoramlNode_right_count=0
abnoramlNode_right_count2=0
abnoramlNode_count=0

rootcause_label=[]
rootcause_pred=[]
rootcuase_right_count=0
rootcuase_right_count2=0
rootcuase_count=0





# 按单独节点来计算指标
with open('ptuning/AIOPS/jianjie5/dev_jianjie5_final_rewrite.json','r') as f:
    for index,line in enumerate(tqdm(f.readlines())):
        
        line=line.strip()
        chat_input_dict=json.loads(line)
        label=chat_input_dict['summary']
        if '(0)' in label or ('1' in label and ',' not in label):
            labels_01.append(0)
            rootcause_label.append('normal')
            abnoramlNode_label.append('normal')
        elif '(1)' in label:
            labels_01.append(1)
            rootcuase_count+=1
            abnoramlNode_count+=1
            if ',' in label:
                label_node,label_rootcause=label.split(',')[1],label.split(',')[2]
                # 异常节点
                match_node=re.search(r'([a-zA-Z_0-9]+)',label_node)
                if match_node:
                    node_label=match_node.group(1)
                    abnoramlNode_label.append(node_label)
                else:
                    abnoramlNode_label.append('zz')
                # 根因分析
                match_rootcause=re.search(r'([a-zA-Z_]+)',label_rootcause)
                if match_rootcause:
                    metrics_label=match_rootcause.group(1)
                    rootcause_label.append(metrics_label)
                else:
                    rootcause_label.append('yy')
            else:
                rootcause_label.append('xx')
        else:
            print('labels存在问题,不属于0,1的分类')
            labels_01.append(0)
        # print('rootcause_label:',rootcause_label[-1])
        # print('abnoramlNode_label:',abnoramlNode_label[-1])
        chat_input=chat_input_dict['content']
        response, history = model.chat(tokenizer, chat_input, history=[])
        # print(response[:15])
        # 将只有1的标签认定为0
        if '(0)' in response or ('(1)' in response and '，' not in response):
            pred.append(0)
            rootcause_pred.append('normal')
            abnoramlNode_pred.append('normal')
        else:
            pred.append(1)
            response_node,response_rootcause=response.split('，')[1],response.split('，')[2]

            match_response_node=re.search(r'([a-zA-Z_0-9]+)',response_node)
            if match_response_node:
                response_node=match_response_node.group(1)
                abnoramlNode_pred.append(response_node)
            else:
                abnoramlNode_pred.append('norootcuase')

            
            match_response_rootase=re.search(r'([a-zA-Z_]+)',response_rootcause)
            if match_response_rootase:
                response_rootcause=match_response_rootase.group(1)
                rootcause_pred.append(response_rootcause)
            else:
                rootcause_pred.append('norootcuase')
        # print('rootcause_pred:',rootcause_pred[-1])
        # print('abnoramlNode_pred:',abnoramlNode_pred[-1])
        if labels_01[-1]==1 and pred[-1]==1 and response_rootcause==metrics_label:
            # 这个是来计算所有数据下的故障定位准确性
            rootcuase_right_count+=1
        if labels_01[-1]==1 and pred[-1]==1 and response_node==node_label:
            # 这个是来计算异常数据条件下的故障定位准确性
            abnoramlNode_right_count+=1
        if rootcause_pred[-1]==rootcause_label[-1]:
            # 这个是来计算所有数据下的根因定位准确性
            rootcuase_right_count2+=1
        if abnoramlNode_pred[-1]==abnoramlNode_label[-1]:
            # 这个是来计算所有数据下的根因定位准确性
            abnoramlNode_right_count2+=1
        print(f'response:{response}')
        if index>1000:
            break





accuracy=accuracy_score(labels_01,pred)
precision=precision_score(labels_01,pred)
recall=recall_score(labels_01,pred)
report=classification_report(labels_01,pred)
rootcause_acc=f'{rootcuase_right_count/rootcuase_count:.2f}'
rootcause_acc2=f'{rootcuase_right_count2/len(rootcause_label):.2f}'
node_acc=f'{abnoramlNode_right_count/abnoramlNode_count:.2f}'
node_acc2=f'{abnoramlNode_right_count2/len(abnoramlNode_label):.2f}'

with open('result.txt','w') as f:
    f.write(f'accuracy:{accuracy:.2f}\n')
    f.write(f'precision:{precision:.2f}\n')
    f.write(f'recall:{recall:.2f}\n')
    f.write(f'labels:{",".join(list(map(str,labels_01)))}\n')
    f.write(f'pred:{",".join(list(map(str,pred)))}\n')
    f.write(f'rootcuase_label:{",".join(rootcause_label)}\n')
    f.write(f'rootcuase_pred:{",".join(rootcause_pred)}\n')
    f.write(f'rootcause_acc:{rootcause_acc}\n')
    f.write(f'rootcause_acc2:{rootcause_acc2}\n')
    f.write(f'abnoramlNode_label:{abnoramlNode_label}\n')
    f.write(f'abnoramlNode_pred:{abnoramlNode_pred}\n')
    f.write(f'node_acc:{node_acc}\n')
    f.write(f'node_acc2:{node_acc2}\n')

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

print(f'abnoramlNode_label:{abnoramlNode_label}')
print(f'abnoramlNode_pred:{abnoramlNode_pred}')
print(f'node_acc:{node_acc}')
print(f'abnoramlNode_right_count:{abnoramlNode_right_count}')
print(f'abnoramlNode_count:{abnoramlNode_count}')
print(f'node_acc2:{node_acc2}')



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
# CHECKPOINT_PATH='ptuning/output/aiops-chatglm-6b-pt-128-2e-2-jianjie5_0921/checkpoint-500'

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
# chat_input='各节点变化极小。'
# response, history = model.chat(tokenizer, chat_input, history=[])
# print(response)
# chat_input='节点为db_003时,第2时刻变化值Physical_Read_Per_Sec32.04。节点为db_009时,第2时刻变化值Physical_Read_Per_Sec12.43,Sess_Used_Temp13.00。第3时刻变化值Physical_Read_Per_Sec24.41,Sess_Used_Temp-13.00。第4时刻变化值Sess_Used_Temp13.00,Physical_Read_Per_Sec-36.84。第5时刻变化值Sess_Used_Temp-13.00。第6时刻变化值Sess_Used_Temp13.00。节点为os_017时,第4时刻变化值Disk_await23.05。第5时刻变化值Incoming_network_traffic77.64,Disk_await-22.55。第6时刻变化值Incoming_network_traffic-77.64。节点为os_018时,第5时刻变化值Page_pi1977.28。第6时刻变化值Page_pi-1976.34。节点为os_019时,第5时刻变化值Incoming_network_traffic155.82。第6时刻变化值CPU_iowait_time11.14,Recv_total12.33,Send_total13.29,Incoming_network_traffic-98.82。节点为os_022时,第3时刻变化值cost12.06。第5时刻变化值Received_queue32.00,cost-13.96。第6时刻变化值Received_queue-32.00。'
# response, history = model.chat(tokenizer, chat_input, history=[])
# print(response)
# chat_input='节点为db_009时,第2时刻变化值Physical_Read_Per_Sec24.41,Sess_Used_Temp-13.00。第3时刻变化值Sess_Used_Temp13.00,Physical_Read_Per_Sec-36.84。第4时刻变化值Sess_Used_Temp-13.00。第5时刻变化值Sess_Used_Temp13.00。第6时刻变化值Sess_Used_Temp-13.00。节点为os_017时,第3时刻变化值Disk_await23.05。第4时刻变化值Incoming_network_traffic77.64,Disk_await-22.55。第5时刻变化值Incoming_network_traffic-77.64。节点为os_018时,第4时刻变化值Page_pi1977.28。第5时刻变化值Page_pi-1976.34。节点为os_019时,第4时刻变化值Incoming_network_traffic155.82。第5时刻变化值CPU_iowait_time11.14,Recv_total12.33,Send_total13.29,Incoming_network_traffic-98.82。第6时刻变化值Incoming_network_traffic-59.75,Recv_total-12.15,Send_total-13.09。节点为os_021时,第6时刻变化值cost117.74。节点为os_022时,第2时刻变化值cost12.06。第4时刻变化值Received_queue32.00,cost-13.96。第5时刻变化值Received_queue-32.00。第6时刻变化值cost48.56。'
# response, history = model.chat(tokenizer, chat_input, history=[])
# print(response)

# ------------------------