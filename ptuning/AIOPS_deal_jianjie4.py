import math
import os
import json
import numpy as np

timestamp=np.load('AIOPS/npy4/real_timestamp_4.npy') # (2220,)
labels_timestamp=np.load('AIOPS/npy4/real_label_4.npy') # (2220,)
labels_node=np.load('AIOPS/npy4/labels_nodes_4.npy')# (2220, 17)
nodes=np.load('AIOPS/npy4/real_node_4.npy') # (2220, 17, 79, 6)
adjcent=np.load('AIOPS/total_A.npy') # (17,17)

nodes=nodes.transpose(0,1,3,2) # (2220, 17, 6, 79)

input_dict_file='AIOPS/node_dict.json'
with open(input_dict_file, 'r') as f:
    input_dict = json.load(f)

value_name_dict={}
for name,value in input_dict.items():
    value_name_dict[value]=name

print(labels_timestamp.shape)
print(labels_node.shape)
print(timestamp.shape)
print(nodes.shape) # (2220, 17, 6, 79)




system_prompt="""你是一位优秀的网络运维工程师,能够根据一定时间步下的网络节点的metrics(指标)情况以及节点之间
的邻接关系判断网络中是否存在异常。现在我将要给出这个微服务中的节点之间的邻接关系以及每一个节点在一定时间步下的metric(指标)情况，
请你输出此时网络中是否存在异常,存在异常为1,不存在异常为0,你只需要输出0或者1即可。
"""

node_nums=len(adjcent)
metrics_name='ACS,AWS,Asm_Free_Tb,Buffers_used,CPU_free_pct,CPU_idle_pct,CPU_iowait_time,CPU_kernel_number,CPU_number,CPU_pused,CPU_util_pct,Cache_used,Call_Per_Sec,DbFile_Used_Pct,DbTime,Disk_avgqu_sz,Disk_rd_ios,Disk_rd_kbs,FS_max_avail,FS_max_util,FS_total_space,FS_used_pct,FS_used_space,Free_disk_space,Free_inodes,ICMP_ping,Logic_Read_Per_Sec,Login_Per_Sec,MEM_Total,MEM_Used,MEM_Used_Pct,MEM_real_util,Memory_available,Memory_available_pct,Memory_free,Memory_total,Memory_used,Memory_used_pct,New_Tbs_Free_Gb,New_Tbs_Used_Pct,Num_of_processes,Num_of_running_processes,On_Off_State,Outgoing_network_traffic,PGA_used_total,Page_po,Physical_Read_Per_Sec,Proc_Used_Pct,Proc_User_Used_Pct,Processor_load_15_min,Processor_load_1_min,Processor_load_5_min,Received_packets,SctRead_Per_Sec,Sent_errors_packets,Sent_packets,Sent_queue,SeqRead_Per_Sec,Sess_Active,Sess_Connect,Session_pct,Shared_memory,System_block_queue_length,System_wait_queue_length,TPS_Per_Sec,Tbs_Free_Gb,Used_inodes_pct,Zombie_Process,container_cpu_used,container_mem_used,cost,count,proc,ss_total,succ_rate,tnsping_result_time'
metric_name_list=metrics_name.split(',')
# prompt_prefix=f'现在微服务中总共有{node_nums}个节点,其中每个节点在某一个timestamp时刻,会存在一共{len(metric_name_list)}个metrics(指标)状态。'
prompt_prefix=f'微服务中一个节点在某一个timestamp时刻,会存在一共{len(metric_name_list)}个metrics(指标)状态,例如CPU_Used_Pct,CPU_free_pct等。'


# prompt_metric=f"""我们将其记录为metric0,metric1,...一直到metric{len(metric_name_list)-1}。这些metric可能为例如Buffers_used,CPU_free_pct等,值为保留两位小数后的浮点数。接下来我将给出每个不同节点在三个等间隔timestamp所形成的时间窗口下,所对应的每个timestamp相比于上一个timestamp的metrics变化值,如果没有指出某个metric则说明这个metric在这个timestamp中相比于上一个timestamp没有变化,我们将用这个时间窗口的metric变化情况推断该窗口下是否存在异常。"""
prompt_metric=f"""接下来我将给出某个节点在6个等间隔timestamp所形成的时间窗口下,所对应的每个timestamp相比于上一个timestamp存在变化的metrics变化值。"""
prompt_prefix+=prompt_metric
# print(prompt_prefix)

nodes_prompt_list=[]
for index,(time,node_metric_timerange,label) in enumerate(list(zip(timestamp,nodes,labels_node))): # nodes的shape为 (1534, 17, 6, 124),labels_node的shape为（1534，17）
    prompt_element=''
    prompt1=f'在时间窗口{time},共包含{len(node_metric_timerange[0])}个时刻:'
    prompt_element+=prompt1
    for i in range(len(node_metric_timerange)):
        node_name=value_name_dict[i]
        prompt23=''
        prompt2=f'节点为{node_name}时,'
        prompt23+=prompt2
        # prompt_element+=prompt2
        for j in range(len(node_metric_timerange[i])):
            # 跳过第一个
            if j==0:
                continue
        
            metrics_values_list=node_metric_timerange[i][j]
            metrics_values_list=np.round(metrics_values_list,2)
            metrics_last_list=node_metric_timerange[i][j-1]
            metrics_last_list=np.round(metrics_last_list,2)
            # prompt3=f'在timestamp为{timestamp_now}下相较于{timestamp_now-60},'
            prompt3=f'第{j+1}时刻相较于上一时刻,'
            flag=0
            up_list=[]
            down_list=[]
            for index,(metric_value_now,metric_value_last) in enumerate(zip(metrics_values_list,metrics_last_list)):
                # 目前这版差值要大于2.0才进行统计
                if math.fabs(metric_value_last-metric_value_now)>2.0:
                    flag=1
                    diff=metric_value_now-metric_value_last
                    if diff>0:
                        # prompt_diff=f'{metric_name_list[index]}上涨了{diff:.2f},'
                        # prompt_diff=f'metric{index}上涨了{diff:.2f},'
                        up_list.append(f'{metric_name_list[index]}:{diff:.2f}')
                    else:
                        # prompt_diff=f'{metric_name_list[index]}下降了{-diff:.2f},'
                        # prompt_diff=f'metric{index}下降了{-diff:.2f},'
                        down_list.append(f'{metric_name_list[index]}:{diff:.2f}')
            
            if flag==0:
                prompt3+='无较大变化。'
            else:
                up_value=",".join(up_list) if len(up_list)>0 else '无较大变化'
                down_value=",".join(down_list) if len(down_list)>0 else '无较大变化'
                prompt_diff=f'上涨指标及其对应的值为:{up_value},下降指标及其对应的值为:{down_value}。'
                prompt3+=prompt_diff
            prompt23+=prompt3

        content_summary_dict={}
        content_summary_dict['content']=prompt_prefix+prompt_element+prompt23
        summary='该节点存在异常(1)' if label[i]==1 else '该节点为正常状态(0)'

        content_summary_dict['summary']=summary
        # if label[i]==1:
        #     print(content_summary_dict['content'])
        #     print(summary)
        nodes_prompt_list.append(content_summary_dict)



# np.random.seed(42)
# np.random.shuffle(nodes_prompt_list)
print(len(nodes_prompt_list))
train_num_split_point=int(len(timestamp)*0.8*len(value_name_dict))

train_nodes_prompt_list=nodes_prompt_list[:train_num_split_point]
print(len(train_nodes_prompt_list))
dev_nodse_prompt_list=nodes_prompt_list[train_num_split_point:]
print(len(dev_nodse_prompt_list))

# 减少一些训练集中正常样本数量，维持训练集中normal:abnormal的比例为1:5
label_0_list=[]
label_1_list=[]
for csd in train_nodes_prompt_list:
    lb=csd['summary']
    # 去除掉prompt的前面的冗余部分
    csd['content']=csd['content'][csd['content'].index(':')+1:]
    if '0' in lb:
        label_0_list.append(csd)
    elif '1' in lb:
        label_1_list.append(csd)
    else:
        print('error happend')
normal_should_count=len(label_1_list)*8
label_0_list=label_0_list[:normal_should_count]
train_nodes_prompt_list=np.array(label_1_list+label_0_list)
np.random.seed(42)
np.random.shuffle(train_nodes_prompt_list)
print(len(train_nodes_prompt_list))


with open('AIOPS/jianjie4/train_jianjie4.json','w',encoding='utf-8') as f:
    for prompt_finish in train_nodes_prompt_list:
        f.write(json.dumps(prompt_finish,ensure_ascii=False)+'\n')

for csd in dev_nodse_prompt_list:
    # 去除掉prompt的前面的冗余部分
    csd['content']=csd['content'][csd['content'].index(':')+1:]

with open('AIOPS/jianjie4/dev_jianjie4.json','w',encoding='utf-8') as f:
    for prompt_finish in dev_nodse_prompt_list:
        f.write(json.dumps(prompt_finish,ensure_ascii=False)+'\n')



# with open('AIOPS/jianjie3/content_example_jianjie3.txt','w',encoding='utf-8') as f:
#     f.write(nodes_prompt_list[0]['content'])



