import math
import os
import json
import numpy as np

timestamp=np.load(r'D:\计算所研究生学习\毕业设计\renrui code\my_code\ChatGLM-6B\ptuning\AIOPS\real_timestamp.npy') # (1537,)
labels=np.load(r'D:\计算所研究生学习\毕业设计\renrui code\my_code\ChatGLM-6B\ptuning\AIOPS\real_label.npy') # (1537,)
nodes=np.load(r'D:\计算所研究生学习\毕业设计\renrui code\my_code\ChatGLM-6B\ptuning\AIOPS\real_node.npy') # (1537, 17, 124, 3)
adjcent=np.load(r'D:\计算所研究生学习\毕业设计\renrui code\my_code\ChatGLM-6B\ptuning\AIOPS\total_A.npy') # (17,17)

nodes=nodes.transpose(0,1,3,2) # (1537, 17, 3, 124)

input_dict_file=r'D:\计算所研究生学习\毕业设计\renrui code\my_code\ChatGLM-6B\ptuning\AIOPS\node_dict.json'
with open(input_dict_file, 'r') as f:
    input_dict = json.load(f)

value_name_dict={}
for name,value in input_dict.items():
    value_name_dict[value]=name



system_prompt="""你是一位优秀的网络运维工程师,能够根据一定时间步下的网络节点的metrics(指标)情况以及节点之间
的邻接关系判断网络中是否存在异常。现在我将要给出这个微服务中的节点之间的邻接关系以及每一个节点在一定时间步下的metric(指标)情况，
请你输出此时网络中是否存在异常,存在异常为1,不存在异常为0,你只需要输出0或者1即可。
"""

node_nums=len(adjcent)
metrics_name='ACS,AIOS,AWS,Agent_ping,Asm_Free_Tb,Buffers_used,CPU_Used_Pct,CPU_free_pct,CPU_frequency,CPU_idle_pct,CPU_iowait_time,CPU_kernel_number,CPU_number,CPU_pused,CPU_system_time,CPU_user_time,CPU_util_pct,Cache_used,Call_Per_Sec,Cpu_num,DFParaWrite_Per_Sec,DbFile_Used_Pct,DbTime,Disk_avgqu_sz,Disk_await,Disk_io_util,Disk_rd_ios,Disk_rd_kbs,Disk_svctm,Disk_wr_ios,Disk_wr_kbs,Exec_Per_Sec,FS_max_avail,FS_max_util,FS_total_space,FS_used_pct,FS_used_space,Free_disk_space,Free_inodes,Hang,ICMP_ping,Incoming_network_traffic,LFParaWrite_Per_Sec,LFSync_Per_Sec,Logic_Read_Per_Sec,Login_Per_Sec,MEM_Total,MEM_Used,MEM_Used_Pct,MEM_real_util,Memory_available,Memory_available_pct,Memory_free,Memory_total,Memory_used,Memory_used_pct,New_Tbs_Free_Gb,New_Tbs_Used_Pct,Num_of_processes,Num_of_running_processes,On_Off_State,Outgoing_network_traffic,PGA_Used_Pct,PGA_used_total,Page_pi,Page_po,Physical_Read_Per_Sec,Proc_Used_Pct,Proc_User_Used_Pct,Processor_load_15_min,Processor_load_1_min,Processor_load_5_min,Received_errors_packets,Received_packets,Received_queue,Recv_total,Redo_Per_Sec,Row_Lock,SEQ_Used_Pct,SctRead_Per_Sec,Send_total,Sent_errors_packets,Sent_packets,Sent_queue,SeqRead_Per_Sec,Sess_Active,Sess_Connect,Sess_Used_Temp,Sess_Used_Undo,Session_pct,Shared_memory,Swap_used_pct,System_block_queue_length,System_wait_queue_length,TPS_Per_Sec,Tbs_Free_Gb,Tbs_Used_Pct,TempTbs_Pct,Total_Tbs_Size,Total_disk_space,Total_inodes,UndoTbs_Pct,Used_Tbs_Size,Used_disk_space,Used_disk_space_pct,Used_inodes,Used_inodes_pct,User_Commit,Zombie_Process,container_cpu_used,container_fgc,container_fgct,container_mem_used,container_session_used,container_thread_idle,container_thread_running,container_thread_total,container_thread_used_pct,cost,count,proc,ss_total,succ_rate,tnsping_result_time'
metric_name_list=metrics_name.split(',')
prompt_prefix=f'现在微服务中总共有{node_nums}个节点,其中每个节点在某一个timestamp时刻,会存在一共{len(metric_name_list)}个metrics(指标)状态。'



# adjcent_prompt_list=[]

# for row in range(len(adjcent)):
#     for col in range(len(adjcent)):
#         if adjcent[row][col]==1:
#             prompt=f'{value_name_dict[row]}到{value_name_dict[col]}之间有一条调用路径'
#             adjcent_prompt_list.append(prompt)

# adjcent_prompt=','.join(adjcent_prompt_list) # 邻接矩阵的prompt
# adjcent_prompt+='。'
# prompt_prefix+=adjcent_prompt

prompt_metric=f"""我们将其记录为metric0,metric1,...一直到metric{len(metric_name_list)-1}。这些mtric可能为例如Buffers_used,CPU_free_pct等,值为保留两位小数后的浮点数。接下来我将给出每个不同节点在三个等间隔timestamp所形成的时间窗口下,所对应的每个timestamp相比于上一个timestamp的metrics变化值,如果没有指出某个metric则说明这个metric在这个timestamp中相比于上一个timestamp没有变化,我们将用这个时间窗口的metric变化情况推断该窗口下是否存在异常。"""
prompt_prefix+=prompt_metric
# print(prompt_prefix)

nodes_prompt_list=[]
for index,(time,node_metric_timerange,label) in enumerate(list(zip(timestamp,nodes,labels))):
    prompt_element=''
    prompt1=f'这是第{index}个时间窗口,其中共包含{len(node_metric_timerange[0])}个时刻:'
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
                if math.fabs(metric_value_last-metric_value_now)>1e-5:
                    flag=1
                    diff=metric_value_now-metric_value_last
                    if diff>0:
                        # prompt_diff=f'{metric_name_list[index]}上涨了{diff:.2f},'
                        # prompt_diff=f'metric{index}上涨了{diff:.2f},'
                        up_list.append(f'metric{index}:{diff:.2f}')
                    else:
                        # prompt_diff=f'{metric_name_list[index]}下降了{-diff:.2f},'
                        # prompt_diff=f'metric{index}下降了{-diff:.2f},'
                        down_list.append(f'metric{index}:{diff:.2f}')
            
            if flag==0:
                prompt3+='没有变化。'
            else:
                up_value=",".join(up_list) if len(up_list)>0 else '无变化'
                down_value=",".join(down_list) if len(down_list)>0 else '无变化'
                prompt_diff=f'上涨指标及其对应的值为:{up_value},下降指标及其对应的值为:{down_value}。'
                prompt3+=prompt_diff
            prompt23+=prompt3

        content_summary_dict={}
        content_summary_dict['content']=prompt_prefix+prompt_element+prompt23
        summary='由metrics的变化值反映出来此时该节点存在异常(1)' if label==1 else '由metrics的变化值反映出来此时该节点不存在异常(0)'
        # print(content_summary_dict['content'])
        content_summary_dict['summary']=summary
        nodes_prompt_list.append(content_summary_dict)


np.random.seed(42)
np.random.shuffle(nodes_prompt_list)
print(len(nodes_prompt_list))
train_num_split_point=int(len(nodes_prompt_list)*0.8)

train_nodes_prompt_list=nodes_prompt_list[:train_num_split_point]
print(len(train_nodes_prompt_list))
dev_nodse_prompt_list=nodes_prompt_list[train_num_split_point:]
print(len(dev_nodse_prompt_list))


with open('AIOPS/jianjie2/train_jianjie2.json','w',encoding='utf-8') as f:
    for prompt_finish in train_nodes_prompt_list:
        f.write(json.dumps(prompt_finish,ensure_ascii=False)+'\n')

with open('AIOPS/jianjie2/dev_jianjie2.json','w',encoding='utf-8') as f:
    for prompt_finish in dev_nodse_prompt_list:
        f.write(json.dumps(prompt_finish,ensure_ascii=False)+'\n')



with open('AIOPS/jianjie2/content_example_jianjie2.txt','w',encoding='utf-8') as f:
    f.write(nodes_prompt_list[0]['content'])



