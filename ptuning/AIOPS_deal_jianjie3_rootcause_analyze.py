import os
import json

file_origin='AIOPS/jianjie3_rootcause_analyze/train_jianjie3_final.json'
new_ob_list=[]
with open(file_origin,'r') as f:
    for line in f.readlines():
        line=line.strip()
        ob=json.loads(line)
        summary=ob['summary']
        if '1' in summary and ',' not in summary:
            continue
        new_ob_list.append(ob)

file_save='AIOPS/jianjie3_rootcause_analyze/train_jianjie3_withcause.json'
with open(file_save,'w') as f:
    for ob in new_ob_list:
        f.write(json.dumps(ob,ensure_ascii=False)+'\n')