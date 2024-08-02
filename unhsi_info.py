import os
import json

with open("sceneplan/partnet_simple_all.json", "r") as f:
    parnet_info = json.load(f)

for i in range(len(parnet_info)):
    key = str(i).rjust(4, '0')
    value = parnet_info[key]['obj']['000']
    if value['name'] == 'reach':
        continue

    obj_id = value['id']
    if obj_id == '38637':
        continue

    value['stand_point'] = value['stand_point'][0]

    with open(os.path.join('data/partnet_UniHSI_0802', obj_id, 'meta.json'), 'w') as f:
        json.dump(value, f, indent=4)
