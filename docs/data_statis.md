generate data root: `/home/liuyun/licheng/UniHSI/gen_data`

retarget data root (notalign): `/home/liuyun/licheng/h1_retargeting/retarget_data`

final data root: `/home/liuyun/licheng/retarget_data`

## chair 

|folder |         Note           |  data size | final data folder |retargeted folder |
|---|:---------------------:|:--------------:|:--------------:|:--------------:|
| partnet_UniHSI_chair_AUG_0814 | chairs in origin UniHSI       | 49 chairs * 10 aug * 10 init|UniHSI_retargeted_data_augmented_sit |-- |
| partnet_UniHSI_chair_AUG_0825 | 500 extra chairs in parnet  | 495 chairs * 10 aug * 1 init| |UniHSI_retargeted_partnet_add500_aug_chair_sit_0825_notalign|
| partnet_UniHSI_chair_0902 | all extra chairs in parnet    | 8000 chairs * 1 aug * 1 init| | |


## bed 

|folder |         Note           |  data size | final data folder |retargeted folder |
|---|:---------------------:|:--------------:|:--------------:|:--------------:|
| partnet_UniHSI_bed_AUG_0831 | beds in origin UniHSI | 22 beds * 10 aug * 10 init|  | |
| partnet_UniHSI_bed_AUG_0901 | all extra 56 regular beds in partnet | 51 beds * 10 aug * 10 init|  | |