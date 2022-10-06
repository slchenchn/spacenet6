set -x
python tools/train.py configs/dbes/dbes_hr18_512x512_40k_sn6_sar_pro_rorated.py
python tools/train.py configs/hrnet/fcn_hr18_512x512_40k_sn6_sar_pro_rotated.py