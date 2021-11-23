
#!/usr/bin/env bash

for i in {1..15}
do
    python tools/train.py configs/deeplabv3/deeplabv3_r50-d8_512x512_20k_sn6_sar_pro_rotated.py
done