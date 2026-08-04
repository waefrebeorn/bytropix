#!/bin/bash
# run_sft.sh -- the SFT cold-start run (research/052: chat-template corpus,
# lr 1e-5, seq 2048, resume from the 500-step base). Checkpoints every 200.
cd /home/wubu/wubuwizard || exit 1
./wubu_train_gpu \
    --resume /home/wubu/models/corpus/seed-48.st-0500.st \
    --tok /home/wubu/models/corpus/sft-tok/wubu-sft.tok \
    --steps 2000 --seq 2048 \
    --muon-lr 1e-5 --adam-lr 1e-5 --ckpt 200 \
    --out /home/wubu/models/corpus/checkpoints/seed-sft.st \
    > /home/wubu/models/corpus/checkpoints/sft-run.log 2>&1
echo "EXIT=$?"
