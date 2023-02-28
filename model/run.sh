
# Twitch-e
python3 main.py --dataset fb100 --method IDDG --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 3.0 --lr_a 0.001 --device 1

# Fb-100
python3 main.py --dataset fb100 --method IDDG --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 3.0 --lr_a 0.001 --device 1

#webKB
python3 main.py --dataset webKB --method IDDG --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 3.0 --lr_a 0.001 --device 1

# --gnn change different backbone
