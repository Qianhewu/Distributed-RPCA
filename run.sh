# python rpca_distributed.py  --epoch 100 --sparsity 0.05 --tau 2 --lr 0.8 --mu0 0.15
# python rpca_distributed.py  --epoch 100 --sparsity 0.1 --tau 2 --lr 0.8 --mu0 0.15
# python rpca_distributed.py  --epoch 100 --sparsity 0.15 --tau 2 --lr 0.8 --mu0 0.15
# python rpca_distributed.py  --epoch 100 --sparsity 0.2 --tau 2 --lr 0.8 --mu0 0.15
# python rpca_distributed.py  --epoch 100 --sparsity 0.25 --tau 2 --lr 0.8 --mu0 0.15
# python rpca_distributed.py  --epoch 100 --sparsity 0.3 --tau 2 --lr 0.8 --mu0 0.15
# python rpca_distributed.py  --epoch 100 --sparsity 0.5 --tau 2 --lr 0.8 --mu0 0.15


# Exact rank
python rpca_distributed.py --method DCF --M 500 --N 500 --tau 5 --num_clients 5 --lr 0.2 --epoch 100 --mu0 0.1
python rpca_distributed.py --method DCF --M 1000 --N 1000 --tau 5 --num_clients 10 --lr 0.03 --epoch 100 --mu0 0.2
python rpca_distributed.py --method DCF --M 3000 --N 3000 --tau 5 --num_clients 10 --lr 0.01 --epoch 100 --mu0 0.2
python rpca_distributed.py --method CF --M 500 --N 500 --tau 1 --num_clients 1 --lr 0.06 --epoch 100 --mu0 0.3
python rpca_distributed.py --method CF --M 1000 --N 1000 --tau 1 --num_clients 1 --lr 0.01 --epoch 100 --mu0 0.5
python rpca_distributed.py --method CF --M 3000 --N 3000 --tau 1 --num_clients 1 --lr 0.005 --epoch 100 --mu0 1.3

# 2*rank recovery:
# python rpca_distributed.py --method DCF --M 500 --N 500 --tau 5 --num_clients 5 --lr 0.1 --epoch 100 --mu0 0.2
# python rpca_distributed.py --method DCF --M 1000 --N 1000 --tau 5 --num_clients 10 --lr 0.03 --epoch 100 --mu0 0.2
# python rpca_distributed.py --method DCF --M 3000 --N 3000 --tau 5 --num_clients 10 --lr 0.01 --epoch 100 --mu0 0.2

# APGM:
python rpca_distributed.py --M 100 --N 100 --method APGM --mu0 0.2
python rpca_distributed.py --M 500 --N 500 --method APGM --mu0 1
python rpca_distributed.py --M 1000 --N 1000 --method APGM --mu0 2
python rpca_distributed.py --method APGM --M 3000 --N 3000 --mu0 5

# python rpca_distributed.py --method DCF --M 100 --N 100 --tau 2 --num_clients 4 --lr 0.005 --epoch 100
# python rpca_distributed.py --method DCF --M 500 --N 500 --tau 5 --num_clients 5 --lr 0.0015 --epoch 100
# python rpca_distributed.py --method DCF --M 1000 --N 1000 --tau 5 --num_clients 10 --lr 0.01 --epoch 100
# python rpca_distributed.py --method DCF --M 3000 --N 3000 --tau 5 --num_clients 10 --lr 0.0002 --epoch 100
# python rpca_distributed.py --method DCF --M 10000 --N 10000 --tau 5 --num_clients 10 --lr 0.00001 --epoch 150
# python rpca_distributed.py --method DCF --tau 1 --num_clients 1 --lr 0.01 --epoch 100

# python rpca_distributed.py --method DCF --tau 1 --num_clients 10 --lr 0.003 --epoch 50 --output "differentK.pkl"
# python rpca_distributed.py --method DCF --tau 3 --num_clients 10 --lr 0.003 --epoch 50 --output "differentK.pkl"
# python rpca_distributed.py --method DCF --tau 5 --num_clients 10 --lr 0.003 --epoch 50 --output "differentK.pkl"
# python rpca_distributed.py --method DCF --tau 10 --num_clients 10 --lr 0.003 --epoch 50 --output "differentK.pkl"


# python rpca_distributed.py --method CF --M 100 --N 100 --tau 2 --num_clients 4 --lr 0.005 --epoch 100
# python rpca_distributed.py --method CF --M 500 --N 500 --tau 1 --num_clients 1 --lr 0.0006 --epoch 100
# python rpca_distributed.py --method CF --M 1000 --N 1000 --tau 1 --num_clients 1 --lr 0.0002 --epoch 100
# python rpca_distributed.py --method CF --M 3000 --N 3000 --tau 1 --num_clients 1 --lr 0.0001 --epoch 100