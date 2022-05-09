python  -u train.py --nhid 64 --pretrain 1 --data chameleon >cham.out
python -u train.py --pretrain 1 >1.out
python -u train.py --pretrain 1 --nhid 64 >2.out

python -u train.py --pretrain 0 >3.out