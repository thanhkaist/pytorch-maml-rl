#python train.py --config configs/maml/halfcheetah-dir.yaml --output-folder dir_no_alpha_no_prior --seed 1 --num-workers 8 --num_prior 0 
#python train.py --config configs/maml/halfcheetah-dir.yaml --output-folder dir_alpha_prior --seed 1 --num-workers 8 --num_prior 10 --adapt_lr
#python train.py --config configs/maml/halfcheetah-dir.yaml --output-folder dir_alpha_no_prior --seed 1 --num-workers 8 --num_prior 0 --adapt_lr
#python train.py --config configs/maml/halfcheetah-dir.yaml --output-folder dir_no_alpha_prior --seed 1 --num-workers 8 --num_prior 10

python train.py --config configs/maml/2d-navigation.yaml --output-folder 2d_no_alpha_no_prior1 --seed 1 --num-workers 8 --num_prior 0 
#python train.py --config configs/maml/2d-navigation.yaml --output-folder 2d_alpha_prior1 --seed 1 --num-workers 8 --num_prior 10 --adapt_lr
#python train.py --config configs/maml/2d-navigation.yaml --output-folder 2d_alpha_no_prior1 --seed 1 --num-workers 8 --num_prior 0 --adapt_lr
#python train.py --config configs/maml/2d-navigation.yaml --output-folder 2d_no_alpha_prior1 --seed 1 --num-workers 8 --num_prior 10
