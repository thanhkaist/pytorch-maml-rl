#python train_1.py --config configs/maml/halfcheetah-vel.yaml --output-folder noise_easy_vel_alpha_prior --seed 1 --num-workers 8 --num_prior 10 --adapt_lr 
#python train_1.py --config configs/maml/halfcheetah-vel.yaml --output-folder noise_easy_vel_noalpha_prior --seed 1 --num-workers 8 --num_prior 10 #--adapt_lr 
#python train_1.py --config configs/maml/halfcheetah-vel.yaml --output-folder noise_easy_vel_alpha_noprior --seed 1 --num-workers 8 --num_prior 0 --adapt_lr 
#python train_1.py --config configs/maml/halfcheetah-vel.yaml --output-folder noise_easy_vel_noalpha_noprior --seed 1 --num-workers 8 --num_prior 0 #--adapt_lr 
#

python train_1.py --config configs/maml/halfcheetah-vel.yaml --output-folder noise_med_vel_alpha_prior_0.5 --seed 2 --num-workers 8 --num_prior 10 --adapt_lr 
python train_1.py --config configs/maml/halfcheetah-vel.yaml --output-folder noise_med_vel_noalpha_prior_0.5 --seed 2 --num-workers 8 --num_prior 10 #--adapt_lr 
python train_1.py --config configs/maml/halfcheetah-vel.yaml --output-folder noise_med_vel_alpha_noprior_0.5 --seed 2 --num-workers 8 --num_prior 0 --adapt_lr 
python train_1.py --config configs/maml/halfcheetah-vel.yaml --output-folder noise_med_vel_noalpha_noprior_0.5 --seed 2 --num-workers 8 --num_prior 0 #--adapt_lr 

