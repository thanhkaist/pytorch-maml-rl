python train_rp.py --config configs/maml/halfcheetah-vel.yaml --output-folder rp_noise_med_vel_alpha_prior_0.2 --prior_type MEDIUM --seed 2 --num-workers 8 --num_prior 10 --adapt_lr 
python train_rp.py --config configs/maml/halfcheetah-vel.yaml --output-folder rp_noise_med_vel_noalpha_prior_0.2 --prior_type MEDIUM --seed 2 --num-workers 8 --num_prior 10 #--adapt_lr 
python train_rp.py --config configs/maml/halfcheetah-vel.yaml --output-folder rp_noise_med_vel_alpha_noprior_0.2 --prior_type MEDIUM --seed 2 --num-workers 8 --num_prior 0 --adapt_lr 
python train_rp.py --config configs/maml/halfcheetah-vel.yaml --output-folder rp_noise_med_vel_noalpha_noprior_0.2 --prior_type MEDIUM --seed 2 --num-workers 8 --num_prior 0 #--adapt_lr 

#python train_rp.py --config configs/maml/halfcheetah-dir.yaml --output-folder dir_no_alpha_no_prior --seed 1 --num-workers 8 --num_prior 0 
#python train_rp.py --config configs/maml/halfcheetah-dir.yaml --output-folder dir_alpha_prior --seed 1 --num-workers 8 --num_prior 10 --adapt_lr
#python train_rp.py --config configs/maml/halfcheetah-dir.yaml --output-folder dir_alpha_no_prior --seed 1 --num-workers 8 --num_prior 0 --adapt_lr
#python train_rp.py --config configs/maml/halfcheetah-dir.yaml --output-folder dir_no_alpha_prior --seed 1 --num-workers 8 --num_prior 10
#
#python train.py --config configs/maml/2d-navigation.yaml --output-folder 2d_no_alpha_no_prior1 --seed 1 --num-workers 8 --num_prior 0 
#python train.py --config configs/maml/2d-navigation.yaml --output-folder 2d_alpha_prior1 --seed 1 --num-workers 8 --num_prior 10 --adapt_lr
#python train.py --config configs/maml/2d-navigation.yaml --output-folder 2d_alpha_no_prior1 --seed 1 --num-workers 8 --num_prior 0 --adapt_lr
#python train.py --config configs/maml/2d-navigation.yaml --output-folder 2d_no_alpha_prior1 --seed 1 --num-workers 8 --num_prior 10
