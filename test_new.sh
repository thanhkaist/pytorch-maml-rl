#python test.py --input mean_vel_noalpha_prior --meta-batch-size 40 --num-batches 10  --num-workers 8
#python test.py --input vel_alpha_no_prior --meta-batch-size 40 --num-batches 10  --num-workers 8
#python test.py --input 2d_alpha_prior1 --meta-batch-size 40 --num-batches 10  --num-workers 8
#python test.py --input dir_alpha_no_prior --meta-batch-size 40 --num-batches 10  --num-workers 8
#python test.py --input noise_vel_alpha_prior --meta-batch-size 40 --num-batches 10  --num-workers 8
#python test.py --input noise_vel_noalpha_prior --meta-batch-size 40 --num-batches 10  --num-workers 8
#python test.py --input noise_vel_alpha_noprior --meta-batch-size 40 --num-batches 10  --num-workers 8
#python test.py --input noise_vel_noalpha_noprior --meta-batch-size 40 --num-batches 10  --num-workers 8

python test.py --input noise_med_vel_alpha_prior_0.5 --meta-batch-size 40 --num-batches 10  --num-workers 8
python test.py --input noise_med_vel_noalpha_prior_0.5 --meta-batch-size 40 --num-batches 10  --num-workers 8
python test.py --input noise_med_vel_alpha_noprior_0.5 --meta-batch-size 40 --num-batches 10  --num-workers 8
python test.py --input noise_med_vel_noalpha_noprior_0.5 --meta-batch-size 40 --num-batches 10  --num-workers 8