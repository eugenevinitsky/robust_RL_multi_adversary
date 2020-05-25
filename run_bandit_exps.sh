date="$(date +'%d_%m_%H%M%S')"
echo $date

# title num_arms horizon seed lr adv_lr extra_options
function run_base_command(){
    ray exec --cluster-name kp_${date}_bandit_arm$2_h$3_s$4_lr$5_advlr$6_$1 autoscale.yaml "python ~/adversarial_sim2real/run_scripts/bandit/run_multiarm_bandit.py --multi_node --use_s3 --num_cpus 39 --exp_title kp_${date}_bandit_arm$2_h$3_lr$5_advlr$6_$1 --num_arms $2 --horizon $3 --num_samples 1 --num_iters 400 --checkpoint_freq 100 --run_transfer --use_lstm --regret_formulation --seed $4 --lr $5 --adv_lr $6 $7" --start --stop --tmux &
}

# exp_title num_arms horizon seed lr
function run_dr(){
    run_base_command dr_$1 $2 $3 $4 $5 0.0 "--num_adv_strengths 0"
}
# exp_title num_arms horizon seed lr adv_lr num_advs
function run(){
    run_base_command $7adv_rarl_$1 $2 $3 $4 $5 $6 "--num_adv_strengths 1 --advs_per_strength $7"
}

# exp_title num_arms horizon
function best_hyperparam_exps(){
    run_dr 10 100 $2 1e-3
    run $1 10 100 $2 1e-4 1e-5 1
    run $1 10 100 $2 1e-4 1e-5 4
    run $1 10 100 $2 1e-4 1e-5 10
}

function seed_sweep_best_hypers(){
    best_hyperparam_exps $1 1
    sleep 60
    best_hyperparam_exps $1 2
    sleep 60
    best_hyperparam_exps $1 3
    sleep 60
    best_hyperparam_exps $1 4
    sleep 60
    best_hyperparam_exps $1 5
    sleep 60
    best_hyperparam_exps $1 6
}