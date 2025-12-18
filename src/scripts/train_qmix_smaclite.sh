#!/bin/sh
algo="qmix"
seed_max=1
scenarios=( "MMM" "MMM2" "corridor" "bane_vs_bane" "25m_vs_30m" "3s5z_vs_3s6z")
total_times=( 1_000_000 2_000_000 3_000_000 2_000_000 2_000_000 5_000_000 )


n_exp=${#scenarios[@]}

for ((i=0; i<n_exp; i++)); do
    for seed in `seq ${seed_max}`;
    do
        scenario="${scenarios[$i]}"
        total_ts="${total_times[$i]}"
        python src/main.py --config=${algo} --env-config=smaclite with env_args.time_limit=150 env="smaclite" \
                env_args.map_name="custom-smaclite/${scenario}" t_max=${total_ts}
    done
done
