#!/bin/bash

for LEARN_RATE in 0.1 
do
    for SEED in 20141119 
    do
        for SCALE in false true 
        do
            for USER_FACTOR in true false 
            do
                for NUM_DIM in 50
                do
                    for NUM_NEG in 5 #10
                    do
                        for RATIO in 0 0.2 0.4 0.6 0.8 1.0 # 0.9 1.0 
                        do
                            for LINEAR in false true
                            do
                                for ASYM in true false
                                do
                                    for LOSS in SQUARE CE
                                    do
                                        for LINFUNC in false
                                        do
                                            autoqsub ./yelp_implicit  --task=train --method=CDAE --learn_rate=${LEARN_RATE} --num_dim=${NUM_DIM} --cnum=1 --cratio=${RATIO} --adagrad=true --asym=${ASYM} --linear=${LINEAR} --scaled=${SCALE} --user_factor=${USER_FACTOR} --loss_type=${LOSS} --beta=1 --linear_function=${LINFUNC} --tanh=0 --seed=${SEED} --linear_output=true
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
