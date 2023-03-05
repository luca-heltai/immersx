for rr in 6 7 8 9 10; do
    for NN in 1 3 5 7 9; do
        for RR in 2 1 05 025; do
            echo parameter_ref${rr}_R0${RR}_N${NN}.prm
            mkdir -p ref${rr}_R0${RR}_N${NN}
            sed -e "s/XXX/$NN/g" -e "s/YYY/$RR/g" -e "s/ZZZ/$rr/g" \
                < ../../prms/one_circle_convergence_various_coeffs.prm \
                > parameter_ref${rr}_R0${RR}_N${NN}.prm        
        done
    done
done