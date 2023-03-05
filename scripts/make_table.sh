# Make the L2 error table for the convergence study - ref10
echo L2 errors, rows: N, columns: r
for RR in 2 1 05 025; do
    printf "0.%-4s" $RR
    for NN in 1 3 5 7 9; do
        file=error_ref10_R0${RR}_N${NN}.txt
        grep 1050625 $file | awk '{printf "%s ",$3}'
    done
    echo ""
done

# Make the H1 error table for the convergence study - ref10
echo H1 errors, rows: N, columns: r
for RR in 2 1 05 025; do
    printf "0.%-4s" $RR
    for NN in 1 3 5 7 9; do
        file=error_ref10_R0${RR}_N${NN}.txt
        grep 1050625 $file | awk '{printf "%s ",$4}'
    done
    echo ""
done