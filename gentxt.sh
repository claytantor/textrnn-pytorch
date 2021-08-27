phrases=("disquieting earth" "truth murmurs" "dark planets" "swift spirit" "hungry creature" "earth sleep" "hungry flower" "swift truth")

# Look for signs of trouble in each log
for i in ${!phrases[@]};
do
    i_phrase=${phrases[$i]}
    python src/predict.py --session $1 --predic 1500 --lines 9 --count 20 --initial "${i_phrase}" -o $2
    # python src/predict.py --session reflect2_2 --predic 800 --lines 10 --initial "${i_phrase}"
done




