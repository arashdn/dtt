# Example of running tester

DS_PATH="data/Datasets/noisy"

#model="/models/_s1/byt5-base-basic_synth_01000_10-checkpoints"
model="/models/byt5-base-basic_synth_02000_10-checkpoints"

mdl=$(basename "$model")

ratio="0.0"

for name in "AJ_Splitted--$ratio--noisy" "FF_Splitted--$ratio--noisy" "Single_Replace_05tr_050rows__08_35len_Splitted--$ratio--noisy" "Single_Substr_05tr_050rows__08_35len_Splitted--$ratio--noisy" "Synthetic_basic_10tr_100rows__08_35len_Splitted--$ratio--noisy"; do
  for len in 1 2 3 4 5 6 7 8 9 10; do
      echo "$name -> $len"
      python3 tester.py --number-of-examples-for-join $len \
              --matching-type edit_dist \
              --rel-auto-out-file-dir "/data/output_noisy_examples/" \
              --rel-dataset-path "/data/Datasets/noisy/$name/" \
              --rel-model-path "$model/best-checkpoint.ckpt" \
              --model-save-name "$(echo "$mdl" | sed "s/-checkpoints/_/")"\
              --model-name 'google/byt5-base' \
              --use-gpu y

      printf "\n\n--------------\n\n"
  done
done
