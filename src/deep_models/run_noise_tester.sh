# Example of running tester

DS_PATH="data/Datasets/noisy"

#model="/models/_s1/byt5-base-basic_synth_01000_10-checkpoints"
model="/models/byt5-base-basic_synth_02000_10-checkpoints"

mdl=$(basename "$model")

for dir in $DS_PATH/*/; do
    if [ -d "$dir" ]; then
      name=$(basename "$dir")
      if [ -n "${name%%_*}" ]; then # ignoring folders starting with _
        echo "$name"
        python3 tester.py --number-of-examples-for-join 5 \
                --matching-type edit_dist \
                --rel-auto-out-file-dir "/data/output_noisy/" \
                --rel-dataset-path "/data/Datasets/noisy/$name/" \
                --rel-model-path "$model/best-checkpoint.ckpt" \
                --model-save-name "$(echo "$mdl" | sed "s/-checkpoints/_/")"\
                --model-name 'google/byt5-base' \
                --use-gpu y

        printf "\n\n--------------\n\n"
      fi # if _
    fi # if dir
done # for ds
