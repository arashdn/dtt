# Example of running tester

DS_PATH="/home/arashdn/projects/deep_transformer/data/Datasets"

#model="/models/_s1/byt5-base-basic_synth_01000_10-checkpoints"
model="/models/byt5-base-basic_synth_02000_10-checkpoints"

mdl=$(basename "$model")

for base in _reverse_lens_splitteds _substr_lens_splitteds _replace_lens_splitteds; do
  for dir in $DS_PATH/$base/*/; do
      if [ -d "$dir" ]; then
        name=$(basename "$dir")
        if [ -n "${name%%_*}" ]; then # ignoring folders starting with _
          python3 tester.py --number-of-examples-for-join 5 \
                  --matching-type edit_dist \
                  --rel-auto-out-file-dir "/data/output/$base/" \
                  --rel-dataset-path "/data/Datasets/$base/$name/" \
                  --rel-model-path "$model/best-checkpoint.ckpt" \
                  --model-save-name "$(echo "$mdl" | sed "s/-checkpoints/_/")"\
                  --model-name 'google/byt5-base' \
                  --use-gpu y

          printf "\n\n--------------\n\n"
        fi # if _
      fi # if dir
  done # for ds
done

