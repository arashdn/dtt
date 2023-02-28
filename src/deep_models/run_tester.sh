# Example of running tester

#python3 tester.py --number-of-examples-for-join 5 \
#                  --matching-type edit_dist \
#                  --rel-auto-out-file-dir '/data/output/' \
#                  --rel-dataset-path '/data/Datasets/FF_AJ_Splitted/' \
#                  --rel-model-path '/models/byt5-base-basic_synth_40000_10.ckpt' \
#                  --model-save-name "byt5-base-basic_synth_40000_10"\
#                  --model-name 'google/byt5-base' \
#                  --use-gpu y


for dir in /home/arashdn/projects/deep_transformer/models/*/; do
#dir="/home/arashdn/projects/deep_transformer/models/byt5-base-basic_synth_15000_10-checkpoints"
    if [ -d "$dir" ]; then
      name=$(basename "$dir")
      if [ -n "${name%%_*}" ]; then # ignoring folders starting with _
        for ds in FF_AJ_Splitted Synthetic_basic_10tr_100rows__08_35len_Splitted Single_Substr_05tr_050rows__08_35len_Splitted Single_Reverse_05tr_050rows__08_35len_Splitted Single_Replace_05tr_050rows__08_35len_Splitted; do
          python3 tester.py --number-of-examples-for-join 5 \
                  --matching-type edit_dist \
                  --rel-auto-out-file-dir '/data/output/' \
                  --rel-dataset-path "/data/Datasets/$ds/" \
                  --rel-model-path "/models/$name/best-checkpoint.ckpt" \
                  --model-save-name "$(echo "$name" | sed "s/-checkpoints/_/")"\
                  --model-name 'google/byt5-base' \
                  --use-gpu y

          printf "\n\n--------------\n\n"

        done # for ds
      fi # if _
    fi # if dir
done

