
dir="/models/byt5-base-basic_synth_02000_10-checkpoints"
name=$(basename "$dir")
if [ -n "${name%%_*}" ]; then # ignoring folders starting with _
  for ds in FF_AJ_Splitted DXF_Splitted Synthetic_basic_10tr_100rows__08_35len_Splitted Single_Substr_05tr_050rows__08_35len_Splitted Single_Reverse_05tr_050rows__08_35len_Splitted Single_Replace_05tr_050rows__08_35len_Splitted; do
    python3 tester.py --number-of-examples-for-join 5 \
            --matching-type edit_dist \
            --rel-auto-out-file-dir '/data/output_two_model/' \
            --rel-dataset-path "/data/Datasets/$ds/" \
            --rel-model-path "/models/$name/best-checkpoint.ckpt" \
            --model-save-name "$(echo "$name" | sed "s/-checkpoints/_/")--gpt3"\
            --model-name 'google/byt5-base,gpt3' \
            --use-gpu y

    printf "\n\n--------------\n\n"

  done # for ds
fi # if _

