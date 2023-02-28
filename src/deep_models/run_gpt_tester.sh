### With our framework ###

for ds in FF_AJ_Splitted Synthetic_basic_10tr_100rows__08_35len_Splitted Single_Substr_05tr_050rows__08_35len_Splitted Single_Reverse_05tr_050rows__08_35len_Splitted Single_Replace_05tr_050rows__08_35len_Splitted; do
  python3 tester.py --number-of-examples-for-join 5 \
          --matching-type edit_dist \
          --rel-auto-out-file-dir '/data/output_gpt/' \
          --rel-dataset-path "/data/Datasets/$ds/" \
          --rel-model-path "" \
          --model-save-name "gpt3"\
          --model-name 'gpt3' \
          --use-gpu n

  printf "\n\n--------------\n\n"

done # for ds


### Without our framework ###

for len in '1' '3' '5' '10'; do
  for ds in FF_AJ_Splitted Synthetic_basic_10tr_100rows__08_35len_Splitted Single_Substr_05tr_050rows__08_35len_Splitted Single_Reverse_05tr_050rows__08_35len_Splitted Single_Replace_05tr_050rows__08_35len_Splitted; do
    python3 tester.py --number-of-examples-for-join $len \
            --matching-type edit_dist \
            --rel-auto-out-file-dir '/data/output_gpt_nf/' \
            --rel-dataset-path "/data/Datasets/$ds/" \
            --rel-model-path "" \
            --model-save-name "gpt3-nf"\
            --model-name 'gpt3' \
            --use-gpu n \
            --no-framework y

    printf "\n\n--------------\n\n"

  done # for ds
done # len
