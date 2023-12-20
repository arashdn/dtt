BASE_PATH="SPECIFY_ROOT_PATH_FOR_PROJECT"
REL_DS_PATH="/data/Datasets"
DS_PATH="$BASE_PATH/$REL_DS_PATH"


for ds in FF_Splitted AJ_Splitted Synthetic_basic_10tr_100rows__08_35len_Splitted Single_Replace_05tr_050rows__08_35len_Splitted Single_Substr_05tr_050rows__08_35len_Splitted; do
  for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
      name=$(basename "$dir")
      out_dir="$REL_DS_PATH/noisy/$ds--$ratio--noisy"
      inp_dir="$REL_DS_PATH/$ds/"
      python3 dataset_noiser.py --rel-dataset-path "$inp_dir" \
           --rel-out-dir "$out_dir" --noise-ratio "$ratio"
#      echo "$inp_dir"
#      echo "$out_dir"
#      echo "$ratio"
      printf "\n\n--------------\n\n"
  done # dir
done # ds
