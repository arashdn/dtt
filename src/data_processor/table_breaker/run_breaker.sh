# Example of running breaker

#python3 table_breaker.py --rel-dataset-path '/data/Datasets/FF_AJ/' \
#                  --rel-out-dir '/data/Datasets/FF_AJ_Splitted/'

BASE_PATH="/home/arashdn/projects/deep_transformer"
REL_DS_PATH="/data/Datasets"
DS_PATH="$BASE_PATH/$REL_DS_PATH"


for ds in _substr_lens _reverse_lens _replace_lens; do
#for ds in _replace_lens; do
  for dir in $DS_PATH/$ds/*/; do
      if [ -d "$dir" ]; then
        name=$(basename "$dir")
        if [ -n "${name%%_*}" ]; then # ignoring folders starting with _
          out_dir="$REL_DS_PATH/${ds}_splitteds/$name"
#          mkdir -p $out_dir
          python3 table_breaker.py --rel-dataset-path "$REL_DS_PATH/$ds/$name" --rel-out-dir "$out_dir"
#          echo "$dir"
#          echo "$name"
          printf "\n\n--------------\n\n"
        fi # if _
      fi # if dir
  done # dir
done # ds



#for ds in _substr_lens; do
#  for dir1 in $DS_PATH/$ds/*/; do
#      if [ -d "$dir1" ]; then
#        name1=$(basename "$dir1")
#        for dir in $dir1/*/; do
#          if [ -d "$dir" ]; then
#            name=$(basename "$dir")
#            if [ -n "${name%%_*}" ]; then # ignoring folders starting with _
#              out_dir="$REL_DS_PATH/${ds}_splitteds/$name1/$name"
##              mkdir -p $out_dir
#              python3 table_breaker.py --rel-dataset-path "$REL_DS_PATH/$ds/$name1/$name" --rel-out-dir "$out_dir"
#              echo "$dir"
##              echo "$name"
#              printf "\n\n--------------\n\n"
#            fi # if _
#          fi # if dir
#      done # dir
#    fi # if dir1
#  done # dir 1
#done # ds
