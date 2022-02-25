###
 # @Author: jianzhnie
 # @Date: 2021-11-16 14:14:32
 # @LastEditTime: 2022-02-24 16:13:51
 # @LastEditors: jianzhnie
 # @Description:
 #
###

python main.py \
    --output_dir=./logs/Womens_Clothing_E-Commerce_Reviews \
    --task=classification \
    --combine_feat_method=individual_mlps_on_cat_and_numerical_feats_then_concat \
    --do_train \
    --overwrite_output_dir \
    --model_name_or_path=bert-base-uncased \
    --data_path=./datasets/Womens_Clothing_E-Commerce_Reviews \
    --column_info_path=./datasets/Womens_Clothing_E-Commerce_Reviews/column_info.json
