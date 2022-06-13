Non-Autoregressive summarization model with Character-level length Control (NACC)
=======
This repo contains the code for our paper [A Character-Level Length-Control Algorithm for Non-Autoregressive Sentence Summarization](https://arxiv.org/abs/2205.14522).

## Environment setup
The scripts are developed with [Anaconda](https://www.anaconda.com/) python 3.8, and the working environment can be configured with the following commands. 

```
git clone https://github.com/MANGA-UOFA/NACC
cd NACC
conda create -n NACC_MANGA python=3.8

conda activate NACC_MANGA

pip install gdown
pip install git+https://github.com/tagucci/pythonrouge.git
conda install pytorch cudatoolkit=10.2 -c pytorch

git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
cd ..
rm -rf ctcdecode

pip install -e.
```

## Data downloading
The [search-based](https://aclanthology.org/2020.acl-main.452.pdf) summaries on the Gigaword dataset and pre-trained model weights can be found in this publically available [Google drive folder](https://drive.google.com/drive/folders/1LNIs6Qf1Ojvb-QYhgzYlTKv97Z_EQif9), which can be automatically downloaded and organized with the following commands. 

```
chmod +x download_data.sh
./download_data.sh
```

## Preprocess
Execute the following command to preprocess the data.

```
chmod +x preprocess_giga.sh
./preprocess_giga.sh
```


## Model training
Our training script is ```train.py```. We introduce some of its important training parameters, other parameters can be found [here](https://fairseq.readthedocs.io/en/latest/command_line_tools.html).

**data_source**: (Required) Directory to the pre-processed training data (e.g., data-bin/gigaword_10).

**arch**: (Required) Model Architecture. This must be set to ```nat_encoder_only_ctc```.

**task**: (Required) The task we are training for. This must be set to ```summarization```.

**best-checkpoint-metric**: (Required) Criteria to save the best checkpoint. This can be set to either ```rouge``` or ```loss``` (we used rouge).

**criterion**: (Required) Criteria for training loss calculation. This must be set to ```summarization_ctc```. 

**max-valid-steps**: (Optional) Maximum steps during validation. e.g., ```100```. Limiting this number avoids time-consuming validation on a large validation dataset. 

**batch-size-valid** (Optional) Batch size during validation. e.g., ```5```. Set this parameter to ```1``` if you want to test the **unparallel** inference efficiency. 

**decoding_algorithm**: (Optional) Decoding algorithm of model output (logits) sequence. This can be set to ```ctc_char_greedy_decoding``` and ```ctc_char_length_control```.

**truncate_summary**: (Optional) Whether to truncate the generated summaries. This parameter is valid when ```decoding_algorithm``` is set to ```ctc_char_greedy_decoding```.

**desired_length**: (Optional) Desired (maximum) number of characters of the output summary. If ```decoding_algorithm``` is set to ```ctc_char_greedy_decoding```, and ```truncate_summary``` is ```True```, the model will truncate longer summaries to the ```desired_length```.
When ```decoding_algorithm``` is  ```ctc_char_length_control```, the model's decoding strategy depends on the parameter ```force_length```, which will be explained in the next paragraph. 

**force_length**: (Optional) This parameter is only useful when ```decoding_algorithm``` is set to ```ctc_char_length_control```; the parameter determines whether to force the length of the generated summaries to be ```desired_length```. If ```force_length``` is set to ```False```, the model returns the greedily decoded summary if the summary length does not exceed ```desired_length```. Otherwise, the model search for the (approximately) most probable summary of the ```desired_length``` with a [length control](https://arxiv.org/abs/2205.14522) algorithm. 

**bucket_size**: (Optional) This parameter is only useful when ```decoding_algorithm``` is set to ```ctc_char_length_control```. It refers to the bucket size of the length control algorithm.

**valid_subset**: (Optional) Names of the validation dataset, separating by comma, e.g, test,valid.

**max_token**: (Optional) Max tokens in each training batch.

**max_update**: (Optional) Maximum training steps.


For example, if we want to train NACC with 10-word searched summaries, ```ctc_char_length_control``` decoding, desired length of ```50``` and bucket size of ```2```, we can use the following training command. 

```
data_source=gigaword_10
decoding_algorithm=ctc_char_length_control
desired_length=50
bucket_size=2
valid_subset=valid
drop_out=0.1
max_token=4096
max_update=50000
force_length=False
truncate_summary=False
max_valid_steps=100
label_smoothing=0.1

CUDA_VISIBLE_DEVICES=0 nohup python train.py data-bin/$data_source --source-lang article --target-lang summary --save-dir NACC_${data_source}_${max_token}_${decoding_algorithm}_${desired_length}_truncate_summary_${truncate_summary}_label_smoothing_${label_smoothing}_dropout_${drop_out}_checkpoints --keep-interval-updates 5 --save-interval-updates 5000 --validate-interval-updates 5000 --scoring rouge --maximize-best-checkpoint-metric --best-checkpoint-metric rouge --log-format simple --log-interval 100 --keep-last-epochs 5 --keep-best-checkpoints 5 --share-all-embeddings --encoder-learned-pos --optimizer adam --adam-betas "(0.9,0.98)" --lr 0.0005 --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 --warmup-updates 10000 --warmup-init-lr 1e-07 --weight-decay 0.01 --fp16 --clip-norm 2.0 --max-update $max_update --task summarization --criterion summarization_ctc --arch nat_encoder_only_ctc --activation-fn gelu --dropout 0.1 --max-tokens $max_token --valid-subset $valid_subset --decoding_algorithm $decoding_algorithm --desired_length $desired_length --bucket_size $bucket_size --force_length $force_length --truncate_summary $truncate_summary --max-valid-steps $max_valid_steps&
```

## Model evaluation
Our evaluation script is ```fairseq_cli/generate.py```, and it inherits the training arguments related to the data source, model architecture and decoding strategy.
Besides, it requires the following arguments. 

**path**: (Required) Directory to the trained model (e.g., NACC/checkpoint_best.pt).

**gen-subset**: (Required) Names of the generation dataset (e.g., test). 

**scoring**: (Required) Similar to the criteria in training arguments, it must be set to ```rouge```.


For example, the following command evaluates the performance of our pretrained model```HC_8.pt``` on the Gigaword test dataset.

```
data_source=data-bin/gigaword_8
path=model_weights/HC_8.pt
seed=17
gen_subset=test
decoding_algorithm=ctc_char_length_control
desired_length=50
bucket_size=2
truncate_summary=False

CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py $data_source --seed $seed --source-lang article --target-lang summary --path $path  --task summarization --scoring rouge --arch nat_encoder_only_ctc --gen-subset $gen_subset --model-overrides "{'decoding_algorithm': '$decoding_algorithm', 'desired_length': $desired_length, 'bucket_size': $bucket_size, 'truncate_summary': $truncate_summary, 'beam_size': 1, 'generator_type': 'ctc'}"

```

The evaluation result will be saved at the folder ```*_evaluation_result``` by default, including the generated summaries and the statistics of the generated summaries (e.g., ROUGE score).

Notice: if you want to test the **unparallel** inference efficiency, include an extra parameter ```--batch-size 1``` in the evaluation command.

## Final comments
As you may notice, our script is developed based on [Fairseq](https://github.com/pytorch/fairseq), which is a very useful & extendable package to develop Seq2Seq models. We didn't ruin any of its built-in functionality to retain its extension ability. 

