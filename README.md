# Argument Mining in Scientific Reviews (AMSR)
Accompanying repository of our [AAAI2021](https://arxiv.org/abs/2012.07743) paper "Argument Mining Driven Analysis of Peer-Reviews".

We release a new dataset of peer-reviews from different computer science conferences with annotated arguments, called AMSR (**A**rgument **M**ining in **S**cientific **R**eviews).

## Dataset
The dataset is available [here](https://zenodo.org/record/4314390).
It contains:
1. Raw Conference Data
2. Cleaned Conference Data
3. Annotated Conference Data

## Requirements

The requirements of our code are given in [requirements.txt](./requirements.txt).
Make sure you have at least Python 3.7 with pip installed.
Navigate to the root folder of the repository, where the file `requirements.txt` is located, and install all dependencies in the current Python environment with

```bash
pip install -r requirements.txt
```


## Steps

In the following, we explain how to reproduce all steps from scraping the data to training the acceptance classification model.

### 1. Scrape Data
`conferences_raw/` contains directories for each conference we scraped.
The respective directory of each conference comprises multiple `*.json` files, where every file contains the information belonging to a single paper, such as the title, the abstract, the submission date and the reviews.
The reviews are stored in a list called `"review_content"`.

In `scrape_data/executables` you can find the scripts to get the data via the [OpenReview-API](https://openreview-py.readthedocs.io/en/latest/) from [OpenReview](https://openreview.net).
The following commands allow you to reproduce the data scraping for individual conferences.
The scripts create the output directories automatically, if they are not already present.

<summary>AKBC19</summary>
<details>

```bash
python 1_scrape_data/executables/scrap_data.py
--submission_invitation AKBC.ws/2019/Conference/-/Blind_Submission 
--review_invitation AKBC.ws/2019/Conference/-/Paper.*/Official_Review 
--comment_invitation AKBC.ws/2019/Conference/-/Paper.*/Official_Comment 
--metareview_invitation AKBC.ws/2019/Conference/-/Paper.*/Meta_Review 
--outdir 1_scrape_data/data
--conference_name akbc19 
--invitation AKBC.ws/2019/Conference
```
</details>


<summary>AKBC20</summary>

<details>

```bash
python 1_scrape_data/executables/scrap_data.py
--submission_invitation AKBC.ws/2020/Conference/-/Blind_Submission
--review_invitation  AKBC.ws/2020/Conference/Paper.*/-/Official_Review
--comment_invitation AKBC.ws/2020/Conference/Paper.*/-/Official_Comment
--metareview_invitation AKBC.ws/2020/Conference/Paper.*/-/Decision
--outdir 1_scrape_data/data
--conference_name akbc20 
--invitation AKBC.ws/2020/Conference
```
</details>

<summary>ICLR19 (the execution to receive the iclr19 data may take some time, because there are a lot of reviews)</summary>

<details>

```bash
python 1_scrape_data/executables/scrap_data.py
--submission_invitation ICLR.cc/2019/Conference/-/Blind_Submission
--review_invitation ICLR.cc/2019/Conference/-/Paper.*/Official_Review
--comment_invitation ICLR.cc/2019/Conference/-/Paper.*/Official_Comment
--metareview_invitation ICLR.cc/2019/Conference/-/Paper.*/Meta_Review
--outdir 1_scrape_data/data
--conference_name iclr19
--invitation ICLR.cc/2019/Conference
```
</details>

<summary>ICLR20 (the execution to receive the iclr20 data may take some time, because there are a lot of reviews)</summary>

<details>

```bash
python 1_scrape_data/executables/scrap_data.py
--submission_invitation ICLR.cc/2020/Conference/-/Blind_Submission
--review_invitation ICLR.cc/2020/Conference/Paper.*/-/Official_Review
--comment_invitation ICLR.cc/2020/Conference/Paper.*/-/Official_Comment
--metareview_invitation ICLR.cc/2020/Conference/Paper.*/-/Decision
--outdir 1_scrape_data/data
--conference_name iclr20
--invitation ICLR.cc/2020/Conference
```
</details>

<summary>MIDL19</summary>

<details>

```bash
python 1_scrape_data/executables/scrap_data.py
--submission_invitation MIDL.io/2019/Conference/-/Full_Submission
--review_invitation MIDL.io/2019/Conference/-/Paper.*/Official_Review
--comment_invitation MIDL.io/2019/Conference/-/Paper.*/Official_Comment
--metareview_invitation MIDL.io/2019/Conference/-/Paper.*/Decision
--outdir 1_scrape_data/data
--conference_name midl19
--invitation MIDL.io/2019/Conference
```
</details>


<summary>MIDL20</summary>

<details>

```bash
python 1_scrape_data/executables/scrap_data.py
--submission_invitation MIDL.io/2020/Conference/-/Blind_Submission
--review_invitation MIDL.io/2020/Conference/Paper.*/-/Official_Review
--comment_invitation MIDL.io/2020/Conference/Paper.*/-/Official_Comment
--metareview_invitation MIDL.io/2020/Conference/Paper.*/-/Meta_Review
--outdir 1_scrape_data/data
--conference_name midl20
--invitation MIDL.io/2020/Conference
```
</details>

<summary>Graph20</summary>

<details>

```bash
python 1_scrape_data/executables/scrap_data.py
--submission_invitation graphicsinterface.org/Graphics_Interface/2020/Conference/-/Blind_Submission
--review_invitation graphicsinterface.org/Graphics_Interface/2020/Conference/Paper.*/-/Official_Review
--comment_invitation None
--metareview_invitation graphicsinterface.org/Graphics_Interface/2020/Conference
--outdir 1_scrape_data/data
--conference_name graph20
--invitation graphicsinterface.org/Graphics_Interface/2020/Conference
```
</details>

<summary>NeuroAI19</summary>

<details>

```bash
python 1_scrape_data/executables/scrap_data.py 
--submission_invitation NeurIPS.cc/2019/Workshop/Neuro_AI/-/Blind_Submission 
--review_invitation NeurIPS.cc/2019/Workshop/Neuro_AI/Paper.*/-/Official_Review 
--comment_invitation NeurIPS.cc/2019/Workshop/Neuro_AI/Paper.*/-/Official_Comment 
--metareview_invitation NeurIPS.cc/2019/Workshop/Neuro_AI/Paper.*/-/Meta_Review 
--outdir 1_scrape_data/data 
--invitation NeurIPS.cc/2019/Workshop/Neuro_AI 
--conference_name neuroai19
```

Important for the NeuroAI19 script is, that you are currently in the base directory of the git, called **git**, and that the following folder structure already exists **"1_scrape_data/data/neuroai19/"**. Otherwise the execution will throw a *FileNotFoundError*.
When the path inside the script is changed, it does not have to be executed from the **git** folder.
</details>

**Note:** While we scraped data from 8 conferences, we selected a subset of 6 large conferences to continue (i.e., ICLR19/20, MIDL19/20, AKBC 19/20, Graphics20, NeuroAI19).

---

### 2. Clean Data
In a second step, we remove all unwanted character sequences from the reviews.
For details on how this works please refer to our paper.

The scripts, located in [2_clean_data/executables](./2_clean_data/executables), have to be called individually for each conference, since the file format differs across conferences.
They read all JSON files, and produce two CSV files: `X_reviews.csv` and `X_papers.csv`.
The scripts expect scraped inputs in [1_scrape_data/data/<conference>](./1_scrape_data/data) and output to [2_clean_data/data](./2_clean_data/data).

```bash
cd 2_clean_data/executables
python clean_dataset_midl19.py
```

---

### 3. Annotation Study

Next, the data was annotated by us using the web annotation tool [webanno](https://webanno.github.io/webanno/).
All scripts and data can be found in [3_annotation_study](./3_annotation_study).
---

### 4. Annotation Post-Processing
All steps to reproduce the post-processing of the annotations can be found in [4_post_processing](./4_post_processing).

The annotated data is processed with [`new_preparation.py`](./4_post_processing/executables/new_preparation.py) and creates a file which contains all annotated sentences with the majority vote.
If there is no majority, the label is `None`.
A different person who has not previously seen this review is consulted to break the tie.
These manual tie breaks are done using [`add_manual_decisions.py`](./4_post_processing/executables/add_manual_decisions.py).
Afterwards, we manually removed sentences with a low quality to facilitate the training.
The dataset resulting from this can be found at conferences_annotated folder.
For the `Argumentation` model and the `Joint` model, we take `NONARG` tag as majority whereas for the `POS_NEG` model we ignore the `NON` tag and take `CON` as majority.

In the directory [4_post_processing/exploratory_data_analysis](./4_post_processing/exploratory_data_analysis/), you can find the scripts [`human_accuracy_sentence.py`](./4_post_processing/exploratory_data_analysis/human_accuracy_sentence.py) to get the human performance on sentence level and [`human_accuracy_token.py`](./4_post_processing/exploratory_data_analysis/human_accuracy_token.py) for the human performance on token level. 

---

### 5. Work with Models
Lastly we trained a classification layer on top of different BERT models for multiple tasks and settings. 
The data of this step can be found in [5_model_works](./5_model_works).

It consists of a pre-trained BERT-model (we evaluated cased vs. uncased and BERT-base vs. BERT-large) with a classificaton layer, fine-tuned by us (fine-tuning on combinations of AURC dataset and/or AMSR dataset).


#### 5.1. Sentence Level Models

##### 5.1.1. Data Preprocessing at Sentence Level

Run the following command on `sentences_just_one_position.csv` to generate the required train-dev-test sets.

```bash
cd 5_model_works/sentence_level
python executables/s_data_preprocess.py
```

##### 5.1.2. Model Training at Sentence Level

The following command can be used to train sentence-level model:

```bash
cd 5_model_works/sentence_level
python executables/s_train.py \
--max_seq_length 100 \
--batch_size 100 \
--traindata_file train.csv \
--devdata_file val.csv \
--testdata_file test.csv \
--model bert-base-cased \
--num_labels 2 \
--learning_rate 1e-5 \
--outdir_model model/ \
--save_model true \
--finetuned false \
--epochs 10 \
--patience 3 \
--task train
```
##### 5.1.3. Model Prediction at Sentence Level

Use the following command to make predictions using the optimized model on the unlabeled reviews data file, `all_reviews_by_sentences.csv`

```bash
cd 5_model_works/sentence_level
python executables/s_predict.py
```


#### 5.2. Token Level Models

##### 5.2.1. Data Pre-processing at Token Level
Before training the models, the data folders have to be prepared using the command 

```bash
cd 5_model_works/token_level
python executables/t_data_preprocess.py
```

Ensure that before running the command, you copy "sentence_just_one_position.csv" 
and "text_sentence_majority.csv" from /4_post_processing/data and save to the raw_data_folder.
and you download "AURC_DATA.tsv" and "AURC_DOMAIN_SPLITS.tsv from 
https://github.com/trtm/AURC/tree/master/data and save them to the raw_data_folder.

##### 5.2.2. Model Training at Token Level: 
The following command can be used to train token-level model:

```bash
SEED=0
cd 5_model_works/token_level
python executables/t_modelprocessing.py \
--data_dir=data/amsr/train20 \
--models=bert-base-cased \
--task_name=classify \
--output_dir=models/classify_topic/bert-base-cased/rev20_classify_topic${SEED} \
--use_topic \
--do_train \
--use_weight \
--max_seq_length 85 \
--do_eval \
--seed ${SEED}
```

##### 5.2.3. Model Prediction at Token Level:

`t_predict.py` script is used to generate predictions on the unlabeled data file, `all_reviews_by_sentences.csv`.

---

#### 6 Argument Extraction


The following command can be used to extract the arguments from all reviews:

```bash
cd 6_arg_extraction
python executables/extract_topk.py
```
The paths for the input data and output data have been defined within the script. The same script can be used to generate k% extracted arguments at sentence level and at token level.

---

#### 7 Paper Acceptance Model using ToBERT

First, we need to fine-tune a BERT model to help generate hidden states of the extracted arguments.

##### Step 1: Data Preparation for the BERT model:

```bash
cd 7_paper_acceptance_model
python executables/p_data_preprocess.py
```

##### Step 2: Train the BERT model:

```bash
cd 7_paper_acceptance_model 
python executables/p_modelprocessing.py \ 
--data_dir=data/k_50/ \ 
--models=bert-base-cased \
--output_dir=paper_models/k_50 \
--do_train \
--do_eval \
--seed 1 \
--num_train_epochs 10
```

##### Step 3: Train the ToBERT Model:

The following command can be used to train the ToBERT model next. 

```bash
cd 7_paper_acceptance_model
python executables/ToBERT.py
```
The parameters and paths to the data and model have to be defined in the code directly.
