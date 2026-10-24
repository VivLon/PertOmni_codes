# PertOmni

PertOmni aligns pseudobulk expression of chemical (Tahoe-100M) and genetic (CRISPRi Perturb-seq) perturbations with
LLM text embeddings of the perturbation and cell line, using one shared expression encoder and a sigmoid contrastive
loss with same-cell-line negatives.

## Setup

```bash
conda env create -f environment.yml && conda activate pertomni    # or: pip install -e .
```

Inputs are read from `data/` (processed files) and `scratch/` (raw / large files); runs are written to
`results/{save_name}/` (config, `checkpoints/best.ckpt`, logs, embeddings, scores). All three can be symlinks or set
with `PERTOMNI_DATA`, `PERTOMNI_SCRATCH`, `PERTOMNI_RESULTS`. `--overwrite` replaces a run.

The paper figures are drawn by `notebooks/Figure {2..5}/*.ipynb` from the results in `data/` (with the package
installed as above).

## 1. Preprocessing

GenePT gene embeddings: download `GenePT_emebdding_v2.zip` from [Zenodo](https://doi.org/10.5281/zenodo.10833191)
and place its files directly in `data/GenePT/` (the code reads `data/GenePT/GenePT_gene_protein_embedding_model_3_text.pickle.`,
keep the trailing dot), or pass another path with `--genept`.

```bash
# pseudobulk (summed counts per perturbation (dosage) x cell line, log-normalized)
python scripts/preprocessing/make_pseudobulk.py --dataset tahoe --inputs scratch/tahoe/plate*.h5ad.gz \
    --out data/tahoe_adata.h5ad
python scripts/preprocessing/make_pseudobulk.py --dataset crispr --inputs k562=... k562_gwps=... rpe1=... hepg2=... jurkat=... \
    hct116=... --out data/crispr_adata.h5ad
# text inputs: cell line / drug / pathway embeddings (OpenAI text-embedding-3-small, needs OPENAI_API_KEY),
# GenePT gene embeddings (data/GenePT/), MSigDB GO:BP pathway per target gene
python scripts/preprocessing/make_text_inputs.py cell_lines drugs genes pathways
# held-out-perturbation split (2/3 train, 1/6 valid, 1/6 test)
python scripts/preprocessing/make_splits.py target
```

Cell painting (RxRx3-core, image-text retrieval). Inputs of `scripts/train_rxrx3.py`; the image preprocessing
follows [CellCLIP](https://github.com/suinleelab/CellCLIP) and runs in a CellCLIP clone, in its environment:

1. **Metadata**: `metadata_rxrx3_core.csv` of [RxRx3-core](https://huggingface.co/datasets/recursionpharma/rxrx3-core)
   in `scratch/cellclip_data/rxrx3/`.
2. **Image embeddings** `scratch/cellclip_data/rxrx3_dinov2.h5` (6 instances x 5 channels x 1536 per well):
   `preprocessing/preprocess_images_rxrx3.py` stacks the 6 images of every well into one `.npz` (set its
   `output_dir`, e.g. to `scratch/cellclip_data/rxrx3_processed/`), then every channel is embedded with DINOv2-giant
   (GPU):

   ```bash
   python preprocessing/convert_npz_to_avg_emb.py --model_card facebook/dinov2-giant --dataset rxrx3-core \
       --input_dir scratch/cellclip_data/rxrx3_processed --aggregation_strategy attention --n_crop 5 \
       --output_file scratch/cellclip_data/rxrx3_dinov2.h5
   ```

3. **Well labels, split and text embeddings** (needs `OPENAI_API_KEY`):

   ```bash
   python scripts/preprocessing/make_rxrx3_inputs.py
   ```
   Produces
   - well labels `data/cellclip_data/rxrx3_results/mapping.pkl`: `{well_id: "<gene or treatment>-<address>"}`;
   - split `scratch/cellclip_data/rxrx3_results/split/cellpainting-split-{train,valid,test}-imgpermol.csv` (CellCLIP
     format, one `SAMPLE_KEY` = `well_id` per row), by perturbation: the test perturbations of
     `data/cellclip_data/ft_rxrx3_results/test_perts.pkl` (`--test_perts`; 20% drawn if the file is missing), 10% of
     all perturbations drawn for validation (`--seed`), the rest and the controls (`EMPTY_control`, `CRISPR_control`)
     for training;
   - text embeddings in `data/cellclip_data/`: `ensem_emb_drug.pickle` (compound descriptions of
     `--drug_descriptions`, from MedChemExpress, GPT-4o for the missing ones; OpenAI text-embedding-3-small; controls
     are zero vectors), `ensem_emb_gene.pickle` (GenePT, plus text-embedding-3-large embeddings of the NCBI summaries
     of C7orf26, POLR2B and INTS2, which GenePT lacks) and `ensem_emb_cellline.pickle` (GPT-4o description of HUVEC,
     text-embedding-3-small).

## 2. Text-transcriptome retrieval

```bash
# PertOmni-chem trained on Tahoe-100M
python scripts/train_single.py --save_name pertomni_chem_seed0 --dataset tahoe --split_mode target \
        --losstype siglip --temp 5 --cellline_mask hard --num_workers 6 --seed $seed
# PertOmni-cri trained on the merged CRISPR dataset
python scripts/train_single.py --save_name pertomni_cri_seed0 --dataset crispr --split_mode target \
        --losstype siglip --temp 5 --cellline_mask hard --num_workers 6 --seed $seed
# PertOmni jointly trained on both datasets
python scripts/train_multi.py --save_name pertomni_seed0 --split_mode target --losstype siglip \
    --temp_tahoe 5 --temp_crispr 5 --hardmask both --export_full --seed 0      
```

Scores: `intra_test_{tahoe,crispr}_scores.pkl` (retrieval within each cell line; `gene_*` = expression -> text,
`text_*` = text -> expression). `--export_full` saves embeddings of all samples, used by sections 3 and 4.

Baselines: `--method baseline` uses random text embeddings; `--method scgpt` / `pascient` replaces pseudobulk expression by one [scGPT](https://github.com/bowang-lab/scGPT) ([docs](https://scgpt.readthedocs.io/en/latest/)) or [PaSCient](https://github.com/genentech/pascient) embedding per sample, read from `$PERTOMNI_DATA/{tahoe,crispr}_{scgpt,pascient}.h5ad`.

**Cell morphology (RxRx3-core).** Channel mode can be set as mean_all, per_channel or concat.

```bash
python scripts/train_rxrx3.py --save_name rxrx3_mean_all_seed0 --channel_mode mean_all --enc_cl \
    --losstype siglip --temperature 5 --es_mode es_on_acc --seed 0
```

Score the saved `test_{cmp,kd}_{img,txt}_embeds.pkl` with `pertomni.evaluation.retrieval.benchmark`.

## 3. Drug-gene interaction

Drug-gene interactions: download `interactions.tsv` from [DGIdb](https://dgidb.org/downloads)
(https://dgidb.org/data/latest/interactions.tsv; the paper used the version of 10 June 2024) into
`data/gene_drug_retrieval/`, or pass another path with `--interactions`.

```bash
# DGIdb ground truth and DGI-loss id maps
python scripts/preprocessing/make_dgi_inputs.py --interactions data/gene_drug_retrieval/interactions.tsv
python scripts/make_dgi_pairs.py --retrieval drug2gene           # DGIdb pairs; gene2drug for gene -> drug
# fine-tune the text encoders on interacting pairs, score by cosine similarity
python scripts/train_multi.py --save_name pertomni_dgi_seed0 --split_mode target --losstype siglip --temp_tahoe 5 \
    --temp_crispr 5 --hardmask both --drug_target_loss ft_text_drug --ft_retrieval text \
    --init_from results/pertomni_seed0 --export_full --seed 0
python scripts/evaluate_dgi.py --kind embedding --retrieval drug2gene --use_emb text \
    --runs "results/pertomni_dgi_seed{seed}" --save_name dgi_finetuned
# baselines: classifier on PertOmni embeddings
python scripts/make_dgi_embeddings.py --run_dir results/pertomni_seed0 --retrieval drug2gene
python scripts/train_dgi.py --save_name dgi_snn_seed0 --retrieval drug2gene \
    --emb_dir results/pertomni_seed0/dgi_drug2gene --dropout 0 --lr 1e-4 --es_modes es_loss --seed 0
python scripts/evaluate_dgi.py --kind snn --retrieval drug2gene --runs "results/dgi_snn_seed{seed}" --save_name dgi_snn
```

Gene -> drug: `--drug_target_loss ft_text_gene --retrieval gene2drug`. Scores: AUROC, AUPRC, weighted recall / F1.

## 4. Perturbation prediction (CRISPRi)

kNN over training samples of the same cell line, ranked by similarity to the query's text embedding (K selected on
validation genes):

```bash
python scripts/find_test_degs.py --split target --out_dir data/langpert/deg_split_test/
python scripts/predict_perturbation_split.py --save_name pert_pred --runs "results/pertomni_seed{seed}" \
    --deg_dir data/langpert/deg_split_test/
```

Pearson correlation of predicted vs measured expression (and of the change from control), over all genes and the
top-20 DEGs.

## 5. Baselines

The one-hot, scGPT and PaSCient retrieval baselines and the classifier on PertOmni embeddings are in sections 2 and
3; the scripts of the other baselines are in `scripts/baselines/`. All of them use the splits and metrics of
PertOmni and five seeds where they are stochastic.

**Text-transcriptome retrieval: random matching.** Each test sample ranks the text embeddings of its cell line's test
samples in random order.

```bash
python scripts/baselines/random_matching.py        # results/random_matching/{tahoe,crispr}_intra_scores_seed{seed}.pkl
```

**Drug-gene interaction: VISION gene-set signatures.** [VISION](https://github.com/YosefLab/VISION) (R) scores every
pseudobulk sample for the MSigDB 2025.1 GO:BP (`c5.go.bp`) and C6 gene sets
([download](https://www.gsea-msigdb.org/gsea/msigdb)). Two baselines use these scores:

* **Tahoe blog**, reproduced from the Tahoe blog post
  [Target deconvolution through data](https://blog.tahoebio.ai/p/target-deconvolution-through-data): a (drug, gene)
  pair is scored by the cosine similarity of the drug's and the knockdown's differential signatures (mean score minus
  the control's, within each cell line), averaged over all (Tahoe cell line, CRISPRi cell type) pairs;
* **SNN + VISION**: the classifier of section 3 on the mean signature scores of each pair's drug and gene.

```bash
python scripts/baselines/vision_inputs.py --out_dir results/vision     # counts of the shared genes + metadata
Rscript scripts/baselines/vision_scores.R results/vision c5.go.bp.v2025.1.Hs.symbols.gmt c6.all.v2025.1.Hs.symbols.gmt
python scripts/baselines/tahoe_blog.py --save_name tahoe_blog_drug2gene --retrieval drug2gene --vision_dir results/vision
python scripts/baselines/vision_pair_inputs.py --vision_dir results/vision --retrieval drug2gene
python scripts/train_dgi.py --save_name dgi_snn_vision_seed0 --retrieval drug2gene \
    --emb_dir results/vision/dgi_drug2gene --use_emb all --dropout 0 --lr 1e-4 --es_modes es_loss --seed 0
python scripts/evaluate_dgi.py --kind snn --retrieval drug2gene --use_emb all \
    --runs "results/dgi_snn_vision_seed{seed}" --save_name dgi_snn_vision
```

**Perturbation prediction.** `predict_perturbation_split.py` (section 4) also runs the non-control mean, GenePert
(ridge regression on GenePT target-gene embeddings) and STRING GNN (kNN regression on
[STRING](https://string-db.org/) GNN target-gene embeddings) baselines, each with its hyperparameter selected on the
validation genes. [LangPert](https://github.com/novonordisk-research/LangPert) runs with gpt-oss-120b, single pass and
refined; *PertOmni_text + refined LangPert* restricts LangPert's candidate genes to the reference genes closest to the
query by PertOmni text-embedding retrieval, with the number of candidates selected on the validation genes. `langpert_run.py`
needs one 141 GB GPU (H200; about two days for the three cell lines) and an environment with transformers, this
package (`pip install -e . --no-deps`) and our LangPert fork, which adds the gpt-oss backend and per-gene candidate
lists ([VivLon/LangPert](https://github.com/VivLon/LangPert/tree/pertomni), branch `pertomni`:
`pip install git+https://github.com/VivLon/LangPert.git@pertomni`); the other two steps also run there.

```bash
python scripts/baselines/langpert_prep.py --run_dir results/pertomni_seed0      # queries, observed effects, candidates
python scripts/baselines/langpert_run.py --out_dir results/langpert
python scripts/baselines/langpert_score.py --langpert_dir results/langpert --deg_dir data/langpert/deg_split_test/ \
    --out_dir results/pert_pred                                                 # langpert_scores.csv
```

**Image-text retrieval: CellCLIP.** [CellCLIP](https://github.com/suinleelab/CellCLIP) on the same DINOv2-giant well
embeddings and split of RxRx3-core, zero-shot with the released checkpoint
([Hugging Face](https://huggingface.co/suinleelab/CellCLIP), `model.safetensors`) and fine-tuned on the training
wells; run from the CellCLIP repository, in its environment:

```bash
# zero-shot: writes val_img_features.pkl, val_text_features.pkl and test_perts.pkl
python retrieval_whole.py --model_type cell_clip --input_dim 1536 --loss cwcl --split_label_dir split/ \
    --ckpt_path model.safetensors --image_resolution_train 224 --image_resolution_val 224 --outdir rxrx3_results/
# fine-tuning (CWCL loss, 50 epochs), then the same retrieval with its checkpoint
accelerate launch --num_processes=4 main.py --opt_seed=0 --fine_tune_ckpt=model.safetensors --dataset=rxrx3-core \
    --img_dir="" --image_resolution_train=224 --image_resolution_val=224 --molecule_path=rxrx3-core_cell_caption.csv \
    --model_type=cell_clip --input_dim=1536 --loss=cwcl --epochs=50 --batch_size=128 --lr=2e-4 \
    --lr_scheduler=cosine-restarts --warmup=1000 --num_cycles=5 --ckpt_freq=1000 --keep_all_ckpts --log_freq=20 \
    --eval_freq=500 --outdir=ft_rxrx3_results/ --is_train --unique
python retrieval_whole.py --model_type cell_clip --input_dim 1536 --loss cwcl --split_label_dir split/ \
    --ckpt_path ft_rxrx3_results/results/rxrx3-core/models/cell_clip/<run>/ckpt_steps_00000650.pt \
    --image_resolution_train 224 --image_resolution_val 224 --outdir ft_rxrx3_results/
```

The CellCLIP features are scored with `pertomni.evaluation.retrieval.benchmark(img, txt, scale=True)` on the test
compounds and CRISPR knockouts separately.

## License

MIT, see [LICENSE](LICENSE).
