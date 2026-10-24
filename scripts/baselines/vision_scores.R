# VISION gene-set signature scores of every pseudobulk sample, for the drug-gene interaction baselines
# (tahoe_blog.py, vision_pair_inputs.py); follows https://yoseflab.github.io/VISION/articles/VISION-vignette.html
#
#   Rscript scripts/baselines/vision_scores.R <vision dir> <signatures.gmt> [<signatures.gmt> ...]
#
# Reads {tahoe,crispr}_expr.txt.gz and {tahoe,crispr}_metadata.csv of vision_inputs.py from <vision dir>, scales the
# counts of each sample to the median library size and writes {tahoe,crispr}_raw_scores.csv (samples x signatures,
# rows in the order of the metadata) to <vision dir>. The paper uses MSigDB 2025.1 GO:BP (c5.go.bp) and C6.

library(VISION)

args <- commandArgs(trailingOnly = TRUE)
vision_dir <- args[1]
signatures <- args[-1]

for (ds in c("tahoe", "crispr")) {
  counts <- read.table(file.path(vision_dir, paste0(ds, "_expr.txt.gz")), header = TRUE, sep = "\t", row.names = 1,
                       check.names = FALSE)
  meta <- read.csv(file.path(vision_dir, paste0(ds, "_metadata.csv")), row.names = 1)
  meta[] <- lapply(meta, as.factor)
  rownames(meta) <- colnames(counts)
  n.umi <- colSums(counts)
  vis <- analyze(Vision(t(t(counts) / n.umi) * median(n.umi), signatures = signatures, meta = meta))
  write.csv(getSignatureScores(vis), file.path(vision_dir, paste0(ds, "_raw_scores.csv")))
}
