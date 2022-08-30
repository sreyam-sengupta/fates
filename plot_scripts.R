

# 


library(igraph)
library(mgcv)
library(quadprog) 
library(pcaMethods) 
library(Rcpp) 
library(inline) 
library(RcppArmadillo) 
library(pbapply)
library(glmnet)

library(crestree)


data(crest)

rownames(fpm)
gene <- "Neurog2"
gene <- "Sema3c"
gene <- "Rhob"

visualise.trajectory(ppt,gene,fpm[gene,],cex.main = 3,lwd.t2=0.5)

r <- ppt
segs <- unique(r$cell.summary$seg)
ind <- r$cell.summary$seg%in%segs
Xgene <- fpm[gene, ]
plot(r$cell.summary$t[ind], Xgene[rownames(r$cell.summary)][ind], col=adjustcolor(r$cell.summary$color[ind],0.5),pch=19, cex=0.5)

lwd.t1 <- 1
for(seg in segs ){
  indtmp <- r$cell.summary$seg == seg
  t.ord <- order(r$cell.summary$t[indtmp])
  lines(r$cell.summary$t[indtmp][t.ord],r$fit.summary[gene,rownames(r$cell.summary)][indtmp][t.ord],
        col=r$cell.summary$color[indtmp][t.ord],lwd=lwd.t1)
}

write.table(fpm, file = "fpm_output.txt", quote = FALSE, sep = "\t", row.names = TRUE, col.names = NA)

library(data.table)
data.table::fwrite(r$cell.summary, file = "cell_metadata_summary.txt", quote = FALSE, sep = "\t")
