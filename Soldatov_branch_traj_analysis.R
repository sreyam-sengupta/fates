# 1: Loading libraries:
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

# 2: Loading data:
data(crest)

#Loading stuff from 'crest':
emb <- crest$emb
str(emb)

clcol <- crest$clcol
str(clcol)

nc.cells <- crest$nc.cells
str(nc.cells)

#Displaying embeddings of cell clusters:
par(mfrow=c(1,1),mar=c(4,4,1,1))
plot(crest$emb,col=crest$clcol,pch=ifelse( rownames(crest$emb)%in%crest$nc.cells,19,1),cex=0.2,xlab="tSNE 1",ylab="tSNE 2")
legend("bottomright",c("neural crest","neural tube"),pch=c(19,1),cex=0.2)

#Loading the matrix of expression levels normalized to cell size:
fpm <- crest$fpm
str(fpm)

#Those adjusted to mean-variance trend:
wgm <- crest$wgm
str(wgm)

#Matrix of expression weights:
wgwm <- crest$wgwm

# 3: Running tree reconstruction:
metrics <- "cosine" 
M <- length(nc.cells) # use as many pricipal points as the number of cells

lambda <- 150
sigma <- 0.015

#Defining a tree with the 
z <- ppt.tree(X=fpm[rownames(wgm),nc.cells], emb=emb, lambda=lambda, sigma=sigma, metrics=metrics, M=M, err.cut = 5e-3, n.steps=50, seed=1, plot=FALSE)

#Displaying tree:
plotppt(z,emb,tips=FALSE,cex.tree = 0.1,cex.main=0.2,lwd.tree = 1)

#Now defining tree with the wgm matrix:
lambda <- 250
sigma <- 0.04
ppt <- ppt.tree(X=wgm[,nc.cells], W=wgwm[,nc.cells], emb=emb, lambda=250, sigma=0.04, metrics="cosine", M=M,
                err.cut = 5e-3, n.steps=30, seed=1, plot=FALSE)

plotppt(ppt,emb,tips=FALSE,cex.tree = 0.1,cex.main=0.2,lwd.tree = 1)

#Subsampling 90% of cells without replacement:
ppt_ensemble <- bootstrap.ppt(X=wgm[,nc.cells], W=wgwm[,nc.cells], emb=emb, metrics=metrics, M=as.integer(length(nc.cells)*0.9), lambda=lambda, sigma=sigma, plot=FALSE,
                              n.samples=20, seed=NULL,replace=FALSE)

#Visualizing:
plotpptl(ppt_ensemble,emb, cols=adjustcolor("grey",alpha=0.1),alpha=0.05, lwd=1)

# 4: Tree processing:
plotppt(ppt,emb,tips=TRUE,forks=FALSE,cex.tree = 0.2,lwd.tree = 2)

#Cleaning spurious branches:
ppt <- cleanup.branches(ppt,tips.remove = c(249,316))

plotppt(ppt,emb,tips=TRUE,forks=FALSE,cex.tree = 0.2,lwd.tree = 2)

write.table(ppt$cell.summary, file = "cell_summary_pseudotime_info.txt", append = FALSE, quote = FALSE, sep = "\t", row.names = TRUE, col.names = NA)

#Setting the root to provide directionality:
ppt <- setroot(ppt,root=197)

#Probabilistic distribution of a given cell on the tree:
cell <- nc.cells[2] # choose a cell
pprobs <- ppt$R[cell,] # probabilities of tree projections
plotppt(ppt,emb,pattern.tree = ppt$R[cell,],cex.tree = 1,lwd.tree = 0.1) # plot probabilities using pattern.tree parameter
points(emb[cell,1],emb[cell,2],cex=1,pch=19,col="black") # show cell position on embedding

#Maximum likelihood projection of each cell on the tree and pseudotime estimation:
ppt <- project.cells.onto.ppt(ppt,emb,n.mapping = 100)

# 5: Analysis of tree-associated genes:

#This identifies genes whose expression levels vary along the tree:
ppt <- test.associated.genes(ppt,n.map=1,fpm,summary=TRUE)

#Some summary statistics:
head(ppt$stat.association[order(ppt$stat.association$pval),])

#Defining the differentially expressed genes as those with sign=TRUE:
genes.tree <- crest$genes.tree
ppt$stat.association$sign <- FALSE
ppt$stat.association[genes.tree,]$sign <- TRUE

#Modeling expression level of DE (differentially expressed) genes as a function of pseudotime:
ppt <- fit.associated.genes(ppt,fpm,n.map=1)

#Visualizing a gene eas a function of pseudotime:
gene <- "Neurog2"
visualise.trajectory(ppt,gene,fpm[gene,],cex.main = 3,lwd.t2=0.5)

#Another way, showing how fitted expression levels change along the tree:
par(mar=c(4,4,3,1))
plotppt(ppt,emb,pattern.cell = ppt$fit.summary[gene,],gene="Neurog2",cex.main=1,cex.tree = 1.0,lwd.tree = 0.1,par=FALSE)

#selecting a subset of genes with large magnitude of variability along the tree:
genes <- rownames(ppt$stat.association)[ppt$stat.association$sign==TRUE & ppt$stat.association$A > 2]
str(genes)

#Visualizing clusters:
visualise.clusters(ppt,emb,clust.n = 10,cex.gene=1,cex.cell=0.05,cex.tree=0.2)

#Hierarchical clustering with Euclidean distance:
hc <- hclust(dist(ppt$fit.summary[genes.tree,]),method="ward.D") # hierarchical clustering
clust <- cutree(hc,10) # partition of genes in 4 clusters
str(clust)

visualise.clusters(ppt,emb,clust=clust,cex.gene=1,cex.cell=0.05,cex.tree=0.2)
#NOTE: hierarchical clustering did not work, says index out of bounds.

# 6: Analysis of subtree of interest:
#Choosing a single trajectory:
plotppt(ppt,emb[,],tips=TRUE,tree.col = ppt$pp.info$color,forks=TRUE,cex.tree = 1,lwd.tree = 0.1) # visualize tree tips

#Defining a subtree (root to terminal leaf):
zseg <- extract.subtree(ppt,c("197","103")) # select root and terminal leave of the trajectory

#Visualizing differential expression along the subtree defined above:
plotppt(ppt,emb,gene=gene,mat=fpm,cex.main=1,cex.tree = 1.5,lwd.tree = 0.1,subtree=zseg)

#Visualizing the gene as a function of pseudotime:
visualise.trajectory(ppt,gene,fpm,cex.main = 3,subtree = zseg,lwd.t2=1)

#Differential expression along the subtree:
stat.subtree <- test.associated.genes(ppt,n.map=1,fpm,subtree = zseg)

#Generating summary table of genes associated with the subtree:
head(stat.subtree[order(stat.subtree$pval),])

# 7: Inference of regulatory activity of transcription factors:
#Matrix of predicted target-TF scores:
str(crest$motmat)

#Smoothed expression levels of targets as a linear combination of unknown TF activities using lasso regression:
act <- activity.lasso(ppt$fit.list[[1]],crest$motmat)
dim(act) #'act' contains predicted activity of each TF in each cell

#Tree-projected pattern of Neurog2 activity:
tf <- "Neurog2"

par(mar=c(4,4,3,1))
plotppt(ppt,emb,pattern.cell = act[tf,],gene=tf,cex.main=0.5,cex.tree = 0.5,lwd.tree = 0.1,par=FALSE,pallete = colorRampPalette(c("darkgreen","gray50","orange")) )

# 8: Analysis of bifurcation points:
#Displaying the tree:
plotppt(ppt,emb,tips=TRUE,forks=FALSE,cex.tree = 0.2,lwd.tree = 2)

#Let's select the root and two leaves:
root <- 197
leaves <- c(103,94)

#Assessing genes differentially expressed between post-bifurcation branches:
fork.de <- test.fork.genes(ppt,fpm[,],root=root,leaves=leaves,n.mapping = 1)

#Summary statistics:
head(fork.de[order(fork.de$p),],)

# 9: Selection of optimal tree parameters:
#Selecting sigma as optimum of cross validation when lambda=0:
sig <- sig.explore(X=wgm[,nc.cells],metrics="cosine",sig.lims=seq(0.01,0.1,0.01),plot=TRUE)

#Upon getting the optimal sigma, we can select lambda using entropy criteria:
lambda.stat <- lambda.explore(X=wgm[,nc.cells],M=length(nc.cells),metrics="cosine",emb=emb,sigma=sig,base=2)

# 10: Writing the matrices fpm and wgm to a text file:
write.table(fpm, file = "fpm_output3.txt", quote = FALSE, sep = "\t", row.names = TRUE, col.names = NA)
