load_matrix <- function(path){
  expr_matrix = read.table(path,sep=',',header=T,row.names=1)
  expr_matrix = as.matrix(expr_matrix)
  return(expr_matrix)
}

load_pathway <- function(path){
  gSet=getGmt(path)
  return(gSet)
}

pathway_scoring <- function(gSet, mat_gene){
  cells_rankings <- AUCell_buildRankings(mat_gene, plotStats=FALSE)
  cells_AUC <- AUCell_calcAUC(gSet, cells_rankings, nCores=1,aucMaxRank = 300)
  aucMatrix <- getAUC(cells_AUC)
  aucMatrix = aucMatrix[rowSums(aucMatrix)>0.0,]
  return(aucMatrix)
}


AUC <- function(scPath,paPath,output_path) {

  mat_gene = load_matrix(scPath) # genes*cells
  gSet1=load_pathway(paPath) # gene set
  print(gSet1)
  gSet1=subsetGeneSets(gSet1, rownames(mat_gene)) # Subset the gene set to only include genes present in the expression matrix
  print(gSet1)
  mat_path = pathway_scoring(gSet1, mat_gene) # Scoring the gene expression matrix

  write.csv(mat_path,file=output_path)
}
