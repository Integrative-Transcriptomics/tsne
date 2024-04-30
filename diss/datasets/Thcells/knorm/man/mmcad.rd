\name{mmcad}
\docType{data}
\alias{mmcad}
\title{Multiple Microarray Example Dataset - A simulated dataset example}
\description{
This is a simple simulated dataset of gene expression values for 100 genes (or probes) and 30 arrays, with 3 replicates for each array.  Values are generated from a multivariate matrix normal distribution with a specified mean matrix, covariance matrices and other (nuisance) parameters (see Teng et al., 2007 for more details).
}
\usage{mmcad}
\format{Data is represented in a matrix where rows represent arrays and columns represent genes.  Values from replicated arrays are placed in consecutive rows.}
\references{Teng, S.L., Huang, H., and Zhou, X. Jasmine. (2008), "A statistical framework to infer functional gene relationships from biologically interrelated microarray experiments".}

\examples{
#Importing simulated Multiple Microarray data.
#For more information on data set imported, look at help file for mmcd
#for futher information.
data(mmcad)

#Creating vector fo the number of replicates for each experiment.  There
#will be three replications for each experiment in the mmcd data.
repli=rep(3,30)

results <- knorm(mmcad, 25, 0.01, 1e-10, 1e-10, 2, length(repli),ncol(mmcad),repli)
a_cor_est <- results$a_cor_est
g_cor_est <- results$g_cor_est
 }
\keyword{datasets}