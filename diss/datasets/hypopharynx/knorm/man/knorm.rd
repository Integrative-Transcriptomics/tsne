\name{knorm}
\alias{knorm}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Microarray Data From Multiple Biologically Interrelated Experiments}
\description{
 Produces Knorm correlations between genes (or probes) from microarray data obtained across multiple biologically interrelated experiments.  The Knorm correlation adjusts for experiment dependencies (correlations) and reduces to the Pearson coefficient when experiment dependencies are absent.  The Knorm estimation approach can be generally applicable to obtain between-row correlations from data matrices with two-way dependencies.
}
\usage{
knorm(data, bsamples, thres_diff, thres_ev1, thres_ev2, burn_in, no_subgenes, no_fullgenes, repli)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{matrix containing (normalized) gene expression data. Rows correspond to arrays and columns correspond to genes (or probes). Data from replicates of experiments are placed in consecutive rows.}
  \item{bsamples}{number of bootstrap samples for estimation.}
  \item{thres_diff}{threshold of difference between log-likelihood values.}
  \item{thres_ev1}{threshold for eigen values of experiment covariance matrix. Eigen values smaller than thres\_ev1 are considered negligible.}
  \item{thres_ev2}{threshold for eigen values of gene covariance matrix. Eigen values smaller than thres\_ev2 are considered negligible.}
  \item{burn_in}{minimum number of iterations for estimation. Default is 2.}
  \item{no_subgenes}{number of genes (or probes) to be used in the row-subsampling technique for estimating the experiment covariance matrix. This number should not be more than the number of experiments in the data.}
  \item{no_fullgenes}{number of genes (or probes) in data.}
  \item{repli}{vector of number of replicates for each experiment. For example, c(2,3) denotes two replicates for experiment 1 and three replicates for experiment 2.}
}
\details{
This estimation procedure consists of a gene (or row) sub-sampling and a covariance shrinkage technique that iteratively estimates the gene and experiment covariance matrices.  The covariance shrinkage method using the diagonal matrix with unequal covariances as the target matrix was used (Schafer and Strimmer, 2005).  For more details on the estimation procedure, model assumptions and conditions, please refer to Teng et al. (2007).}
\value{A list containing: 
  \item{a_cor_est }{Experiment correlation matrix estimate.}
  \item{g_cor_est }{Knorm correlations (between genes).}
  \item{m_est }{Mean matrix estimate.}
}
\references{Teng, S.L., Huang, H., and Zhou, X. Jasmine. (2008), "A statistical framework to infer functional gene relationships from biologically interrelated microarray experiments"}
\author{Siew Leng Teng}
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

\keyword{models}

