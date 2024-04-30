library(MASS)

#### ---------------------------------------------------------------------------------
knorm <- function (data, bsamples, thres_diff, thres_ev1, thres_ev2, burn_in=2, no_subgenes, no_fullgenes, repli) {

	a <- form.mat(data, repli)
	A <- a

      ans1 <- matrix(0, nrow=bsamples, ncol=(length(A))^2)
      ans1_1 <- matrix(0, nrow=bsamples, ncol=(length(A))^2)
      ans2 <- matrix(0, nrow=bsamples, ncol=(ncol(A[[1]]))^2)
      ans3 <- matrix(0, nrow=bsamples, ncol=ncol(A[[1]]))
	a <- unlist(lapply(A, function(x){ifelse(is.null(nrow(x)), 1, nrow(x))}))
	for (i in 1 : bsamples) {		h <- matrix(0, nrow=length(A), ncol=ncol(A[[1]]))

		for (j in 1 : nrow(h)) {
			h[j,] <- A[[j]][sample(1:a[j], 1),]
		}
		h2 <- cor(t(h))

		sub_genes <- sample(1:no_fullgenes, no_subgenes)
		h <- h[, sub_genes]
		g <- iter_est(h, h2, thres_diff, thres_ev1, thres_ev2, burn_in)
		ans1[i, ] <- as.vector(cov2cor(g[[1]]))
		ans1_1[i, ] <- as.vector(g[[1]])
	}
      ans1 <- matrix(apply(ans1, 2, mean), byrow = FALSE, ncol = nrow(h))
	ans1_1 <- matrix(apply(ans1_1, 2, mean), byrow = FALSE, ncol = nrow(h))

	cat(c("..... estimating Knorm correlations"))
	
	b <- rep(1, nrow(h))
	b2 <- ginv(ans1_1)
	b2[row(b2) > col(b2)] <- t(b2)[row(t(b2)) > col(t(b2))]

	for (i in 1 : bsamples) {		
		h <- matrix(0, nrow=length(A), ncol=ncol(A[[1]]))

		for (j in 1 : nrow(h)) {
			h[j,] <- A[[j]][sample(1:a[j], 1),]
		}
		m2 <- (as.numeric((t(h)) %*% b2 %*% b))/as.numeric(((t(b)) %*% b2 %*% b))
		ans3[i,] <- m2

		y <- matsqrt(ans1_1)
		y <- y %*% (h - t(m2 %*% (t(b))))
		g2 <- cov_shrink(y)
		ans2[i,] <- as.numeric(cov2cor(g2))

		if (i/bsamples == 0.2) {cat(c("\n", "..... 20% completed"))}
		if (i/bsamples == 0.4) {cat(c("\n", "..... 40% completed"))}
		if (i/bsamples == 0.6) {cat(c("\n", "..... 60% completed"))}
		if (i/bsamples == 0.8) {cat(c("\n", "..... 80% completed"))}
		if (i/bsamples == 1) {cat(c("\n", "..... 100% completed"))}
	}
	ans2 <- matrix(apply(ans2, 2, mean), byrow = FALSE,  ncol = ncol(h))
	ans3 <- t(apply(ans3, 2, mean) %*% t(rep(1, nrow(h))))

	ans <- setNames(list(ans1, ans2, ans3), c("a_cor_est", "g_cor_est", "m_est"))

	return(ans)
}

#### ---------------------------------------------------------------------------------
iter_est <- function (A, B, thres_diff, thres_ev1, thres_ev2, burn_in) {

	ai <- B
	a <- ginv(ai)
	a[row(a) > col(a)] <- t(a)[row(t(a)) > col(t(a))]

	b <- rep(1, nrow(B))
	mi <- (as.numeric((t(A)) %*% a %*% b))/as.numeric(((t(b)) %*% a %*% b))
	mi_temp <- t(mi %*% (t(b)))

	y <- matsqrt(ai)
	y <- y %*% (A - mi_temp)
	gi <- cov_shrink(y)

	ait_prev <- ai
	mit_prev <- mi
	mit_temp_prev <- mi_temp
	git_prev <- gi

	aa <- ginv(git_prev)
	aa[row(aa) > col(aa)] <- t(aa)[row(t(aa)) > col(t(aa))]

	no_iter <- 1
	gamma <- gamma_mat(ait_prev, git_prev, thres_ev1, thres_ev2, mit_temp_prev, A)
	l <- compute_logl(gamma)

	flag <- 1
	while (flag) {
		no_iter <- no_iter + 1
	        b <- rep(1, nrow(B))

		aa <- ginv(git_prev)
		aa[row(aa) > col(aa)] <- t(aa)[row(t(aa)) > col(t(aa))]

		ait_now <- (A - mit_temp_prev) %*% aa %*% (t(A - mit_temp_prev))
		ait_now <- ait_now/ncol(git_prev)

		a <- ginv(ait_now)
		a[row(a) > col(a)] <- t(a)[row(t(a)) > col(t(a))]

		mit_now <- (as.numeric((t(A)) %*% a %*% b))/as.numeric(((t(b)) %*% a %*% b))
		mit_temp_now <- t(mit_now %*% (t(b)))

		y <- matsqrt(ait_now)
		y <- y %*% (A - mit_temp_now)
		git_now <- cov_shrink(y)

		aa <- ginv(git_now)
		aa[row(aa) > col(aa)] <- t(aa)[row(t(aa)) > col(t(aa))]

	      a_violate <- sum(diag(ait_now) <= 0)
        	g_violate <- sum(diag(git_now) <= 0)

		if (a_violate == 0 & g_violate == 0) {
			gamma <- gamma_mat(ait_now, git_now, thres_ev1, thres_ev2, mit_temp_now, A)
			l <- append(l, compute_logl(gamma))

			if (length(gamma) != 1) {
				if (no_iter <= burn_in) {
					flag <- 1
				}
				if (no_iter > burn_in) {
					flag <- ifelse((l[no_iter] - l[(no_iter - 1)]) > thres_diff, 1, 0)
				}
				if (flag == 1) {
					mit_temp_prev <- mit_temp_now
					mit_prev <- mit_now
					ait_prev <- ait_now
					git_prev <- git_now
				}
				if (flag == 0) {
					mit_temp_now <- mit_temp_prev
					mit_now <- mit_prev
					ait_now <- ait_prev
					git_now <- git_prev

					a <- ginv(ait_now)
					a[row(a) > col(a)] <- t(a)[row(t(a)) > col(t(a))]

					aa <- ginv(git_now)
					aa[row(aa) > col(aa)] <- t(aa)[row(t(aa)) > col(t(aa))]

					y <- matsqrt(ait_now)
					y <- y %*% (A - mit_temp_now)

					no_iter <- no_iter - 1
					l <- l[1:no_iter]
				}
			}

			if (length(gamma) == 1) {
				flag <- 0
				mit_temp_now <- mit_temp_prev
				mit_now <- mit_prev

				ait_now <- ait_prev
				a <- ginv(ait_now)
				a[row(a) > col(a)] <- t(a)[row(t(a)) > col(t(a))]

				git_now <- git_prev
				aa <- ginv(git_now)
				aa[row(aa) > col(aa)] <- t(aa)[row(t(aa)) > col(t(aa))]

				y <- matsqrt(ait_now)
				y <- y %*% (A - mit_temp_now)

				no_iter <- no_iter - 1
				l <- l[1:no_iter]
			}
		}

		if (!(a_violate == 0 & g_violate == 0)) {
			flag <- 0
			mit_temp_now <- mit_temp_prev
			mit_now <- mit_prev
			ait_now <- ait_prev
			git_now <- git_prev

			a <- ginv(ait_now)
			a[row(a) > col(a)] <- t(a)[row(t(a)) > col(t(a))]

			aa <- ginv(git_now)
			aa[row(aa) > col(aa)] <- t(aa)[row(t(aa)) > col(t(aa))]

			y <- matsqrt(ait_now)
			y <- y %*% (A - mit_temp_now)

			no_iter <- no_iter - 1
			l <- l[1:no_iter]
		}
	}

	ans <- setNames(list(ait_now), c("a.cov.iterated"))

	return(ans)
}

#### ---------------------------------------------------------------------------------
matsqrt <- function(A) {
	e <- ginv(A)
	e[row(e) > col(e)] <- t(e)[row(t(e)) > col(t(e))]

	a <- eigen(e)
	a_values <- a$values
	b <- a_values[which(!(a_values > 0))]

	a_values[which(!(a_values > 0))] <- 0
	d <- diag(sqrt(a_values))
	p <- a$vectors

	ans <- p %*% d %*% ginv(p)

	return(ans)
}

#### ----------------------------------------------------------------
cov_shrink <- function(A) {

	m <- apply(A, 2, mean)
	m2 <- rep(1, nrow(A)) %*% t(m)
	f <- cov(A)

	b <- A - m2
	w <- array(0, c(ncol(A), ncol(A), nrow(A)))
	w2 <- matrix(0, nrow=ncol(A), ncol=ncol(A))
	s <- w2

	for (i in 1 : ncol(A)) {
		for (j in 1 : ncol(A)) {
			for (k in 1 : nrow(A)) {
				w[i,j,k] <- b[k,i] * b[k,j]
			}
			w2[i,j] <- mean(w[i,j,])
			s[i,j] <- s[i,j] + (w[i,j,k] - w2[i,j])^2
		}
	}
	s <- (ncol(A)/(ncol(A) - 1)^3)*s

	a <- apply(A, 2, var)
	T <- diag(a)

	e <- s
	diag(e) <- 0
	g <- f
	diag(g) <- 0
	l <- sum(e)/ sum(g^2)
	if (l > 1 | l < 0) {l <- max(0, min(1, l))}

	ans <- l*T + (1-l)*f

	return(ans)
}

#### ----------------------------------------------------------------
gamma_mat <- function(A, B, E, thres_ev1, G, H) {

	ans <- NaN

	H <- t(H)
	G <- t(G)
	svd_e <- eigen(A)
	svd_g <- eigen(B)
	p <- svd_e$values
	v <- svd_e$vectors
	d <- svd_g$values
	u <- svd_g$vectors

	pp <- p
	pp[abs(pp) <= E] <- 0
	dd <- d
	dd[abs(dd) <= thres_ev1] <- 0
	pp <- as.numeric(pp)
	dd <- as.numeric(dd)

	if (sum(pp >= 0) == length(pp) &  sum(dd >= 0) == length(dd)) {
		pp <- ginv(diag(sqrt(pp)))
		dd <- ginv(diag(sqrt(dd)))
		vv <- v
		uu <- u

		ans <- dd %*% t(uu) %*% (H - G) %*% (vv %*% pp)
		ans <- ans[which(abs(d) > thres_ev1), which(abs(p) > E)]
	}

	return(ans)
}


#### ----------------------------------------------------------------
compute_logl <- function(A) {

	ans <- sum(-log(sqrt(2*pi)) -1/2*(as.numeric(A))^2)

	return(ans)
}

#### -----------------------------------------------------------------
form.mat <- function(A, B) {
	
	ans <- list(A[1:B[1], ])
	for (i in 2: length(B)) {
		ans <- append(ans, list(A[(sum(B[1 : (i-1)]) + 1):sum(B[1:i]), ]))
	}
	
	return(ans)
}

setwd('/Users/zabel/projects/tsne/')
d <- read.csv('diss/datasets/hypopharynx/GSE2379_all_samples_log_qnorm_sorted.txt', sep = '\t', header = 1)
selected_genes <- d[, order(apply(d[-1, ], 2, var), decreasing=TRUE)][, 1:50]
selected_genes_m <- as.matrix(selected_genes)

repli=c(4, 4, 15, 11, 4)
results <- knorm(selected_genes_m, 25, 0.01, 1e-10, 1e-10, 2, length(repli),ncol(selected_genes_m),repli)

a_cor_est <- results$a_cor_est
g_cor_est <- results$g_cor_est
mean <- results$m_est

write.csv(selected_genes_m, 'diss/datasets/hypopharynx/selected_data.txt', row.names = FALSE)
write.csv(mean, 'diss/datasets/hypopharynx/mean.txt', row.names = FALSE)
write.csv(a_cor_est, 'diss/datasets/hypopharynx/cov1.txt', row.names = FALSE)
write.csv(g_cor_est, 'diss/datasets/hypopharynx/cov2.txt', row.names = FALSE)
