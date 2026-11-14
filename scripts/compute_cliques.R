#!/usr/bin/env Rscript
# Compute maximal cliques + separators of a visibility graph for the VisGP model.
#
# Matches Wang's workflow: cliques/separators are saved as R list objects
# (.rds) of 1-indexed integer vectors, readable by the Python loader
# (domain.load_cliques) and by Wang's original rpy2 path.
#
# Usage:
#   Rscript compute_cliques.R <adjacency.csv> <out_cliques.rds> <out_separators.rds>
#
# <adjacency.csv> : an n x n 0/1 matrix (no header) of the visibility graph,
#                   e.g. produced by domain.build_visibility_graph and saved with
#                   numpy.savetxt(path, A, fmt='%d', delimiter=',').
#
# Requires R packages: igraph (cliques), gRbase (junction tree / RIP ordering).

suppressMessages({
  library(igraph)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript compute_cliques.R <adjacency.csv> <cliques.rds> <separators.rds>")
}
amat_path <- args[1]; cli_out <- args[2]; sep_out <- args[3]

A <- as.matrix(read.csv(amat_path, header = FALSE))
storage.mode(A) <- "integer"
g <- graph_from_adjacency_matrix(A, mode = "undirected", diag = FALSE)

# Triangulate (chordal completion) so the junction tree is valid, then get the
# maximal cliques of the chordal graph and separators from a RIP ordering.
if (!is_chordal(g)$chordal) {
  fill <- is_chordal(g, fillin = TRUE)$fillin
  if (length(fill) > 0) {
    fill_mat <- matrix(fill, ncol = 2, byrow = TRUE)
    g <- add_edges(g, t(fill_mat))
  }
}

cl <- max_cliques(g)
cliques <- lapply(cl, function(v) as.integer(sort(as.integer(v))))

# Separators via maximum-weight spanning tree of the clique-intersection graph
# (weight = |Ci ∩ Cj|); separators are the intersections along the tree edges.
m <- length(cliques)
seps <- list()
if (m > 1) {
  cg <- make_empty_graph(n = m, directed = FALSE)
  wts <- c(); edges <- c()
  for (i in 1:(m - 1)) for (j in (i + 1):m) {
    w <- length(intersect(cliques[[i]], cliques[[j]]))
    if (w > 0) { edges <- c(edges, i, j); wts <- c(wts, w) }
  }
  if (length(edges) > 0) {
    cg <- add_edges(cg, edges, weight = wts)
    mst <- mst(cg, weights = -E(cg)$weight)  # max spanning tree
    el <- as_edgelist(mst)
    for (k in seq_len(nrow(el))) {
      s <- sort(intersect(cliques[[el[k, 1]]], cliques[[el[k, 2]]]))
      if (length(s) > 0) seps[[length(seps) + 1]] <- as.integer(s)
    }
  }
}

saveRDS(cliques, cli_out)
saveRDS(seps, sep_out)
cat(sprintf("Saved %d cliques and %d separators.\n", length(cliques), length(seps)))
