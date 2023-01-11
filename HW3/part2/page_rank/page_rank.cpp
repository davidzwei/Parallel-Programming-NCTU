#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double *score_new = new double[numNodes];
  bool converged = false;
  double global_diff = 0.0;
  double sum_no_out = 0.0;

#pragma omp parallel for reduction(+ \
                                   : sum_no_out)
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
    if (outgoing_size(g, i) == 0)
      sum_no_out += solution[i];
  }

  while (!converged)
  {
#pragma omp parallel for
    for (int vi = 0; vi < numNodes; vi++)
    {
      score_new[vi] = 0.0;

      const Vertex *start = incoming_begin(g, vi);
      const Vertex *end = incoming_end(g, vi);
      // sum over all nodes vj reachable from incoming edges
      for (const Vertex *v = start; v != end; v++)
        score_new[vi] += solution[*v] / outgoing_size(g, *v);

      score_new[vi] = (damping * score_new[vi]) + (1.0 - damping) / numNodes;
      // sum over all nodes v in graph with no outgoing edges
      score_new[vi] += damping * sum_no_out / numNodes;
    }

    // update old score and sum_no_out
    global_diff = 0.0;
    sum_no_out = 0.0;
#pragma omp parallel for reduction(+ \
                                   : global_diff, sum_no_out)
    for (int vi = 0; vi < numNodes; vi++)
    {
      global_diff += abs(score_new[vi] - solution[vi]);
      solution[vi] = score_new[vi];
      if (outgoing_size(g, vi) == 0)
        sum_no_out += solution[vi];
    }

    converged = (global_diff < convergence);
  }

  delete[] score_new;

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}

