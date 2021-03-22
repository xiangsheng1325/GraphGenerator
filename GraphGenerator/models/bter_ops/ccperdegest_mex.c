/*  CCPERDEGEST_MEX.C: Estimates clustering coefficient for given bins
using the wedge sampling technique

For computational results for this algorithm, see 
C. Seshadhri, A. Pinar, and T.G. Kolda, 
Triadic Measures on Graphs: The Power of Wedge Sampling, 
Proc. SIAM Data Mining, May 2013. 

Tamara G. Kolda, Ali Pinar, and others, FEASTPACK v1.1, Sandia National
Laboratories, SAND2013-4136W, http://www.sandia.gov/~tgkolda/feastpack/,
January 2014  

** License **
Copyright (c) 2014, Sandia National Laboratories
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:  

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer. 

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.  

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          

Sandia National Laboratories is a multi-program laboratory managed and
operated by Sandia Corporation, a wholly owned subsidiary of Lockheed
Martin Corporation, for the U.S. Department of Energy's National Nuclear
Security Administration under contract DE-AC04-94AL85000.                                         
*/

#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory.h>

typedef double stype;

struct graph 
{
  int V;
  int E;  
  mwIndex *ind; 
  mwIndex *ptr;
};

/* 
Perfoms linear search to  find the right bin 
Inputs A: defines bin boundaries
N: number of bins
x: key for which the bin is being sought 
Output    i: the bin number such that A[i-1] < x <= A[i]
Anything less than A[0] is assigned to the first bin; 
Anything more than A[N] is assigned to the last bin
*/   

int find_bin(double *A, int N, double x)
{ 
  int i;

  for(i = 0; x >= A[i+1] && i <= N; i ++);

  if (i <= N)
  {
    return(i);
  }
  return(N);
}

/* 
Perfoms binary search to  find the right bin in the interval a[lb]--a[ub]
Assumes the entry os already in this interval
Inputs a: an array of bins
lb: the lower index for the search 
ub: the upper index for the search
s: the  key for which the bin is being sought
Output    i: the bin number such that A[i-1] < x <= A[i]
*/
int fbinary_search(stype *a, int lb, int ub, stype s) 
{
  int m;
  while (lb < ub-1) 
  {
    m = (lb + ub )/2;
    if (s < a[m])
    {
      ub = m;
    }
    else 
    {
      lb = m;
    }
  }
  return (lb);
}
/* -----------------------------------------------------------------------------
The main algorithm for  computing  clustering coefficients per degree
Inputs     G: graph in  adjacency list format 
scnt: number of samples per bin 
bcnt: number of bins 
sep: array that defines bin boundaries
cc: array that stores the clustering coefficients
----------------------------------------------------------------------------- */
void sampleByDegree(struct graph *G, int scnt, int bcnt, double *sep, double *cc)
{

  int vi, ind1, ind2, n, v, i, j, k, N, newN, *dd, *vlist, *mybin;
  mwIndex *ptr,*ind;
  double *w, x, *d, t, wcnt;

  N = G->V;  
  ptr = G->ptr; 
  ind = G->ind;  

  /* d[i] is the degree of  the ith vertex */
  d = (stype *) malloc(sizeof(stype)*N);

  /* mybin[i] is the bin of the ith vertex */
  mybin = (int *) malloc(sizeof(int)*N);

  /* dd[i] is the index of the first vertex  that is on the ith bin, on the array vlist */  
  dd = (int *) malloc(sizeof(int)*(bcnt+1));
  memset(dd, 0, sizeof(int)*(bcnt+1));

  /* Compute the degrees and bins for al vertices */
  for(i = 0; i < N; i ++) 
  {
    t = (double)(ptr[i+1]-ptr[i]);
    d[i] = ((stype) t);
    if (t > 1) 
    {
      mybin[i] = find_bin(sep, bcnt, t);
      dd[mybin[i]] ++;
    }
  }
  /* prefix sum  the dd array to set up insertion of vertices to vlist 
  in the reverse order of appearence */

  for(i = 1; i <= bcnt; i ++)
  {
    dd[i] += dd[i-1];
  }
  dd[bcnt] = dd[bcnt-1];
  newN = dd[bcnt];

  /*  vlist is an ordered list of  vertex indices such that  
  if i<= j  then mybin[vlist[i]] <=mybin[vlist[j]] */

  vlist = (int *) malloc(sizeof(int)*newN);
  for(i = 0; i < N; i ++)
  {
    if (d[i] > 1) 
    {
      vlist[--dd[mybin[i]]] = i;  
    }
  }

  /* w[i+1]-w[i] is the number of wedges centered on the ith vertex 
  This array enables uniform sampling of wedges */

  w = (stype *)malloc(sizeof(stype)*(newN+1));
  for(i = 0; i < bcnt; i ++) 
  {
    if (dd[i+1] > dd[i]) /* skip empty bins */   
    {  
      /* set up uniform wedge sampling */
      w[0] = 0; 
      n = dd[i+1] - dd[i];
      for(j = 0; j < n; j ++) 
      {
        w[j+1] = w[j] + d[vlist[dd[i]+j]]*(d[vlist[dd[i]+j]]-1);
      }
      wcnt = w[n];

      /* sample scnt wedges (with replacement) to
      compute the clustering coefficient */
      cc[i] = 0.0;
      for(j = 0; j < scnt;j ++)  
      {
        x = wcnt*((double)rand()/(double)RAND_MAX);

        vi = fbinary_search(w,0,n,x);
        v = vlist[dd[i]+vi];  /* figure which vertex this wedge is centered */ 

        /* By constructionw[v+1]-w[v]=d[v]*d[v-1], and each wedge is represented twice.
           Double representation for each  wedge is for convenience we can  either pick 
           u-v-w or w-v-u */
        ind1 = (int) floor((x-w[vi]) / (d[v]-1));  /* Pick the first vertex */
        ind2 = (int) floor((x-w[vi]-(ind1*(d[v]-1))));  /* pick the second */ 
        if (ind2 >= ind1) /* adjust for the position of the first */
        {
          ind2++;
        }
        /* convert relatives indices to actual vertex indices */
        ind1 = ind[ptr[v]+ind1];
        ind2 = ind[ptr[v]+ind2];

        /*  Check if the wedge is closed;  the "if" enables searching on the shorter list */ 
        if (d[ind1]<d[ind2]) 
        {
          for(k = G->ptr[ind1]; (k < G->ptr[ind1+1]) && (G->ind[k] != ind2); k ++);

          if (k < G->ptr[ind1+1])
          {
            cc[i] += 1.0;
          }
        }
        else 
        {
          for(k = G->ptr[ind2]; (k < G->ptr[ind2+1]) && (G->ind[k] != ind1); k ++);

          if (k < G->ptr[ind2+1])
          {
            cc[i] += 1.0;
          }
        }
      }

      /* Compute the clustering coefficient as the ratio of closed wedges */
      cc[i] = cc[i] / ((double)scnt);
    }
    else 
    {
      cc[i] = 0;
    }
  }
  free(d); free(dd); free(vlist); free(w); free(mybin);
}

/* -----------------------------------------------------------------------------
This function provides the interface to Matlab 
To call this function, you need to execute  in Matlab the following
>> mex Sp_ccperdegest_mex.c -largeArrayDims

The matlab function sould be called  as 
>> cc=Sp_ccperdegest_mex(Graph, number_of_samples,bin_boundaries)

Graph  is assumed to be a sparse matrix.
It returns cc,  such that cc[i] is the clustering coefficient of the ith bin
----------------------------------------------------------------------------- */
void mexFunction(int nlhs, mxArray *plhs[],  int nrhs, const mxArray *prhs[] )
{
  double *x, *y, *N, *dtd, *cc, *dcc, *dsep;
  int i, m, n, scnt, bcnt, *sep;
  struct graph G;

  if (nrhs != 3 || ! mxIsSparse (prhs[0]))
  {
    mexErrMsgTxt ("expects sparse matrix, #samples, and bin separators as input");
  }

  G.V = mxGetN(prhs[0]);
  G.E = mxGetNzmax(prhs [0]);
  G.ind = mxGetIr(prhs[0]);
  G.ptr = mxGetJc(prhs[0]);
  dsep = mxGetPr(prhs[1]);

  scnt = (int)(mxGetPr(prhs[2]))[0];

  bcnt = mxGetM(prhs[1]);
  if (bcnt == 1)
  {
    bcnt = mxGetN(prhs[1]);
  }

  /* Create matrix for the return argument. */
  plhs[0] = mxCreateDoubleMatrix(bcnt-1, 1, mxREAL);
  cc = mxGetPr(plhs[0]);

  sampleByDegree(&G, scnt, bcnt-1, dsep, cc); 
}
