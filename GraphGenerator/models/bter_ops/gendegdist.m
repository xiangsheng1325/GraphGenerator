function dd = gendegdist(n,pdf,cutoff)
%GENDEGDIST Create a random degree distribution from a given PDF.
%
%   ND = GENDEGDIST(N,PDF) creates a degree distribution on N nodes using
%   the discrete probability distribution function specified by PDF. The
%   result is a degree distribution: ND(d) = number of nodes of degree d. 
%
%   ND = GENDEGDIST(N,PDF,D0) estimates the number of nodes for d < DO as
%   ND(d) = PDF(d) * N. This is much faster for large N, but D0 should not
%   be too small or it will cause errors in the degree distribution.
%
%   Examples
%   maxdeg=1e5; alpha = 2; beta = 2; pdf = dglnpdf(maxdeg, alpha, beta);
%   dd = gendegdist(1e7, pdf, 1e2);
%   loglog(dd,'b*');
%
%   See also DGLNPDF, DGLNCDF.
%
%   Reference:
%   * T. G. Kolda, A. Pinar, T. Plantenga and C. Seshadhri. A Scalable
%     Generative Graph Model with Community Structure,  arXiv:1302.6636,
%     March 2013. (http://arxiv.org/abs/1302.6636)
%
% Tamara G. Kolda, Ali Pinar, and others, FEASTPACK v1.1, Sandia National
% Laboratories, SAND2013-4136W, http://www.sandia.gov/~tgkolda/feastpack/,
% January 2014  

%% License
% Copyright (c) 2014, Sandia National Laboratories
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:  
%
% # Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer. 
% # Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution.  
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          
%
%
% Sandia National Laboratories is a multi-program laboratory managed and
% operated by Sandia Corporation, a wholly owned subsidiary of Lockheed
% Martin Corporation, for the U.S. Department of Energy's National Nuclear
% Security Administration under contract DE-AC04-94AL85000. 

% ** 
if ~exist('cutoff','var')
    cutoff = 0;
end

% ** For any degree smaller than the cutoff, the PDF*n is good enough.
dd1(1:cutoff,1) = round(n*pdf(1:cutoff));
n1 = sum(dd1); %<- Number of nodes "distributed" so far.

% ** Do the tail by actual sampling
n2 = n - n1;
tailpdf = pdf(cutoff+1:end)/sum(pdf(cutoff+1:end));
tailcdf = cumsum(tailpdf);
idx2 = find(tailcdf < 1, 1, 'last');
tailcdf = [0; tailcdf(1:idx2); 1];
coins = rand(n2,1);
cnts = histc(coins,tailcdf);

% ** Assemble second half of dd
idx3 = find(cnts > 0, 1, 'last');
dd2 = cnts(1:idx3);

% **
dd = [dd1;dd2];