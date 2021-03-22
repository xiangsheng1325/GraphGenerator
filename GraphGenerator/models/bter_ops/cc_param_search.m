function p1 = cc_param_search(nd,maxcc,gcc,varargin)
%CC_PARAM_SEARCH Clustering coefficient parameter search
%
%   XI = CC_PARAM_SEARCH(ND, MAXCCD, GCC) finds the parameter XI such that
%   the clustering coefficint profile defined by
%
%   CCD(D) = MAXCCD * exp(-(D-1)*XI) for D >= 2,
%
%   has the specified global clustering coefficient (GCC) and maximum
%   clustering coefficient (MAXCCD).
%
%   Examples
%     % nd <- degree distribution
%     % maxccd_target <- target for maximum ccd value
%     % gcc_target <- target for global clustering coefficient
%     xi = cc_param_search(nd, maxccd_target, gcc_target);
%     ccd_target = [0; maxccd_target * exp(-(0:maxdeg-2)'.* xi)];
%     maxdeg = find(nd>0,1,'last');
%
%   See also DEGDIST_PARAM_SEARCH, BTER
%
%   Reference:
%   T. G. Kolda, A. Pinar, T. Plantenga and C. Seshadhri. A Scalable
%   Generative Graph Model with Community Structure, arXiv:1302.6636, 
%   March 2013. (http://arxiv.org/abs/1302.6636)
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


params = inputParser;
params.addParamValue('fminsearch_opts', optimset('TolFun', 1e-4, 'TolX', 1e-4));
params.parse(varargin{:});
options = params.Results.fminsearch_opts;

fhandle = @(x) objfunc(nd, maxcc, gcc, x);
[xstar,~,~] = fminsearch(fhandle, 0.5, options);
p1 = xstar;

function y = objfunc(nd,maxcc,gcc,xi)
%OBJFUNC Compute objectiv function, as described above
maxd = length(nd);
ccd_mean = [0; maxcc*exp(-(0:maxd-2)'.* xi)];
nWedges = nd' .* ((1:maxd).*((1:maxd)-1)/2);
gcc_xi = (nWedges*ccd_mean) / sum(nWedges);
y = abs(gcc - gcc_xi);
fprintf('xi = %e, target gcc = %f, current gcc = %f\n', xi, gcc, gcc_xi);


