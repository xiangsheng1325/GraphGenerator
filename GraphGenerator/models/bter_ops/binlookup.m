function i = binlookup(idx, omega, tau, idx0)
%BINLOOKUP For a given index, determine its appropriate bin.
%
%   I = BINLOOKUP(K,OMEGA,TAU,K0) returns the bin number of index K, where
%   the bins are defined by paramtesr TAU, OMEGA, and K0. The parameters
%   TAU, OMEGA, and K0 are  optional. If they are not defined or defined as
%   an emptyset ([]), then they take on the default values, which are
%   OMEGA=2, TAU=1, K0=1. 
%
%   Note: If K is a vector, than I is a vector of bins.
%
%   See also BINSTART, BINDATA.
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
if ~exist('omega','var') || isempty(omega)
    omega = 2;
end

if ~exist('tau','var') || isempty(tau)
    tau = 1;
end

if ~exist('idx0','var') || isempty(idx0)
    idx0 = 1;
end

% **
if any(idx < idx0)
    error('Index is smaller than the start of the first bin');
end

n = length(idx);
i = zeros(n,1);
for k = 1:n
    if (idx(k)-idx0+1) < tau
        i(k) = idx(k)-idx0+1;
    else
        tmp = 1 + (omega-1)*(idx(k)-idx0+1 - tau);
        i(k) = floor(log(tmp)/log(omega)) + tau ;
    end
end