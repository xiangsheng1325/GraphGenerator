function idx = binstart(i, omega, tau, idx0)
%BINSTART - Specify start of bin for the specified parameters.
%
%   K = BINSTART(I,OMEGA,TAU,K0) returns the index of the I-th bin defined
%   by parameters TAU, OMEGA, and K0. The parameters TAU, OMEGA, and K0 are
%   optional. The default values are OMEGA=2, TAU=1, K0=1.
%
%   The end of a bin I one less than the end of the next bin, i.e., 
%   KEND = BINSTART(I+1,OMEGA,TAU,K0)-1.
%
%   See also BINLOOKUP, BINDATA.
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
n = length(i);
idx = zeros(n,1);
for k = 1:n
    if i(k) <= tau
        idx(k) = i(k) + idx0 - 1;
    else
        idx(k) = ceil((omega.^(i(k)-tau)-1)/(omega-1)) + tau + idx0 - 1;
    end
end