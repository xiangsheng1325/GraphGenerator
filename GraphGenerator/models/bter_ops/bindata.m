function [xx,yy,bins] = bindata(x,y,varargin)
%BINDATA Exponential data binning.
%
%   [XX,YY] = BINDATA(X,Y) logarithmically bins data. We assume both X and
%   Y are column vectors of the same length. By default, the k-th bin is
%   [2^k, 2^k-1]. The return values are defined as follows: XX(k) = 2^k is
%   the "bin label" and YY(k) = sum_i { Y(i) s.t. 2^k <= X(i) < 2^(k-1) }
%   is the "bin value".
%
%   [XX,YY] = BINDATA([],Y) sets X = (1:length(Y))'.
%
%   [XX,YY,BB] = BINDATA(...) returns the data bins, i.e., the k-th bin is
%   defined by [BB(k), BB(k+1)-1]. This can be useful if the meaning of XX
%   is modified by the parameters defined below.
%
%   [XX,YY,BB] = BINDATA(X,Y,'param',value,...) also accepts
%   parameter-value pairs, as described below.
%
%      --- Bin Definitions ---
%      The start of the k-th bin is given by 
%        BB(k) = k + idx0 - 1 if k <= tau, else 
%        BB(k) = ceil((omega.^(k-tau)-1)/(omega-1)) + tau + idx0 - 1.
%
%      o 'omega' - Bin increase multiplier. Default: 2.
%      o 'tau' - Number of singleton bins. Default: 1.
%      o 'idx0' - Starting index to be binned. Default: 1.
%
%      --- Binning Behavior ---
%      o 'bin' - Do binning? If false, returns X and Y unchanged unless X
%        was empty on input, in which case it's been reset to
%        (1:length(Y)). Default: true.    
%      o 'ybinfun' - Function for the "bin value", used to combine all the
%        y-values in the same bin. Default: @sum. 
%      o 'xbinfun' - Function for the "bin index". By default, XX(k)=BB(k).
%        If a function is specified, however, then this is used to combine
%        all the values in the same bin. Specifying @mean, for instance,
%        gives a weighted mean of the x-value as the bin index. 
%        Default: [] (indicate to use the bin starts).
% 
%      --- Preprocessing ---
%      o 'prebin' - Collect values together for same x. This has the side
%        effect of ensuring the x values are dense, even for zero y values.
%        Default: false.  
%      o 'prebinfun' - Specified function to combine values with same x.
%        Default: @mean. 
%
%      --- Postprocessing ---
%      o 'nozeros' - Remove any zero yy-values (and corresponding xx) from
%        the output. Default: false. 
%
%   EXAMPLES
%   y = [10 8 6 0 4]';
%   [xx,yy] = bindata([],y) % Create 3 bins and gives total per bin.
%   [xx,yy] = bindata([],y,'bin',false) % Returns xx = (1:5)' and yy = y.
%   x = [2 3 5 5 6]'; 
% 
%   See also BINLOOKUP, BINSTART.
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

% ** Parse inputs
params = inputParser;
params.addParamValue('bin', true);
params.addParamValue('prebin', false);
params.addParamValue('prebinfun', @mean);
params.addParamValue('omega', 2);
params.addParamValue('tau', 1);
params.addParamValue('idx0', 1);
params.addParamValue('xbinfun', []);
params.addParamValue('ybinfun', @sum);
params.addParamValue('nozeros', false);
params.parse(varargin{:});

binparams = {params.Results.omega, params.Results.tau, params.Results.idx0};
% ** Check and fix empty x
if isempty(x)
    x = (1:length(y))';
end

% ** Make sure both x and y are column vectors
x = reshape(x,[],1);
y = reshape(y,[],1);

% ** Check inputs are the same length
if numel(x) ~= numel(y)
    error('Input vectors are not the same length');
end

% ** Check for no binning
if ~params.Results.bin % No binning
    xx = x;
    yy = y;
    bins = [];
    return;
end

% ** Number of bins?
nbins = binlookup(max(x), binparams{:});

% ** Pre-binning?
% Pre-binning creates dense x and y arrays, with an entry for every
% possible x-value. If there are multiple copies of x, then the default is
% to take the mean of the associated y-values.
if params.Results.prebin
    xmax = binstart(nbins+1, binparams{:})-1;  
    y = accumarray(x,y,[xmax 1],params.Results.prebinfun);
    x = (1:xmax)';
end

% ** Determine xx
idx = binlookup(x, binparams{:});
if isempty(params.Results.xbinfun)
    xx = binstart((1:nbins)', binparams{:});
else
    xx = accumarray(idx, x, [], params.Results.xbinfun);
end
yy = accumarray(idx, y, [], params.Results.ybinfun);
bins = binstart((1:(nbins+1))', binparams{:});

% ** Remove zero entries?
if params.Results.nozeros
    tf = yy > 0;
    yy = yy(tf);
    xx = xx(tf);
end