function [p1,p2] = degdist_param_search(avgdeg,maxdeg,varargin)
%DEGDIST_PARAM_SEARCH Find parameters for "ideal" degree distribution.
% 
%   [A,B] = DEGDIST_PARAM_SEARCH(AVG,BND) will attempt to find ideal
%   parameters for generating a discrete generalized log-normal
%   distribution with the expected average degree (AVG) and maximum degree
%   bound (BND) with probability less than 1e-10.
%
%   G = DEGDIST_PARAM_SEARCH(AVG,BND,'type','dpl') is the same as above
%   except that it will attempt to find the ideal parameter for generating
%   a discrete power law distribution.
%
%   Optional Parameters:
%   o 'type' - Type of degree distribution. Choices are discrete
%        generalized log normal ('dgln') or discrete power law ('dpl'). 
%   o 'maxdeg_prbnd' - Ideally, the probability of a node with degree BND
%        (the maximum possible) is less than this bound. Default: 1e-10.
%   o 'fminsearch_opts' - The options passed to the function fminsearch.
%        Default: optimset('TolFun', 1e-4, 'TolX', 1e-4).
%   o 'verbose' - True to print out details of the progress of the search.
%        Default: true.
%
%   See also GENDEGDIST, DGLNPDF, DPLPDF.
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
params.addParamValue('maxdeg_prbnd', 1e-10);
params.addParamValue('fminsearch_opts', optimset('TolFun', 1e-4, 'TolX', 1e-4));
params.addParamValue('type','dgln');
params.addParamValue('verbose',true);
params.parse(varargin{:});

options = params.Results.fminsearch_opts;
bnd = params.Results.maxdeg_prbnd;
type = params.Results.type;
verbose = params.Results.verbose;

if strcmp(type,'dgln')
    fhandle = @(x) dglnobjfunc(x(1), x(2), maxdeg, bnd, avgdeg, verbose);
    [xstar,fstar,exitflag] = fminsearch(fhandle, [2 2], options);
    p1 = xstar(1);
    p2 = xstar(2);
elseif strcmp(type,'dpl')
    fhandle = @(x) dplobjfunc(x, maxdeg, bnd, avgdeg, verbose);
    [xstar,fstar,exitflag] = fminsearch(fhandle, 2, options);
    p1 = xstar;
    p2 = 0;
else
    error('Invalid type');
end

if ((exitflag ~= 1) || (fstar > 0.01))
    warning('Could not find ideal solution. F(X)=%e, Exit Flag = %d.\n', fstar, exitflag);
end

function y = dglnobjfunc(alpha,beta,maxdeg,bnd,avgdeg,verbose)
%DGLNOBJFUNC Function to evaluate degree distribution
%
%    Y = DGLNOBJFUNC(ALPHA,BETA,MEXDEG,BND,AVGDEG,BND) computes a score for
%    the DGLN degree distribution with MAXDEG and parameters ALPHA and
%    BETA. The goal is that the final degree distribution should have an
%    average degree of AVGDEG and the probability of obtaining the maximum
%    degree should be less than BND. A perfect match would have a score of
%    zero.  
%
%    Y = DGLNOBJFUNC(...,VERBOSE) also indicates whether or not the function
%    should print anything. By default, VERBOSE = true.
%
%T. Kolda, November 2012.

% ** Input checking
if ~exist('verbose','var')
    verbose = true;
end

% ** Find maximum expected degree
% We want to find x such that P(random vertex has degree > x) < bnd.
p = dglnpdf(maxdeg,alpha,beta);

% Penalty should grow quickly!
if p(end) > bnd 
    y1 = (exp(1+p(end)-bnd))^2 - 1;
else
    y1 = 0;
end

% ** Find expected average degree
a = ((1:maxdeg)*p); % Compute average degree
y2 = (a-avgdeg)^2;

% ** Sum the two values
y = y1+y2;

% ** Optional printing

if verbose
    fprintf('alpha=%.3f, beta=%.3f, maxdeg=%d, p(maxdeg)=%e, avgdeg=%.1f, y=%.2f\n', ...
        alpha, beta, maxdeg, p(end), a, y);
end

function y = dplobjfunc(gamma,maxdeg,bnd,avgdeg,verbose)
%DPLOBJFUNC Function to evaluate degree distribution
%
%    Y = DPLOBJFUNC(GAMMA,MEXDEG,BND,AVGDEG,BND) computes a score for
%    the powerlaw degree distribution with MAXDEG and parameter GAMMA. The
%    goal is that the final degree distribution should have an average
%    degree of AVGDEG and the probability of obtaining the maximum degree
%    should be less than BND. A perfect match would have a score of zero.  
%
%    Y = DPLOBJFUNC(...,VERBOSE) also indicates whether or not the function
%    should print anything. By default, VERBOSE = true.
%
%T. G. Kolda and others, Sandia National Laboratories, November 2012.

% Sandia National Laboratories is a multi-program laboratory managed and
% operated by Sandia Corporation, a wholly owned subsidiary of Lockheed
% Martin Corporation, for the U.S. Department of Energy's National Nuclear
% Security Administration under contract DE-AC04-94AL85000. 

% ** Input checking
if ~exist('verbose','var')
    verbose = true;
end

% ** Find maximum expected degree
% We want to find x such that P(random vertex has degree > x) < bnd.
p = dplpdf(maxdeg,gamma);

% Penalty should grow quickly!
if p(end) > bnd 
    y1 = (exp(1+p(end)-bnd))^2 - 1;
else
    y1 = 0;
end

% ** Find expected average degree
a = ((1:maxdeg)*p); % Compute average degree
y2 = (a-avgdeg)^2;

% ** Sum the two values
y = y1+y2;

% ** Optional printing

if verbose
    fprintf('gamma=%.3f, maxdeg=%d, p(maxdeg)=%e, avgdeg=%.1f, y=%.2f\n', ...
        gamma, maxdeg, p(end), a, y);
end