
% step1
load('###{Template Block}###');
nnodes = size(G,1);
nedges = nnz(G)/2;
fprintf('Graph name: %s\n', graphname);
fprintf('Number of nodes: %d\n', nnodes);
fprintf('Number of edges: %d\n', nedges);

% step2
nd = accumarray(nonzeros(sum(G,2)),1);
maxdegree = find(nd>0,1,'last');
fprintf('Maximum degree: %d\n', maxdegree);

% step3
[ccd,gcc] = ccperdeg(G);
fprintf('Global clustering coefficient: %.2f\n', gcc);

% step4
fprintf('Running BTER...\n');
t1=tic;
[E1,E2] = bter(nd,ccd);
toc(t1)
fprintf('Number of edges created by BTER: %d\n', size(E1,1) + size(E2,1));

% step5
fprintf('Turning edge list into adjacency matrix (including dedup)...\n');
t2=tic;
G_bter = bter_edges2graph(E1,E2);
toc(t2);
fprintf('Number of edges in dedup''d graph: %d\n', nnz(G)/2);

save('###{Template Block}###','G_bter')