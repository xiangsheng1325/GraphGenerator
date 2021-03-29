#include "stdafx.h"
#include "kronecker.h"

//
//
const double TKronMtx::NInf = -DBL_MAX;
TRnd TKronMtx::Rnd = TRnd(0);

TKronMtx::TKronMtx(const TFltV& SeedMatrix) : SeedMtx(SeedMatrix) {
  MtxDim = (int) sqrt((double)SeedMatrix.Len());
  IAssert(MtxDim*MtxDim == SeedMtx.Len());
}

void TKronMtx::SaveTxt(const TStr& OutFNm) const {
  FILE *F = fopen(OutFNm.CStr(), "wt");
  for (int i = 0; i < GetDim(); i++) {
    for (int j = 0; j < GetDim(); j++) {
      if (j > 0) fprintf(F, "\t");
      fprintf(F, "%f", At(i,j)); }
    fprintf(F, "\n");
  }
  fclose(F);
}

TKronMtx& TKronMtx::operator = (const TKronMtx& Kronecker) {
  if (this != &Kronecker){
    MtxDim=Kronecker.MtxDim;
    SeedMtx=Kronecker.SeedMtx;
  }
  return *this;
}

bool TKronMtx::IsProbMtx() const {
  for (int i = 0; i < Len(); i++) {
    if (At(i) < 0.0 || At(i) > 1.0) return false;
  }
  return true;
}

void TKronMtx::SetRndMtx(const int& PrmMtxDim, const double& MinProb) {
  MtxDim = PrmMtxDim;
  SeedMtx.Gen(MtxDim*MtxDim);
  for (int p = 0; p < SeedMtx.Len(); p++) {
    do {
      SeedMtx[p] = TKronMtx::Rnd.GetUniDev();
    } while (SeedMtx[p] < MinProb);
  }
}

void TKronMtx::SetEpsMtx(const double& Eps1, const double& Eps0, const int& Eps1Val, const int& Eps0Val) {
  for (int i = 0; i < Len(); i++) {
    double& Val = At(i);
    if (Val == Eps1Val) Val = double(Eps1);
    else if (Val == Eps0Val) Val = double(Eps0);
  }
}

//
void TKronMtx::SetForEdges(const int& Nodes, const int& Edges) {
  const int KronIter = GetKronIter(Nodes);
  const double EZero = pow((double) Edges, 1.0/double(KronIter));
  const double Factor = EZero / GetMtxSum();
  for (int i = 0; i < Len(); i++) {
    At(i) *= Factor;
    if (At(i) > 1) { At(i) = 1; }
  }
}

void TKronMtx::AddRndNoise(const double& SDev) {
  Dump("before");
  double NewVal;
  int c =0;
  for (int i = 0; i < Len(); i++) {
    for(c = 0; ((NewVal = At(i)*Rnd.GetNrmDev(1, SDev, 0.8, 1.2)) < 0.01 || NewVal>0.99) && c <1000; c++) { }
    if (c < 999) { At(i) = NewVal; } else { printf("XXXXX\n"); }
  }
  Dump("after");
}

TStr TKronMtx::GetMtxStr() const {
  TChA ChA("[");
  for (int i = 0; i < Len(); i++) {
    ChA += TStr::Fmt("%g", At(i));
    if ((i+1)%GetDim()==0 && (i+1<Len())) { ChA += "; "; }
    else if (i+1<Len()) { ChA += ", "; }
  }
  ChA += "]";
  return TStr(ChA);
}

void TKronMtx::ToOneMinusMtx() {
  for (int i = 0; i < Len(); i++) {
    IAssert(At(i) >= 0.0 && At(i) <= 1.0);
    At(i) = 1.0 - At(i);
  }
}

void TKronMtx::GetLLMtx(TKronMtx& LLMtx) {
  LLMtx.GenMtx(MtxDim);
  for (int i = 0; i < Len(); i++) {
    if (At(i) != 0.0) { LLMtx.At(i) = log(At(i)); }
    else { LLMtx.At(i) = NInf; }
  }
}

void TKronMtx::GetProbMtx(TKronMtx& ProbMtx) {
  ProbMtx.GenMtx(MtxDim);
  for (int i = 0; i < Len(); i++) {
    if (At(i) != NInf) { ProbMtx.At(i) = exp(At(i)); }
    else { ProbMtx.At(i) = 0.0; }
  }
}

void TKronMtx::Swap(TKronMtx& KronMtx) {
  ::Swap(MtxDim, KronMtx.MtxDim);
  SeedMtx.Swap(KronMtx.SeedMtx);
}

int TKronMtx::GetNodes(const int& NIter) const {
  return (int) pow(double(GetDim()), double(NIter));
}

int TKronMtx::GetEdges(const int& NIter) const {
  return (int) pow(double(GetMtxSum()), double(NIter));
}

int TKronMtx::GetKronIter(const int& Nodes) const {
  return (int) ceil(log(double(Nodes)) / log(double(GetDim()))); //
  //
}

int TKronMtx::GetNZeroK(const PNGraph& Graph) const {
 return GetNodes(GetKronIter(Graph->GetNodes()));
}

double TKronMtx::GetEZero(const int& Edges, const int& KronIters) const {
  return pow((double) Edges, 1.0/double(KronIters));
}

double TKronMtx::GetMtxSum() const {
  double Sum = 0;
  for (int i = 0; i < Len(); i++) {
    Sum += At(i); }
  return Sum;
}

double TKronMtx::GetRowSum(const int& RowId) const {
  double Sum = 0;
  for (int c = 0; c < GetDim(); c++) {
    Sum += At(RowId, c); }
  return Sum;
}

double TKronMtx::GetColSum(const int& ColId) const {
  double Sum = 0;
  for (int r = 0; r < GetDim(); r++) {
    Sum += At(r, ColId); }
  return Sum;
}

double TKronMtx::GetEdgeProb(int NId1, int NId2, const int& NKronIters) const {
  double Prob = 1.0;
  for (int level = 0; level < NKronIters; level++) {
    Prob *= At(NId1 % MtxDim, NId2 % MtxDim);
    if (Prob == 0.0) { return 0.0; }
    NId1 /= MtxDim;  NId2 /= MtxDim;
  }
  return Prob;
}

double TKronMtx::GetNoEdgeProb(int NId1, int NId2, const int& NKronIters) const {
  return 1.0 - GetEdgeProb(NId1, NId2, NKronIters);
}

double TKronMtx::GetEdgeLL(int NId1, int NId2, const int& NKronIters) const {
  double LL = 0.0;
  for (int level = 0; level < NKronIters; level++) {
    const double& LLVal = At(NId1 % MtxDim, NId2 % MtxDim);
    if (LLVal == NInf) return NInf;
    LL += LLVal;
    NId1 /= MtxDim;  NId2 /= MtxDim;
  }
  return LL;
}

double TKronMtx::GetNoEdgeLL(int NId1, int NId2, const int& NKronIters) const {
  return log(1.0 - exp(GetEdgeLL(NId1, NId2, NKronIters)));
}

//
double TKronMtx::GetApxNoEdgeLL(int NId1, int NId2, const int& NKronIters) const {
  const double EdgeLL = GetEdgeLL(NId1, NId2, NKronIters);
  return -exp(EdgeLL) - 0.5*exp(2*EdgeLL);
}

bool TKronMtx::IsEdgePlace(int NId1, int NId2, const int& NKronIters, const double& ProbTresh) const {
  double Prob = 1.0;
  for (int level = 0; level < NKronIters; level++) {
    Prob *= At(NId1 % MtxDim, NId2 % MtxDim);
    if (ProbTresh > Prob) { return false; }
    NId1 /= MtxDim;  NId2 /= MtxDim;
  }
  return true;
}

//
double TKronMtx::GetEdgeDLL(const int& ParamId, int NId1, int NId2, const int& NKronIters) const {
  const int ThetaX = ParamId % GetDim();
  const int ThetaY = ParamId / GetDim();
  int ThetaCnt = 0;
  for (int level = 0; level < NKronIters; level++) {
    if ((NId1 % MtxDim) == ThetaX && (NId2 % MtxDim) == ThetaY) {
      ThetaCnt++; }
    NId1 /= MtxDim;  NId2 /= MtxDim;
  }
  return double(ThetaCnt) / exp(At(ParamId));
}

//
double TKronMtx::GetNoEdgeDLL(const int& ParamId, int NId1, int NId2, const int& NKronIters) const {
  const int& ThetaX = ParamId % GetDim();
  const int& ThetaY = ParamId / GetDim();
  int ThetaCnt = 0;
  double DLL = 0, LL = 0;
  for (int level = 0; level < NKronIters; level++) {
    const int X = NId1 % MtxDim;
    const int Y = NId2 % MtxDim;
    const double LVal = At(X, Y);
    if (X == ThetaX && Y == ThetaY) {
      if (ThetaCnt != 0) { DLL += LVal; }
      ThetaCnt++;
    } else { DLL += LVal; }
    LL += LVal;
    NId1 /= MtxDim;  NId2 /= MtxDim;
  }
  return -ThetaCnt*exp(DLL) / (1.0 - exp(LL));
}

//
double TKronMtx::GetApxNoEdgeDLL(const int& ParamId, int NId1, int NId2, const int& NKronIters) const {
  const int& ThetaX = ParamId % GetDim();
  const int& ThetaY = ParamId / GetDim();
  int ThetaCnt = 0;
  double DLL = 0;//
  for (int level = 0; level < NKronIters; level++) {
    const int X = NId1 % MtxDim;
    const int Y = NId2 % MtxDim;
    const double LVal = At(X, Y); IAssert(LVal > NInf);
    if (X == ThetaX && Y == ThetaY) {
      if (ThetaCnt != 0) { DLL += LVal; }
      ThetaCnt++;
    } else { DLL += LVal; }
    //
    NId1 /= MtxDim;  NId2 /= MtxDim;
  }
  //
  //
  //
  return -ThetaCnt*exp(DLL) - ThetaCnt*exp(At(ThetaX, ThetaY)+2*DLL);
}

uint TKronMtx::GetNodeSig(const double& OneProb) {
  uint Sig = 0;
  for (int i = 0; i < (int)(8*sizeof(uint)); i++) {
    if (TKronMtx::Rnd.GetUniDev() < OneProb) {
      Sig |= (1u<<i); }
  }
  return Sig;
}

double TKronMtx::GetEdgeProb(const uint& NId1Sig, const uint& NId2Sig, const int& NIter) const {
  Assert(GetDim() == 2);
  double Prob = 1.0;
  for (int i = 0; i < NIter; i++) {
    const uint Mask = (1u<<i);
    const uint Bit1 = NId1Sig & Mask;
    const uint Bit2 = NId2Sig & Mask;
    Prob *= At(int(Bit1!=0), int(Bit2!=0));
  }
  return Prob;
}

PNGraph TKronMtx::GenThreshGraph(const double& Thresh) const {
  PNGraph Graph = TNGraph::New();
  for (int i = 0; i < GetDim(); i++) {
    Graph->AddNode(i); }
  for (int r = 0; r < GetDim(); r++) {
    for (int c = 0; c < GetDim(); c++) {
      if (At(r, c) >= Thresh) { Graph->AddEdge(r, c); }
    }
  }
  return Graph;
}

PNGraph TKronMtx::GenRndGraph(const double& RndFact) const {
  PNGraph Graph = TNGraph::New();
  for (int i = 0; i < GetDim(); i++) {
    Graph->AddNode(i); }
  for (int r = 0; r < GetDim(); r++) {
    for (int c = 0; c < GetDim(); c++) {
      if (RndFact * At(r, c) >= TKronMtx::Rnd.GetUniDev()) { Graph->AddEdge(r, c); }
    }
  }
  return Graph;
}

int TKronMtx::GetKronIter(const int& GNodes, const int& SeedMtxSz) {
  return (int) ceil(log(double(GNodes)) / log(double(SeedMtxSz)));
}

//
PNGraph TKronMtx::GenKronecker(const TKronMtx& SeedMtx, const int& NIter, const bool& IsDir, const int& Seed) {
  const TKronMtx& SeedGraph = SeedMtx;
  const int NNodes = SeedGraph.GetNodes(NIter);
  printf("  Kronecker: %d nodes, %s...\n", NNodes, IsDir ? "Directed":"UnDirected");
  PNGraph Graph = TNGraph::New(NNodes, -1);
  TExeTm ExeTm;
  TRnd Rnd(Seed);
  int edges = 0;
  for (int node1 = 0; node1 < NNodes; node1++) {
    Graph->AddNode(node1); }
  if (IsDir) {
    for (int node1 = 0; node1 < NNodes; node1++) {
      for (int node2 = 0; node2 < NNodes; node2++) {
        if (SeedGraph.IsEdgePlace(node1, node2, NIter, Rnd.GetUniDev())) {
          Graph->AddEdge(node1, node2);
          edges++;
        }
      }
      if (node1 % 1000 == 0) printf("\r...%dk, %dk", node1/1000, edges/1000);
    }
  } else {
    for (int node1 = 0; node1 < NNodes; node1++) {
      for (int node2 = node1; node2 < NNodes; node2++) {
        if (SeedGraph.IsEdgePlace(node1, node2, NIter, Rnd.GetUniDev())) {
          Graph->AddEdge(node1, node2);
          Graph->AddEdge(node2, node1);
          edges++;
        }
      }
      if (node1 % 1000 == 0) printf("\r...%dk, %dk", node1/1000, edges/1000);
    }
  }
  printf("\r             %d edges [%s]\n", Graph->GetEdges(), ExeTm.GetTmStr());
  return Graph;
}

//
PNGraph TKronMtx::GenFastKronecker(const TKronMtx& SeedMtx, const int& NIter, const bool& IsDir, const int& Seed) {
  const TKronMtx& SeedGraph = SeedMtx;
  const int MtxDim = SeedGraph.GetDim();
  const double MtxSum = SeedGraph.GetMtxSum();
  const int NNodes = SeedGraph.GetNodes(NIter);
  const int NEdges = SeedGraph.GetEdges(NIter);
  //
  //
  printf("  FastKronecker: %d nodes, %d edges, %s...\n", NNodes, NEdges, IsDir ? "Directed":"UnDirected");
  PNGraph Graph = TNGraph::New(NNodes, -1);
  TRnd Rnd(Seed);
  TExeTm ExeTm;
  //
  TVec<TFltIntIntTr> ProbToRCPosV; //
  double CumProb = 0.0;
  for (int r = 0; r < MtxDim; r++) {
    for (int c = 0; c < MtxDim; c++) {
      const double Prob = SeedGraph.At(r, c);
      if (Prob > 0.0) {
        CumProb += Prob;
        ProbToRCPosV.Add(TFltIntIntTr(CumProb/MtxSum, r, c));
      }
    }
  }
  //
  for (int i = 0; i < NNodes; i++) {
    Graph->AddNode(i); }
  //
  int Rng, Row, Col, Collision=0, n = 0;
  for (int edges = 0; edges < NEdges; ) {
    Rng=NNodes;  Row=0;  Col=0;
    for (int iter = 0; iter < NIter; iter++) {
      const double& Prob = Rnd.GetUniDev();
      n = 0; while(Prob > ProbToRCPosV[n].Val1) { n++; }
      const int MtxRow = ProbToRCPosV[n].Val2;
      const int MtxCol = ProbToRCPosV[n].Val3;
      Rng /= MtxDim;
      Row += MtxRow * Rng;
      Col += MtxCol * Rng;
    }
    if (! Graph->IsEdge(Row, Col)) { //
      Graph->AddEdge(Row, Col);  edges++;
      if (! IsDir) {
        if (Row != Col) Graph->AddEdge(Col, Row);
        edges++;
      }
    } else { Collision++; }
    //
  }
  //
  printf("             collisions: %d (%.4f)\n", Collision, Collision/(double)Graph->GetEdges());
  return Graph;
}

//
PNGraph TKronMtx::GenFastKronecker(const TKronMtx& SeedMtx, const int& NIter, const int& Edges, const bool& IsDir, const int& Seed) {
  const TKronMtx& SeedGraph = SeedMtx;
  const int MtxDim = SeedGraph.GetDim();
  const double MtxSum = SeedGraph.GetMtxSum();
  const int NNodes = SeedGraph.GetNodes(NIter);
  const int NEdges = Edges;
  //
  //
  printf("  RMat Kronecker: %d nodes, %d edges, %s...\n", NNodes, NEdges, IsDir ? "Directed":"UnDirected");
  PNGraph Graph = TNGraph::New(NNodes, -1);
  TRnd Rnd(Seed);
  TExeTm ExeTm;
  //
  TVec<TFltIntIntTr> ProbToRCPosV; //
  double CumProb = 0.0;
  for (int r = 0; r < MtxDim; r++) {
    for (int c = 0; c < MtxDim; c++) {
      const double Prob = SeedGraph.At(r, c);
      if (Prob > 0.0) {
        CumProb += Prob;
        ProbToRCPosV.Add(TFltIntIntTr(CumProb/MtxSum, r, c));
      }
    }
  }
  //
  for (int i = 0; i < NNodes; i++) {
    Graph->AddNode(i); }
  //
  int Rng, Row, Col, Collision=0, n = 0;
  for (int edges = 0; edges < NEdges; ) {
    Rng=NNodes;  Row=0;  Col=0;
    for (int iter = 0; iter < NIter; iter++) {
      const double& Prob = Rnd.GetUniDev();
      n = 0; while(Prob > ProbToRCPosV[n].Val1) { n++; }
      const int MtxRow = ProbToRCPosV[n].Val2;
      const int MtxCol = ProbToRCPosV[n].Val3;
      Rng /= MtxDim;
      Row += MtxRow * Rng;
      Col += MtxCol * Rng;
    }
    if (! Graph->IsEdge(Row, Col)) { //
      Graph->AddEdge(Row, Col);  edges++;
      if (! IsDir) {
        if (Row != Col) Graph->AddEdge(Col, Row);
        edges++;
      }
    } else { Collision++; }
    //
  }
  //
  printf("             collisions: %d (%.4f)\n", Collision, Collision/(double)Graph->GetEdges());
  return Graph;
}

PNGraph TKronMtx::GenDetKronecker(const TKronMtx& SeedMtx, const int& NIter, const bool& IsDir) {
  const TKronMtx& SeedGraph = SeedMtx;
  const int NNodes = SeedGraph.GetNodes(NIter);
  printf("  Deterministic Kronecker: %d nodes, %s...\n", NNodes, IsDir ? "Directed":"UnDirected");
  PNGraph Graph = TNGraph::New(NNodes, -1);
  TExeTm ExeTm;
  int edges = 0;
  for (int node1 = 0; node1 < NNodes; node1++) { Graph->AddNode(node1); }

  for (int node1 = 0; node1 < NNodes; node1++) {
    for (int node2 = 0; node2 < NNodes; node2++) {
      if (SeedGraph.IsEdgePlace(node1, node2, NIter, Rnd.GetUniDev())) {
        Graph->AddEdge(node1, node2);
        edges++;
      }
    }
    if (node1 % 1000 == 0) printf("\r...%dk, %dk", node1/1000, edges/1000);
  }
  return Graph;
}

void TKronMtx::PlotCmpGraphs(const TKronMtx& SeedMtx, const PNGraph& Graph, const TStr& FNmPref, const TStr& Desc) {
  const int KronIters = SeedMtx.GetKronIter(Graph->GetNodes());
  PNGraph KronG, WccG;
  const bool FastGen = true;
  if (FastGen) { KronG = TKronMtx::GenFastKronecker(SeedMtx, KronIters, true, 0); }
  else { KronG = TKronMtx::GenKronecker(SeedMtx, KronIters, true, 0); }
  TSnap::DelZeroDegNodes(KronG);
  WccG = TSnap::GetMxWcc(KronG);
  const TStr Desc1 = TStr::Fmt("%s", Desc.CStr());
  TGStatVec GS(tmuNodes, TFSet() | gsdInDeg | gsdOutDeg | gsdWcc | gsdHops | gsdScc | gsdClustCf | gsdSngVec | gsdSngVal);
  //
  //
  GS.Add(Graph, TSecTm(1), TStr::Fmt("GRAPH  G(%d, %d)", Graph->GetNodes(), Graph->GetEdges()));
  GS.Add(KronG, TSecTm(2), TStr::Fmt("KRONECKER  K(%d, %d)", KronG->GetNodes(), KronG->GetEdges()));
  GS.Add(WccG, TSecTm(3),  TStr::Fmt("KRONECKER  wccK(%d, %d)", WccG->GetNodes(), WccG->GetEdges()));
  const TStr Style = "linewidth 1 pointtype 6 pointsize 1";
  GS.ImposeDistr(gsdInDeg, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdInDeg, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdOutDeg, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdOutDeg, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdHops, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdClustCf, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdClustCf, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdSngVal, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdSngVal, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdSngVec, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdSngVec, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdWcc, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdWcc, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdScc, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdScc, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
//
//
}

void TKronMtx::PlotCmpGraphs(const TKronMtx& SeedMtx1, const TKronMtx& SeedMtx2, const PNGraph& Graph, const TStr& FNmPref, const TStr& Desc) {
  const int KronIters1 = SeedMtx1.GetKronIter(Graph->GetNodes());
  const int KronIters2 = SeedMtx2.GetKronIter(Graph->GetNodes());
  PNGraph KronG1, KronG2;
  const bool FastGen = true;
  if (FastGen) {
    KronG1 = TKronMtx::GenFastKronecker(SeedMtx1, KronIters1, true, 0);
    KronG2 = TKronMtx::GenFastKronecker(SeedMtx2, KronIters2, false, 0); } //
  else {
    KronG1 = TKronMtx::GenKronecker(SeedMtx1, KronIters1, true, 0);
    KronG2 = TKronMtx::GenKronecker(SeedMtx2, KronIters2, true, 0);  }
  TSnap::DelZeroDegNodes(KronG1);
  TSnap::DelZeroDegNodes(KronG2);
  const TStr Desc1 = TStr::Fmt("%s", Desc.CStr());
  TGStatVec GS(tmuNodes, TFSet() | gsdInDeg | gsdOutDeg | gsdWcc | gsdScc | gsdHops | gsdClustCf | gsdSngVec | gsdSngVal | gsdTriadPart);
  //
  //
  GS.Add(Graph, TSecTm(1), TStr::Fmt("GRAPH  G(%d, %d)", Graph->GetNodes(), Graph->GetEdges()));
  GS.Add(KronG1, TSecTm(2), TStr::Fmt("KRONECKER1  K(%d, %d) %s", KronG1->GetNodes(), KronG1->GetEdges(), SeedMtx1.GetMtxStr().CStr()));
  GS.Add(KronG2, TSecTm(3),  TStr::Fmt("KRONECKER2  K(%d, %d) %s", KronG2->GetNodes(), KronG2->GetEdges(), SeedMtx2.GetMtxStr().CStr()));
  const TStr Style = "linewidth 1 pointtype 6 pointsize 1";
  //
  GS.ImposeDistr(gsdInDeg, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdOutDeg, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdHops, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdClustCf, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdSngVal, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdSngVec, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdWcc, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdScc, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdTriadPart, FNmPref, Desc1, false, false, gpwLinesPoints, Style);
  //
  GS.ImposeDistr(gsdInDeg, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdOutDeg, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdClustCf, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdScc, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdWcc, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdSngVec, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdSngVal, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
  GS.ImposeDistr(gsdTriadPart, FNmPref+"-B", Desc1, true, false, gpwLinesPoints, Style);
}

void TKronMtx::PlotCmpGraphs(const TVec<TKronMtx>& SeedMtxV, const PNGraph& Graph, const TStr& FNmPref, const TStr& Desc) {
  const TStr Desc1 = TStr::Fmt("%s", Desc.CStr());
  TGStatVec GS(tmuNodes, TFSet() | gsdInDeg | gsdOutDeg | gsdWcc | gsdScc | gsdHops | gsdClustCf | gsdSngVec | gsdSngVal);
  GS.Add(Graph, TSecTm(1), TStr::Fmt("GRAPH  G(%d, %d)", Graph->GetNodes(), Graph->GetEdges()));
  //
  //
  for (int m = 0; m < SeedMtxV.Len(); m++) {
    const int KronIters = SeedMtxV[m].GetKronIter(Graph->GetNodes());
    PNGraph KronG1 = TKronMtx::GenFastKronecker(SeedMtxV[m], KronIters, true, 0);
    printf("*** K(%d, %d) n0=%d\n", KronG1->GetNodes(), KronG1->GetEdges(), SeedMtxV[m].GetDim());
    TSnap::DelZeroDegNodes(KronG1);
    printf(" del zero deg K(%d, %d) n0=%d\n", KronG1->GetNodes(), KronG1->GetEdges(), m);
    GS.Add(KronG1, TSecTm(m+2), TStr::Fmt("K(%d, %d) n0^k=%d n0=%d", KronG1->GetNodes(), KronG1->GetEdges(), SeedMtxV[m].GetNZeroK(Graph), SeedMtxV[m].GetDim()));
    //
    const TStr Style = "linewidth 1 pointtype 6 pointsize 1";
    GS.ImposeDistr(gsdInDeg, FNmPref, Desc1, false, false, gpwLines, Style);
    GS.ImposeDistr(gsdInDeg, FNmPref+"-B", Desc1, true, false, gpwLines, Style);
    GS.ImposeDistr(gsdOutDeg, FNmPref, Desc1, false, false, gpwLines, Style);
    GS.ImposeDistr(gsdOutDeg, FNmPref+"-B", Desc1, true, false, gpwLines, Style);
    GS.ImposeDistr(gsdHops, FNmPref, Desc1, false, false, gpwLines, Style);
    GS.ImposeDistr(gsdClustCf, FNmPref, Desc1, false, false, gpwLines, Style);
    GS.ImposeDistr(gsdClustCf, FNmPref+"-B", Desc1, true, false, gpwLines, Style);
    GS.ImposeDistr(gsdSngVal, FNmPref, Desc1, false, false, gpwLines, Style);
    GS.ImposeDistr(gsdSngVal, FNmPref+"-B", Desc1, true, false, gpwLines, Style);
    GS.ImposeDistr(gsdSngVec, FNmPref, Desc1, false, false, gpwLines, Style);
    GS.ImposeDistr(gsdSngVec, FNmPref+"-B", Desc1, true, false, gpwLines, Style);
    GS.ImposeDistr(gsdWcc, FNmPref, Desc1, false, false, gpwLines, Style);
    GS.ImposeDistr(gsdWcc, FNmPref+"-B", Desc1, true, false, gpwLines, Style);
    GS.ImposeDistr(gsdScc, FNmPref, Desc1, false, false, gpwLines, Style);
    GS.ImposeDistr(gsdScc, FNmPref+"-B", Desc1, true, false, gpwLines, Style);
  }
  //
  //
}

void TKronMtx::KronMul(const TKronMtx& Left, const TKronMtx& Right, TKronMtx& Result) {
  const int LDim = Left.GetDim();
  const int RDim = Right.GetDim();
  Result.GenMtx(LDim * RDim);
  for (int r1 = 0; r1 < LDim; r1++) {
    for (int c1 = 0; c1 < LDim; c1++) {
      const double& Val = Left.At(r1, c1);
      for (int r2 = 0; r2 < RDim; r2++) {
        for (int c2 = 0; c2 < RDim; c2++) {
          Result.At(r1*RDim+r2, c1*RDim+c2) = Val * Right.At(r2, c2);
        }
      }
    }
  }
}

void TKronMtx::KronSum(const TKronMtx& Left, const TKronMtx& Right, TKronMtx& Result) {
  const int LDim = Left.GetDim();
  const int RDim = Right.GetDim();
  Result.GenMtx(LDim * RDim);
  for (int r1 = 0; r1 < LDim; r1++) {
    for (int c1 = 0; c1 < LDim; c1++) {
      const double& Val = Left.At(r1, c1);
      for (int r2 = 0; r2 < RDim; r2++) {
        for (int c2 = 0; c2 < RDim; c2++) {
          if (Val == NInf || Right.At(r2, c2) == NInf) {
            Result.At(r1*RDim+r2, c1*RDim+c2) = NInf; }
          else {
            Result.At(r1*RDim+r2, c1*RDim+c2) = Val + Right.At(r2, c2); }
        }
      }
    }
  }
}

void TKronMtx::KronPwr(const TKronMtx& KronMtx, const int& NIter, TKronMtx& OutMtx) {
  OutMtx = KronMtx;
  TKronMtx NewOutMtx;
  for (int iter = 0; iter < NIter; iter++) {
    KronMul(OutMtx, KronMtx, NewOutMtx);
    NewOutMtx.Swap(OutMtx);
  }

}

void TKronMtx::Dump(const TStr& MtxNm, const bool& Sort) const {
  /*printf("%s: %d x %d\n", MtxNm.Empty()?"Mtx":MtxNm.CStr(), GetDim(), GetDim());
  for (int r = 0; r < GetDim(); r++) {
    for (int c = 0; c < GetDim(); c++) { printf("  %8.2g", At(r, c)); }
    printf("\n");
  }*/
  if (! MtxNm.Empty()) printf("%s\n", MtxNm.CStr());
  double Sum=0.0;
  TFltV ValV = SeedMtx;
  if (Sort) { ValV.Sort(false); }
  for (int i = 0; i < ValV.Len(); i++) {
    printf("  %10.4g", ValV[i]());
    Sum += ValV[i];
    if ((i+1) % GetDim() == 0) { printf("\n"); }
  }
  printf(" (sum:%.4f)\n", Sum);
}

//
double TKronMtx::GetAvgAbsErr(const TKronMtx& Kron1, const TKronMtx& Kron2) {
  TFltV P1 = Kron1.GetMtx();
  TFltV P2 = Kron2.GetMtx();
  IAssert(P1.Len() == P2.Len());
  P1.Sort();  P2.Sort();
  double delta = 0.0;
  for (int i = 0; i < P1.Len(); i++) {
    delta += fabs(P1[i] - P2[i]);
  }
  return delta/P1.Len();
}

//
double TKronMtx::GetAvgFroErr(const TKronMtx& Kron1, const TKronMtx& Kron2) {
  TFltV P1 = Kron1.GetMtx();
  TFltV P2 = Kron2.GetMtx();
  IAssert(P1.Len() == P2.Len());
  P1.Sort();  P2.Sort();
  double delta = 0.0;
  for (int i = 0; i < P1.Len(); i++) {
    delta += pow(P1[i] - P2[i], 2);
  }
  return sqrt(delta/P1.Len());
}

//
TKronMtx TKronMtx::GetMtx(TStr MatlabMtxStr) {
  TStrV RowStrV, ColStrV;
  MatlabMtxStr.ChangeChAll(',', ' ');
  MatlabMtxStr.SplitOnAllCh(';', RowStrV);  IAssert(! RowStrV.Empty());
  RowStrV[0].SplitOnWs(ColStrV);    IAssert(! ColStrV.Empty());
  const int Rows = RowStrV.Len();
  const int Cols = ColStrV.Len();
  IAssert(Rows == Cols);
  TKronMtx Mtx(Rows);
  for (int r = 0; r < Rows; r++) {
    RowStrV[r].SplitOnWs(ColStrV);
    IAssert(ColStrV.Len() == Cols);
    for (int c = 0; c < Cols; c++) {
      Mtx.At(r, c) = (double) ColStrV[c].GetFlt(); }
  }
  return Mtx;
}

TKronMtx TKronMtx::GetRndMtx(const int& Dim, const double& MinProb) {
  TKronMtx Mtx;
  Mtx.SetRndMtx(Dim, MinProb);
  return Mtx;
}

TKronMtx TKronMtx::GetInitMtx(const int& Dim, const int& Nodes, const int& Edges) {
  const double MxParam = 0.8+TKronMtx::Rnd.GetUniDev()/5.0;
  const double MnParam = 0.2-TKronMtx::Rnd.GetUniDev()/5.0;
  const double Step = (MxParam-MnParam) / (Dim*Dim-1);
  TFltV ParamV(Dim*Dim);
  if (Dim == 1) { ParamV.PutAll(0.5); } //
  else {
    for (int p = 0; p < ParamV.Len(); p++) {
      ParamV[p] = MxParam - p*Step; }
  }
  //
  TKronMtx Mtx(ParamV);
  Mtx.SetForEdges(Nodes, Edges);
  return Mtx;
}

TKronMtx TKronMtx::GetInitMtx(const TStr& MtxStr, const int& Dim, const int& Nodes, const int& Edges) {
  TKronMtx Mtx(Dim);
  if (TCh::IsNum(MtxStr[0])) { Mtx = TKronMtx::GetMtx(MtxStr); }
  else if (MtxStr[0] == 'r') { Mtx = TKronMtx::GetRndMtx(Dim, 0.1); }
  else if (MtxStr[0] == 'a') {
    const double Prob = TKronMtx::Rnd.GetUniDev();
    if (Prob < 0.4) {
      Mtx = TKronMtx::GetInitMtx(Dim, Nodes, Edges); }
    else { //
      const double Max = 0.9+TKronMtx::Rnd.GetUniDev()/10.0;
      const double Min = 0.1-TKronMtx::Rnd.GetUniDev()/10.0;
      const double Med = (Max-Min)/2.0;
      Mtx.At(0,0)      = Max;       Mtx.At(0,Dim-1) = Med;
      Mtx.At(Dim-1, 0) = Med;  Mtx.At(Dim-1, Dim-1) = Min;
      for (int i = 1; i < Dim-1; i++) {
        Mtx.At(i,i) = Max - double(i)*(Max-Min)/double(Dim-1);
        Mtx.At(i, 0) = Mtx.At(0, i) = Max - double(i)*(Max-Med)/double(Dim-1);
        Mtx.At(i, Dim-1) = Mtx.At(Dim-1, i) = Med - double(i)*(Med-Min)/double(Dim-1);
      }
      for (int i = 1; i < Dim-1; i++) {
        for (int j = 1; j < Dim-1; j++) {
          if (i >= j) { continue; }
          Mtx.At(i,j) = Mtx.At(j,i) = Mtx.At(i,i) - (j-i)*(Mtx.At(i,i)-Mtx.At(i,Dim-1))/(Dim-i-1);
        }
      }
      Mtx.AddRndNoise(0.1);
    }
  } else { FailR("Wrong mtx: matlab str, or random (r), or all (a)"); }
  Mtx.SetForEdges(Nodes, Edges);
  return Mtx;
}

TKronMtx TKronMtx::GetMtxFromNm(const TStr& MtxNm) {
  if (MtxNm == "3chain") return TKronMtx::GetMtx("1 1 0; 1 1 1; 0 1 1");
  else if (MtxNm == "4star") return TKronMtx::GetMtx("1 1 1 1; 1 1 0 0 ; 1 0 1 0; 1 0 0 1");
  else if (MtxNm == "4chain") return TKronMtx::GetMtx("1 1 0 0; 1 1 1 0 ; 0 1 1 1; 0 0 1 1");
  else if (MtxNm == "4square") return TKronMtx::GetMtx("1 1 0 1; 1 1 1 0 ; 0 1 1 1; 1 0 1 1");
  else if (MtxNm == "5star") return TKronMtx::GetMtx("1 1 1 1 1; 1 1 0 0 0; 1 0 1 0 0; 1 0 0 1 0; 1 0 0 0 1");
  else if (MtxNm == "6star") return TKronMtx::GetMtx("1 1 1 1 1 1; 1 1 0 0 0 0; 1 0 1 0 0 0; 1 0 0 1 0 0; 1 0 0 0 1 0; 1 0 0 0 0 1");
  else if (MtxNm == "7star") return TKronMtx::GetMtx("1 1 1 1 1 1 1; 1 1 0 0 0 0 0; 1 0 1 0 0 0 0; 1 0 0 1 0 0 0; 1 0 0 0 1 0 0; 1 0 0 0 0 1 0; 1 0 0 0 0 0 1");
  else if (MtxNm == "5burst") return TKronMtx::GetMtx("1 1 1 1 0; 1 1 0 0 0; 1 0 1 0 0; 1 0 0 1 1; 0 0 0 1 1");
  else if (MtxNm == "7burst") return TKronMtx::GetMtx("1 0 0 1 0 0 0; 0 1 0 1 0 0 0; 0 0 1 1 0 0 0; 1 1 1 1 1 0 0; 0 0 0 1 1 1 1; 0 0 0 0 1 1 0; 0 0 0 0 1 0 1");
  else if (MtxNm == "7cross") return TKronMtx::GetMtx("1 0 0 1 0 0 0; 0 1 0 1 0 0 0; 0 0 1 1 0 0 0; 1 1 1 1 1 0 0; 0 0 0 1 1 1 0; 0 0 0 0 1 1 1; 0 0 0 0 0 1 1");
  FailR(TStr::Fmt("Unknow matrix: '%s'", MtxNm.CStr()).CStr());
  return TKronMtx();
}

TKronMtx TKronMtx::LoadTxt(const TStr& MtxFNm) {
  PSs Ss = TSs::LoadTxt(ssfTabSep, MtxFNm);
  IAssertR(Ss->GetXLen() == Ss->GetYLen(), "Not a square matrix");
  IAssert(Ss->GetYLen() == Ss->GetXLen());
  TKronMtx Mtx(Ss->GetYLen());
  for (int r = 0; r < Ss->GetYLen(); r++) {
    for (int c = 0; c < Ss->GetXLen(); c++) {
      Mtx.At(r, c) = (double) Ss->At(c, r).GetFlt(); }
  }
  return Mtx;
}


//
//
TKroneckerLL::TKroneckerLL(const PNGraph& GraphPt, const TFltV& ParamV, const double& PermPSwapNd): PermSwapNodeProb(PermPSwapNd) {
  InitLL(GraphPt, TKronMtx(ParamV));
}

TKroneckerLL::TKroneckerLL(const PNGraph& GraphPt, const TKronMtx& ParamMtx, const double& PermPSwapNd) : PermSwapNodeProb(PermPSwapNd) {
  InitLL(GraphPt, ParamMtx);
}

TKroneckerLL::TKroneckerLL(const PNGraph& GraphPt, const TKronMtx& ParamMtx, const TIntV& NodeIdPermV, const double& PermPSwapNd) : PermSwapNodeProb(PermPSwapNd) {
  InitLL(GraphPt, ParamMtx);
  NodePerm = NodeIdPermV;
  SetIPerm(NodePerm);
}

PKroneckerLL TKroneckerLL::New(const PNGraph& GraphPt, const TKronMtx& ParamMtx, const double& PermPSwapNd) {
  return new TKroneckerLL(GraphPt, ParamMtx, PermPSwapNd);
}

PKroneckerLL TKroneckerLL::New(const PNGraph& GraphPt, const TKronMtx& ParamMtx, const TIntV& NodeIdPermV, const double& PermPSwapNd) {
  return new TKroneckerLL(GraphPt, ParamMtx, NodeIdPermV, PermPSwapNd);
}

void TKroneckerLL::SetPerm(const char& PermId) {
  if (PermId == 'o') { SetOrderPerm(); }
  else if (PermId == 'd') { SetDegPerm(); }
  else if (PermId == 'r') { SetRndPerm(); }
  else if (PermId == 'b') { SetBestDegPerm(); }
  else FailR("Unknown permutation type (o,d,r)");
}

void TKroneckerLL::SetOrderPerm() {
  NodePerm.Gen(Nodes, 0);
  for (int i = 0; i < Graph->GetNodes(); i++) {
    NodePerm.Add(i); }
  SetIPerm(NodePerm);
}

void TKroneckerLL::SetRndPerm() {
  NodePerm.Gen(Nodes, 0);
  for (int i = 0; i < Graph->GetNodes(); i++) {
    NodePerm.Add(i); }
  NodePerm.Shuffle(TKronMtx::Rnd);
  SetIPerm(NodePerm);
}

void TKroneckerLL::SetDegPerm() {
  TIntPrV DegNIdV;
  for (TNGraph::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    DegNIdV.Add(TIntPr(NI.GetDeg(), NI.GetId()));
  }
  DegNIdV.Sort(false);
  NodePerm.Gen(DegNIdV.Len(), 0);
  for (int i = 0; i < DegNIdV.Len(); i++) {
    NodePerm.Add(DegNIdV[i].Val2);
  }
  SetIPerm(NodePerm);
}

//
void TKroneckerLL::SetBestDegPerm() {
  NodePerm.Gen(Nodes);
  const int NZero = ProbMtx.GetDim();
  TFltIntPrV DegV(Nodes), CDegV(Nodes);
  TFltV Row(NZero);
  TFltV Col(NZero);
  for(int i = 0; i < NZero; i++) {
	  for(int j = 0; j < NZero; j++) {
		  Row[i] += ProbMtx.At(i, j);
		  Col[i] += ProbMtx.At(j, i);
	  }
  }

  for(int i = 0; i < Nodes; i++) {
	  TNGraph::TNodeI NodeI = Graph->GetNI(i);
	  int NId = i;
	  double RowP = 1.0, ColP = 1.0;
	  for(int j = 0; j < KronIters; j++) {
		  int Bit = NId % NZero;
		  RowP *= Row[Bit];		ColP *= Col[Bit];
		  NId /= NZero;
	  }
	  CDegV[i] = TFltIntPr(RowP + ColP, i);
	  DegV[i] = TFltIntPr(NodeI.GetDeg(), i);
  }
  DegV.Sort(false);		CDegV.Sort(false);
  for(int i = 0; i < Nodes; i++) {
	  NodePerm[DegV[i].Val2] = CDegV[i].Val2;
  }
  SetIPerm(NodePerm);
}

//
void TKroneckerLL::SetIPerm(const TIntV& Perm) {
	InvertPerm.Gen(Perm.Len());
	for (int i = 0; i < Perm.Len(); i++) {
		InvertPerm[Perm[i]] = i;
	}
}

void TKroneckerLL::SetGraph(const PNGraph& GraphPt) {
  Graph = GraphPt;
  bool NodesOk = true;
  //
  for (int nid = 0; nid < Graph->GetNodes(); nid++) {
    if (! Graph->IsNode(nid)) { NodesOk=false; break; } }
  if (! NodesOk) {
    TIntV NIdV;  GraphPt->GetNIdV(NIdV);
    Graph = TSnap::GetSubGraph(GraphPt, NIdV, true);
    for (int nid = 0; nid < Graph->GetNodes(); nid++) {
      IAssert(Graph->IsNode(nid)); }
  }
  Nodes = Graph->GetNodes();
  IAssert(LLMtx.GetDim() > 1 && LLMtx.Len() == ProbMtx.Len());
  KronIters = (int) ceil(log(double(Nodes)) / log(double(ProbMtx.GetDim())));
  //
//
    GEdgeV.Gen(Graph->GetEdges(), 0);
    for (TNGraph::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
      if (EI.GetSrcNId() != EI.GetDstNId()) {
        GEdgeV.Add(TIntTr(EI.GetSrcNId(), EI.GetDstNId(), -1));
      }
    }
//

  RealNodes = Nodes;
  RealEdges = Graph->GetEdges();
  LEdgeV = TIntTrV();
  LSelfEdge = 0;
}


void TKroneckerLL::AppendIsoNodes() {
  Nodes = (int) pow((double)ProbMtx.GetDim(), KronIters);
  //
  for (int nid = Graph->GetNodes(); nid < Nodes; nid++) {
	  Graph->AddNode(nid);
  }
}

//
void TKroneckerLL::RestoreGraph(const bool RestoreNodes) {
	//
	int NId1, NId2;
	for (int e = 0; e < LEdgeV.Len(); e++) {
    	NId1 = LEdgeV[e].Val1;  NId2 = LEdgeV[e].Val2;
		Graph->DelEdge(NId1, NId2);
//
	}
	if(LEdgeV.Len() - LSelfEdge)
		GEdgeV.Del(GEdgeV.Len() - LEdgeV.Len() + LSelfEdge, GEdgeV.Len() - 1);
	LEdgeV.Clr();
	LSelfEdge = 0;

	if(RestoreNodes) {
		for(int i = Graph->GetNodes()-1; i >= RealNodes; i--) {
			Graph->DelNode(i);
		}
	}
}

double TKroneckerLL::GetFullGraphLL() const {
  //
  //
  //
  double ElemCnt = 1;
  const double dim = LLMtx.GetDim();
  //
  for (int i = 1; i < KronIters; i++) {
    ElemCnt = dim*dim*ElemCnt + TMath::Power(dim, 2*i);
  }
  return ElemCnt * LLMtx.GetMtxSum();
}

double TKroneckerLL::GetFullRowLL(int RowId) const {
  double RowLL = 0.0;
  const int MtxDim = LLMtx.GetDim();
  for (int level = 0; level < KronIters; level++) {
    RowLL += LLMtx.GetRowSum(RowId % MtxDim);
    RowId /= MtxDim;
  }
  return RowLL;
}

double TKroneckerLL::GetFullColLL(int ColId) const {
  double ColLL = 0.0;
  const int MtxDim = LLMtx.GetDim();
  for (int level = 0; level < KronIters; level++) {
    ColLL += LLMtx.GetColSum(ColId % MtxDim);
    ColId /= MtxDim;
  }
  return ColLL;
}

double TKroneckerLL::GetEmptyGraphLL() const {
  double LL = 0;
  for (int NId1 = 0; NId1 < LLMtx.GetNodes(KronIters); NId1++) {
    for (int NId2 = 0; NId2 < LLMtx.GetNodes(KronIters); NId2++) {
      LL = LL + LLMtx.GetNoEdgeLL(NId1, NId2, KronIters);
    }
  }
  return LL;
}

//
double TKroneckerLL::GetApxEmptyGraphLL() const {
  double Sum=0.0, SumSq=0.0;
  for (int i = 0; i < ProbMtx.Len(); i++) {
    Sum += ProbMtx.At(i);
    SumSq += TMath::Sqr(ProbMtx.At(i));
  }
  return -pow(Sum, KronIters) - 0.5*pow(SumSq, KronIters);
}

void TKroneckerLL::InitLL(const TFltV& ParamV) {
  InitLL(TKronMtx(ParamV));
}

void TKroneckerLL::InitLL(const TKronMtx& ParamMtx) {
  IAssert(ParamMtx.IsProbMtx());
  ProbMtx = ParamMtx;
  ProbMtx.GetLLMtx(LLMtx);
  LogLike = TKronMtx::NInf;
  if (GradV.Len() != ProbMtx.Len()) {
    GradV.Gen(ProbMtx.Len()); }
  GradV.PutAll(0.0);
}

void TKroneckerLL::InitLL(const PNGraph& GraphPt, const TKronMtx& ParamMtx) {
  IAssert(ParamMtx.IsProbMtx());
  ProbMtx = ParamMtx;
  ProbMtx.GetLLMtx(LLMtx);
  SetGraph(GraphPt);
  LogLike = TKronMtx::NInf;
  if (GradV.Len() != ProbMtx.Len()) {
    GradV.Gen(ProbMtx.Len()); }
  GradV.PutAll(0.0);
}

//
double TKroneckerLL::CalcGraphLL() {
  LogLike = GetEmptyGraphLL(); //
  for (int nid = 0; nid < Nodes; nid++) {
    const TNGraph::TNodeI Node = Graph->GetNI(nid);
    const int SrcNId = NodePerm[nid];
    for (int e = 0; e < Node.GetOutDeg(); e++) {
      const int DstNId = NodePerm[Node.GetOutNId(e)];
      LogLike = LogLike - LLMtx.GetNoEdgeLL(SrcNId, DstNId, KronIters)
        + LLMtx.GetEdgeLL(SrcNId, DstNId, KronIters);
    }
  }
  return LogLike;
}

//
double TKroneckerLL::CalcApxGraphLL() {
  LogLike = GetApxEmptyGraphLL(); //
  for (int nid = 0; nid < Nodes; nid++) {
    const TNGraph::TNodeI Node = Graph->GetNI(nid);
    const int SrcNId = NodePerm[nid];
    for (int e = 0; e < Node.GetOutDeg(); e++) {
      const int DstNId = NodePerm[Node.GetOutNId(e)];
      LogLike = LogLike - LLMtx.GetApxNoEdgeLL(SrcNId, DstNId, KronIters)
        + LLMtx.GetEdgeLL(SrcNId, DstNId, KronIters);
    }
  }
  return LogLike;
}

//
//
//
//
double TKroneckerLL::NodeLLDelta(const int& NId) const {
  if (! Graph->IsNode(NId)) { return 0.0; } //
  double Delta = 0.0;
  const TNGraph::TNodeI Node = Graph->GetNI(NId);
  //
  const int SrcRow = NodePerm[NId];
  for (int e = 0; e < Node.GetOutDeg(); e++) {
    const int DstCol = NodePerm[Node.GetOutNId(e)];
    Delta += - LLMtx.GetApxNoEdgeLL(SrcRow, DstCol, KronIters)
      + LLMtx.GetEdgeLL(SrcRow, DstCol, KronIters);
  }
  //
  const int SrcCol = NodePerm[NId];
  for (int e = 0; e < Node.GetInDeg(); e++) {
    const int DstRow = NodePerm[Node.GetInNId(e)];
    Delta += - LLMtx.GetApxNoEdgeLL(DstRow, SrcCol, KronIters)
      + LLMtx.GetEdgeLL(DstRow, SrcCol, KronIters);
  }
  //
  if (Graph->IsEdge(NId, NId)) {
    Delta += + LLMtx.GetApxNoEdgeLL(SrcRow, SrcCol, KronIters)
      - LLMtx.GetEdgeLL(SrcRow, SrcCol, KronIters);
    IAssert(SrcRow == SrcCol);
  }
  return Delta;
}

//
double TKroneckerLL::SwapNodesLL(const int& NId1, const int& NId2) {
  //
  LogLike = LogLike - NodeLLDelta(NId1) - NodeLLDelta(NId2);
  const int PrevId1 = NodePerm[NId1], PrevId2 = NodePerm[NId2];
  //
  if (Graph->IsEdge(NId1, NId2)) {
    LogLike += - LLMtx.GetApxNoEdgeLL(PrevId1, PrevId2, KronIters)
      + LLMtx.GetEdgeLL(PrevId1, PrevId2, KronIters); }
  if (Graph->IsEdge(NId2, NId1)) {
    LogLike += - LLMtx.GetApxNoEdgeLL(PrevId2, PrevId1, KronIters)
      + LLMtx.GetEdgeLL(PrevId2, PrevId1, KronIters); }
  //
  NodePerm.Swap(NId1, NId2);
  InvertPerm.Swap(NodePerm[NId1], NodePerm[NId2]);
  //
  LogLike = LogLike + NodeLLDelta(NId1) + NodeLLDelta(NId2);
  const int NewId1 = NodePerm[NId1], NewId2 = NodePerm[NId2];
  //
  if (Graph->IsEdge(NId1, NId2)) {
    LogLike += + LLMtx.GetApxNoEdgeLL(NewId1, NewId2, KronIters)
      - LLMtx.GetEdgeLL(NewId1, NewId2, KronIters); }
  if (Graph->IsEdge(NId2, NId1)) {
    LogLike += + LLMtx.GetApxNoEdgeLL(NewId2, NewId1, KronIters)
      - LLMtx.GetEdgeLL(NewId2, NewId1, KronIters); }
  return LogLike;
}

//
bool TKroneckerLL::SampleNextPerm(int& NId1, int& NId2) {
  //
  if (TKronMtx::Rnd.GetUniDev() < PermSwapNodeProb) {
    NId1 = TKronMtx::Rnd.GetUniDevInt(Nodes);
    NId2 = TKronMtx::Rnd.GetUniDevInt(Nodes);
    while (NId2 == NId1) { NId2 = TKronMtx::Rnd.GetUniDevInt(Nodes); }
  } else {
    //
    const int e = TKronMtx::Rnd.GetUniDevInt(GEdgeV.Len());
    NId1 = GEdgeV[e].Val1;  NId2 = GEdgeV[e].Val2;
  }
  const double U = TKronMtx::Rnd.GetUniDev();
  const double OldLL = LogLike;
  const double NewLL = SwapNodesLL(NId1, NId2);
  const double LogU = log(U);
  if (LogU > NewLL - OldLL) { //
    LogLike = OldLL;
    NodePerm.Swap(NId2, NId1); //
	InvertPerm.Swap(NodePerm[NId2], NodePerm[NId1]); //
    return false;
  }
  return true; //
}

//
double TKroneckerLL::GetEmptyGraphDLL(const int& ParamId) const {
  double DLL = 0.0;
  for (int NId1 = 0; NId1 < Nodes; NId1++) {
    for (int NId2 = 0; NId2 < Nodes; NId2++) {
      DLL += LLMtx.GetNoEdgeDLL(ParamId, NodePerm[NId1], NodePerm[NId2], KronIters);
    }
  }
  return DLL;
}

//
double TKroneckerLL::GetApxEmptyGraphDLL(const int& ParamId) const {
  double Sum=0.0, SumSq=0.0;
  for (int i = 0; i < ProbMtx.Len(); i++) {
    Sum += ProbMtx.At(i);
    SumSq += TMath::Sqr(ProbMtx.At(i));
  }
  //
  return -KronIters*pow(Sum, KronIters-1) - KronIters*pow(SumSq, KronIters-1)*ProbMtx.At(ParamId);
}

//
const TFltV& TKroneckerLL::CalcGraphDLL() {
  for (int ParamId = 0; ParamId < LLMtx.Len(); ParamId++) {
    double DLL = 0.0;
    for (int NId1 = 0; NId1 < Nodes; NId1++) {
      for (int NId2 = 0; NId2 < Nodes; NId2++) {
        if (Graph->IsEdge(NId1, NId2)) {
          DLL += LLMtx.GetEdgeDLL(ParamId, NodePerm[NId1], NodePerm[NId2], KronIters);
        } else {
          DLL += LLMtx.GetNoEdgeDLL(ParamId, NodePerm[NId1], NodePerm[NId2], KronIters);
        }
      }
    }
    GradV[ParamId] = DLL;
  }
  return GradV;
}

//
const TFltV& TKroneckerLL::CalcFullApxGraphDLL() {
  for (int ParamId = 0; ParamId < LLMtx.Len(); ParamId++) {
    double DLL = 0.0;
    for (int NId1 = 0; NId1 < Nodes; NId1++) {
      for (int NId2 = 0; NId2 < Nodes; NId2++) {
        if (Graph->IsEdge(NId1, NId2)) {
          DLL += LLMtx.GetEdgeDLL(ParamId, NodePerm[NId1], NodePerm[NId2], KronIters);
        } else {
          DLL += LLMtx.GetApxNoEdgeDLL(ParamId, NodePerm[NId1], NodePerm[NId2], KronIters);
        }
      }
    }
    GradV[ParamId] = DLL;
  }
  return GradV;
}

//
const TFltV& TKroneckerLL::CalcApxGraphDLL() {
  for (int ParamId = 0; ParamId < LLMtx.Len(); ParamId++) {
    double DLL = GetApxEmptyGraphDLL(ParamId);
    for (int nid = 0; nid < Nodes; nid++) {
      const TNGraph::TNodeI Node = Graph->GetNI(nid);
      const int SrcNId = NodePerm[nid];
      for (int e = 0; e < Node.GetOutDeg(); e++) {
        const int DstNId = NodePerm[Node.GetOutNId(e)];
        DLL = DLL - LLMtx.GetApxNoEdgeDLL(ParamId, SrcNId, DstNId, KronIters)
          + LLMtx.GetEdgeDLL(ParamId, SrcNId, DstNId, KronIters);
      }
    }
    GradV[ParamId] = DLL;
  }
  return GradV;
}

//
//
//
double TKroneckerLL::NodeDLLDelta(const int ParamId, const int& NId) const {
  if (! Graph->IsNode(NId)) { return 0.0; } //
  double Delta = 0.0;
  const TNGraph::TNodeI Node = Graph->GetNI(NId);
  const int SrcRow = NodePerm[NId];
  for (int e = 0; e < Node.GetOutDeg(); e++) {
    const int DstCol = NodePerm[Node.GetOutNId(e)];
    Delta += - LLMtx.GetApxNoEdgeDLL(ParamId, SrcRow, DstCol, KronIters)
      + LLMtx.GetEdgeDLL(ParamId, SrcRow, DstCol, KronIters);
  }
  const int SrcCol = NodePerm[NId];
  for (int e = 0; e < Node.GetInDeg(); e++) {
    const int DstRow = NodePerm[Node.GetInNId(e)];
    Delta += - LLMtx.GetApxNoEdgeDLL(ParamId, DstRow, SrcCol, KronIters)
      + LLMtx.GetEdgeDLL(ParamId, DstRow, SrcCol, KronIters);
  }
  //
  if (Graph->IsEdge(NId, NId)) {
    Delta += + LLMtx.GetApxNoEdgeDLL(ParamId, SrcRow, SrcCol, KronIters)
      - LLMtx.GetEdgeDLL(ParamId, SrcRow, SrcCol, KronIters);
    IAssert(SrcRow == SrcCol);
  }
  return Delta;
}

//
//
void TKroneckerLL::UpdateGraphDLL(const int& SwapNId1, const int& SwapNId2) {
  for (int ParamId = 0; ParamId < LLMtx.Len(); ParamId++) {
    //
    NodePerm.Swap(SwapNId1, SwapNId2);
    //
    TFlt& DLL = GradV[ParamId];
    DLL = DLL - NodeDLLDelta(ParamId, SwapNId1) - NodeDLLDelta(ParamId, SwapNId2);
    //
    const int PrevId1 = NodePerm[SwapNId1], PrevId2 = NodePerm[SwapNId2];
    if (Graph->IsEdge(SwapNId1, SwapNId2)) {
      DLL += - LLMtx.GetApxNoEdgeDLL(ParamId, PrevId1, PrevId2, KronIters)
        + LLMtx.GetEdgeDLL(ParamId, PrevId1, PrevId2, KronIters); }
    if (Graph->IsEdge(SwapNId2, SwapNId1)) {
      DLL += - LLMtx.GetApxNoEdgeDLL(ParamId, PrevId2, PrevId1, KronIters)
        + LLMtx.GetEdgeDLL(ParamId, PrevId2, PrevId1, KronIters); }
    //
    NodePerm.Swap(SwapNId1, SwapNId2);
    //
    DLL = DLL + NodeDLLDelta(ParamId, SwapNId1) + NodeDLLDelta(ParamId, SwapNId2);
    const int NewId1 = NodePerm[SwapNId1], NewId2 = NodePerm[SwapNId2];
    //
    if (Graph->IsEdge(SwapNId1, SwapNId2)) {
      DLL += + LLMtx.GetApxNoEdgeDLL(ParamId, NewId1, NewId2, KronIters)
        - LLMtx.GetEdgeDLL(ParamId, NewId1, NewId2, KronIters); }
    if (Graph->IsEdge(SwapNId2, SwapNId1)) {
      DLL += + LLMtx.GetApxNoEdgeDLL(ParamId, NewId2, NewId1, KronIters)
        - LLMtx.GetEdgeDLL(ParamId, NewId2, NewId1, KronIters); }
  }
}

void TKroneckerLL::SampleGradient(const int& WarmUp, const int& NSamples, double& AvgLL, TFltV& AvgGradV) {
  printf("SampleGradient: %s (%s warm-up):", TInt::GetMegaStr(NSamples).CStr(), TInt::GetMegaStr(WarmUp).CStr());
  int NId1=0, NId2=0, NAccept=0;
  TExeTm ExeTm1;
  if (WarmUp > 0) {
    CalcApxGraphLL();
    for (int s = 0; s < WarmUp; s++) { SampleNextPerm(NId1, NId2); }
    printf("  warm-up:%s,", ExeTm1.GetTmStr());  ExeTm1.Tick();
  }
  CalcApxGraphLL(); //
  CalcApxGraphDLL();
  AvgLL = 0;
  AvgGradV.Gen(LLMtx.Len());  AvgGradV.PutAll(0.0);
  printf("  sampl");
  for (int s = 0; s < NSamples; s++) {
    if (SampleNextPerm(NId1, NId2)) { //
      UpdateGraphDLL(NId1, NId2);  NAccept++; }
    for (int m = 0; m < LLMtx.Len(); m++) { AvgGradV[m] += GradV[m]; }
    AvgLL += GetLL();
  }
  printf("ing");
  AvgLL = AvgLL / double(NSamples);
  for (int m = 0; m < LLMtx.Len(); m++) {
    AvgGradV[m] = AvgGradV[m] / double(NSamples); }
  printf(":%s (%.0f/s), accept %.1f%%\n", ExeTm1.GetTmStr(), double(NSamples)/ExeTm1.GetSecs(),
    double(100*NAccept)/double(NSamples));
}

double TKroneckerLL::GradDescent(const int& NIter, const double& LrnRate, double MnStep, double MxStep, const int& WarmUp, const int& NSamples) {
  printf("\n----------------------------------------------------------------------\n");
  printf("Fitting graph on %d nodes, %d edges\n", Graph->GetNodes(), Graph->GetEdges());
  printf("Kron iters:  %d (== %d nodes)\n\n", KronIters(), ProbMtx.GetNodes(KronIters()));
  TExeTm IterTm, TotalTm;
  double OldLL=-1e10, CurLL=0;
  const double EZero = pow((double) Graph->GetEdges(), 1.0/double(KronIters));
  TFltV CurGradV, LearnRateV(GetParams()), LastStep(GetParams());
  LearnRateV.PutAll(LrnRate);
  TKronMtx NewProbMtx = ProbMtx;

  if(DebugMode) {  //
	  LLV.Gen(NIter, 0);
	  MtxV.Gen(NIter, 0);
  }

  for (int Iter = 0; Iter < NIter; Iter++) {
    printf("%03d] ", Iter);
    SampleGradient(WarmUp, NSamples, CurLL, CurGradV);
    for (int p = 0; p < GetParams(); p++) {
      LearnRateV[p] *= 0.95;
      if (Iter < 1) {
        while (fabs(LearnRateV[p]*CurGradV[p]) > MxStep) { LearnRateV[p] *= 0.95; }
        while (fabs(LearnRateV[p]*CurGradV[p]) < 0.02) { LearnRateV[p] *= (1.0/0.95); } //
      } else {
        //
        while (fabs(LearnRateV[p]*CurGradV[p]) > MxStep) { LearnRateV[p] *= 0.95; printf(".");}
        while (fabs(LearnRateV[p]*CurGradV[p]) < MnStep) { LearnRateV[p] *= (1.0/0.95); printf("*");}
        if (MxStep > 3*MnStep) { MxStep *= 0.95; }
      }
      NewProbMtx.At(p) = ProbMtx.At(p) + LearnRateV[p]*CurGradV[p];
      if (NewProbMtx.At(p) > 0.9999) { NewProbMtx.At(p)=0.9999; }
      if (NewProbMtx.At(p) < 0.0001) { NewProbMtx.At(p)=0.0001; }
    }
    printf("  trueE0: %.2f (%d),  estE0: %.2f (%d),  ERR: %f\n", EZero, Graph->GetEdges(),
      ProbMtx.GetMtxSum(), ProbMtx.GetEdges(KronIters), fabs(EZero-ProbMtx.GetMtxSum()));
    printf("  currLL: %.4f, deltaLL: %.4f\n", CurLL, CurLL-OldLL); //
    for (int p = 0; p < GetParams(); p++) {
      printf("    %d]  %f  <--  %f + %9f   Grad: %9.1f   Rate: %g\n", p, NewProbMtx.At(p),
        ProbMtx.At(p), (double)(LearnRateV[p]*CurGradV[p]), CurGradV[p](), LearnRateV[p]());
    }
    if (Iter+1 < NIter) { //
      ProbMtx = NewProbMtx;  ProbMtx.GetLLMtx(LLMtx); }
    OldLL=CurLL;
    printf("\n");  fflush(stdout);

	if(DebugMode) {  //
		LLV.Add(CurLL);
		MtxV.Add(NewProbMtx);
	}
  }
  printf("TotalExeTm: %s %g\n", TotalTm.GetStr(), TotalTm.GetSecs());
  ProbMtx.Dump("FITTED PARAMS", false);
  return CurLL;
}

double TKroneckerLL::GradDescent2(const int& NIter, const double& LrnRate, double MnStep, double MxStep, const int& WarmUp, const int& NSamples) {
  printf("\n----------------------------------------------------------------------\n");
  printf("GradDescent2\n");
  printf("Fitting graph on %d nodes, %d edges\n", Graph->GetNodes(), Graph->GetEdges());
  printf("Skip moves that make likelihood smaller\n");
  printf("Kron iters:  %d (== %d nodes)\n\n", KronIters(), ProbMtx.GetNodes(KronIters()));
  TExeTm IterTm, TotalTm;
  double CurLL=0, NewLL=0;
  const double EZero = pow((double) Graph->GetEdges(), 1.0/double(KronIters));
  TFltV CurGradV, NewGradV, LearnRateV(GetParams()), LastStep(GetParams());
  LearnRateV.PutAll(LrnRate);
  TKronMtx NewProbMtx=ProbMtx, CurProbMtx=ProbMtx;
  bool GoodMove = false;
  //
  for (int Iter = 0; Iter < NIter; Iter++) {
    printf("%03d] ", Iter);
    if (! GoodMove) { SampleGradient(WarmUp, NSamples, CurLL, CurGradV); }
    CurProbMtx = ProbMtx;
    //
    for (int p = 0; p < GetParams(); p++) {
      while (fabs(LearnRateV[p]*CurGradV[p]) > MxStep) { LearnRateV[p] *= 0.95; printf(".");}
      while (fabs(LearnRateV[p]*CurGradV[p]) < MnStep) { LearnRateV[p] *= (1.0/0.95); printf("*");}
      NewProbMtx.At(p) = CurProbMtx.At(p) + LearnRateV[p]*CurGradV[p];
      if (NewProbMtx.At(p) > 0.9999) { NewProbMtx.At(p)=0.9999; }
      if (NewProbMtx.At(p) < 0.0001) { NewProbMtx.At(p)=0.0001; }
      LearnRateV[p] *= 0.95;
    }
    printf("  ");
    ProbMtx=NewProbMtx;  ProbMtx.GetLLMtx(LLMtx);
    SampleGradient(WarmUp, NSamples, NewLL, NewGradV);
    if (NewLL > CurLL) { //
      printf("== Good move:\n");
      printf("  trueE0: %.2f (%d),  estE0: %.2f (%d),  ERR: %f\n", EZero, Graph->GetEdges(),
        ProbMtx.GetMtxSum(), ProbMtx.GetEdges(KronIters), fabs(EZero-ProbMtx.GetMtxSum()));
      printf("  currLL: %.4f  deltaLL: %.4f\n", CurLL, NewLL-CurLL); //
      for (int p = 0; p < GetParams(); p++) {
        printf("    %d]  %f  <--  %f + %9f   Grad: %9.1f   Rate: %g\n", p, NewProbMtx.At(p),
          CurProbMtx.At(p), (double)(LearnRateV[p]*CurGradV[p]), CurGradV[p](), LearnRateV[p]()); }
      CurLL = NewLL;
      CurGradV = NewGradV;
      GoodMove = true;
    } else {
      printf("** BAD move:\n");
      printf("  *trueE0: %.2f (%d),  estE0: %.2f (%d),  ERR: %f\n", EZero, Graph->GetEdges(),
        ProbMtx.GetMtxSum(), ProbMtx.GetEdges(KronIters), fabs(EZero-ProbMtx.GetMtxSum()));
      printf("  *curLL:  %.4f  deltaLL: %.4f\n", CurLL, NewLL-CurLL); //
      for (int p = 0; p < GetParams(); p++) {
        printf("   b%d]  %f  <--  %f + %9f   Grad: %9.1f   Rate: %g\n", p, NewProbMtx.At(p),
          CurProbMtx.At(p), (double)(LearnRateV[p]*CurGradV[p]), CurGradV[p](), LearnRateV[p]()); }
      //
      ProbMtx = CurProbMtx;  ProbMtx.GetLLMtx(LLMtx);
      GoodMove = false;
    }
    printf("\n");  fflush(stdout);
  }
  printf("TotalExeTm: %s %g\n", TotalTm.GetStr(), TotalTm.GetSecs());
  ProbMtx.Dump("FITTED PARAMS\n", false);
  return CurLL;
}

//
//
void TKroneckerLL::SetRandomEdges(const int& NEdges, const bool isDir) {
	int count = 0, added = 0, collision = 0;
	const int MtxDim = ProbMtx.GetDim();
	const double MtxSum = ProbMtx.GetMtxSum();
	TVec<TFltIntIntTr> ProbToRCPosV; //
	double CumProb = 0.0;

	for(int r = 0; r < MtxDim; r++) {
		for(int c = 0; c < MtxDim; c++) {
			const double Prob = ProbMtx.At(r, c);
			if (Prob > 0.0) {
				CumProb += Prob;
				ProbToRCPosV.Add(TFltIntIntTr(CumProb/MtxSum, r, c));
			}
		}
	}

	int Rng, Row, Col, n, NId1, NId2;
	while(added < NEdges) {
		Rng = Nodes;	Row = 0;	Col = 0;
		for (int iter = 0; iter < KronIters; iter++) {
			const double& Prob = TKronMtx::Rnd.GetUniDev();
			n = 0; while(Prob > ProbToRCPosV[n].Val1) { n++; }
			const int MtxRow = ProbToRCPosV[n].Val2;
			const int MtxCol = ProbToRCPosV[n].Val3;
			Rng /= MtxDim;
			Row += MtxRow * Rng;
			Col += MtxCol * Rng;
		}

		count++;

		NId1 = InvertPerm[Row];	NId2 = InvertPerm[Col];

		//
		if(EMType != kronEdgeMiss && IsObsEdge(NId1, NId2)) {
			continue;
		}

		if (! Graph->IsEdge(NId1, NId2)) {
			Graph->AddEdge(NId1, NId2);
			if(NId1 != NId2)	{ GEdgeV.Add(TIntTr(NId1, NId2, LEdgeV.Len())); }
			else { LSelfEdge++; }
			LEdgeV.Add(TIntTr(NId1, NId2, GEdgeV.Len()-1));
			added++;
			if (! isDir) {
				if (NId1 != NId2) {
				   Graph->AddEdge(NId2, NId1);
				   GEdgeV.Add(TIntTr(NId2, NId1, LEdgeV.Len()));
				   LEdgeV.Add(TIntTr(NId2, NId1, GEdgeV.Len()-1));
				   added++;
				}
			}
		} else { collision ++; }
	}
//
}

//
void TKroneckerLL::MetroGibbsSampleSetup(const int& WarmUp) {
	double alpha = log(ProbMtx.GetMtxSum()) / log(double(ProbMtx.GetDim()));
	int NId1 = 0, NId2 = 0;
	int NMissing;

	RestoreGraph(false);
	if(EMType == kronEdgeMiss) {
		CalcApxGraphLL();
		for (int s = 0; s < WarmUp; s++)	SampleNextPerm(NId1, NId2);
	}

	if(EMType == kronFutureLink) {
		NMissing = (int) (pow(ProbMtx.GetMtxSum(), KronIters) - pow(double(RealNodes), alpha));
	} else if(EMType == kronEdgeMiss) {
		NMissing = MissEdges;
	} else {
		NMissing = (int) (pow(ProbMtx.GetMtxSum(), KronIters) * (1.0 - pow(double(RealNodes) / double(Nodes), 2)));
	}
	NMissing = (NMissing < 1) ? 1 : NMissing;

	SetRandomEdges(NMissing, true);

	CalcApxGraphLL();
	for (int s = 0; s < WarmUp; s++)	SampleNextPerm(NId1, NId2);
}

//
void TKroneckerLL::MetroGibbsSampleNext(const int& WarmUp, const bool DLLUpdate) {
	int NId1 = 0, NId2 = 0, hit = 0, GId = 0;
	TIntTr EdgeToRemove, NewEdge;
	double RndAccept;

	if(LEdgeV.Len()) {
		for(int i = 0; i < WarmUp; i++) {
			hit = TKronMtx::Rnd.GetUniDevInt(LEdgeV.Len());

			NId1 = LEdgeV[hit].Val1;	NId2 = LEdgeV[hit].Val2;
			GId = LEdgeV[hit].Val3;
			SetRandomEdges(1, true);
			NewEdge = LEdgeV.Last();

			RndAccept = (1.0 - exp(LLMtx.GetEdgeLL(NewEdge.Val1, NewEdge.Val2, KronIters))) / (1.0 - exp(LLMtx.GetEdgeLL(NId1, NId2, KronIters)));
			RndAccept = (RndAccept > 1.0) ? 1.0 : RndAccept;

			if(TKronMtx::Rnd.GetUniDev() > RndAccept) { //
				Graph->DelEdge(NewEdge.Val1, NewEdge.Val2);
				if(NewEdge.Val1 != NewEdge.Val2) {	GEdgeV.DelLast();	}
				else {	LSelfEdge--;	}
				LEdgeV.DelLast();
			} else {	//
				Graph->DelEdge(NId1, NId2);
				LEdgeV[hit] = LEdgeV.Last();
				LEdgeV.DelLast();
				if(NId1 == NId2) {
					LSelfEdge--;
					if(NewEdge.Val1 != NewEdge.Val2) {
						GEdgeV[GEdgeV.Len()-1].Val3 = hit;
					}
				} else {
					IAssertR(GEdgeV.Last().Val3 >= 0, "Invalid indexing");

					GEdgeV[GId] = GEdgeV.Last();
					if(NewEdge.Val1 != NewEdge.Val2) {
						GEdgeV[GId].Val3 = hit;
					}
					LEdgeV[GEdgeV[GId].Val3].Val3 = GId;
					GEdgeV.DelLast();
				}

      			LogLike += LLMtx.GetApxNoEdgeLL(EdgeToRemove.Val1, EdgeToRemove.Val2, KronIters) - LLMtx.GetEdgeLL(EdgeToRemove.Val1, EdgeToRemove.Val2, KronIters);
      			LogLike += -LLMtx.GetApxNoEdgeLL(NewEdge.Val1, NewEdge.Val2, KronIters) + LLMtx.GetEdgeLL(NewEdge.Val1, NewEdge.Val2, KronIters);

				if(DLLUpdate) {
  					for (int p = 0; p < LLMtx.Len(); p++) {
						GradV[p] += LLMtx.GetApxNoEdgeDLL(p, EdgeToRemove.Val1, EdgeToRemove.Val2, KronIters) - LLMtx.GetEdgeDLL(p, EdgeToRemove.Val1, EdgeToRemove.Val2, KronIters);
						GradV[p] += -LLMtx.GetApxNoEdgeDLL(p, NewEdge.Val1, NewEdge.Val2, KronIters) + LLMtx.GetEdgeDLL(p, NewEdge.Val1, NewEdge.Val2, KronIters);
					}
				}
			}
		}
	}

//
	for (int s = 0; s < WarmUp; s++) {
		if(SampleNextPerm(NId1, NId2)) {
			if(DLLUpdate)	UpdateGraphDLL(NId1, NId2);
		}
	}
}

//
void TKroneckerLL::RunEStep(const int& GibbsWarmUp, const int& WarmUp, const int& NSamples, TFltV& LLV, TVec<TFltV>& DLLV) {
	TExeTm ExeTm, TotalTm;
	LLV.Gen(NSamples, 0);
	DLLV.Gen(NSamples, 0);

	ExeTm.Tick();
	for(int i = 0; i < 2; i++)	MetroGibbsSampleSetup(WarmUp);
	printf("  Warm-Up [%u] : %s\n", WarmUp, ExeTm.GetTmStr());
	CalcApxGraphLL();
	for(int i = 0; i < GibbsWarmUp; i++)	MetroGibbsSampleNext(10, false);
	printf("  Gibbs Warm-Up [%u] : %s\n", GibbsWarmUp, ExeTm.GetTmStr());

	ExeTm.Tick();
	CalcApxGraphLL();
	CalcApxGraphDLL();
	for(int i = 0; i < NSamples; i++) {
		MetroGibbsSampleNext(50, false);

		LLV.Add(LogLike);
		DLLV.Add(GradV);

		int OnePercent = (i+1) % (NSamples / 10);
		if(OnePercent == 0) {
			int TenPercent = ((i+1) / (NSamples / 10)) * 10;
			printf("  %3u%% done : %s\n", TenPercent, ExeTm.GetTmStr());
		}
	}
}

//
double TKroneckerLL::RunMStep(const TFltV& LLV, const TVec<TFltV>& DLLV, const int& GradIter, const double& LrnRate, double MnStep, double MxStep) {
	TExeTm IterTm, TotalTm;
	double OldLL=LogLike, CurLL=0;
	const double alpha = log(double(RealEdges)) / log(double(RealNodes));
	const double EZero = pow(double(ProbMtx.GetDim()), alpha);

	TFltV CurGradV(GetParams()), LearnRateV(GetParams()), LastStep(GetParams());
	LearnRateV.PutAll(LrnRate);

	TKronMtx NewProbMtx = ProbMtx;
	const int NSamples = LLV.Len();
	const int ReCalcLen = NSamples / 10;

	for (int s = 0; s < LLV.Len(); s++) {
		CurLL += LLV[s];
		for(int p = 0; p < GetParams(); p++) { CurGradV[p] += DLLV[s][p]; }
	}
	CurLL /= NSamples;
	for(int p = 0; p < GetParams(); p++) { CurGradV[p] /= NSamples; }

	double MaxLL = CurLL;
	TKronMtx MaxProbMtx = ProbMtx;
	TKronMtx OldProbMtx = ProbMtx;

	for (int Iter = 0; Iter < GradIter; Iter++) {
		printf("    %03d] ", Iter+1);
		IterTm.Tick();

		for (int p = 0; p < GetParams(); p++) {
			if (Iter < 1) {
				while (fabs(LearnRateV[p]*CurGradV[p]) > MxStep) { LearnRateV[p] *= 0.95; }
				while (fabs(LearnRateV[p]*CurGradV[p]) < 5 * MnStep) { LearnRateV[p] *= (1.0/0.95); } //
			} else {
			//
				while (fabs(LearnRateV[p]*CurGradV[p]) > MxStep) { LearnRateV[p] *= 0.95; printf(".");}
				while (fabs(LearnRateV[p]*CurGradV[p]) < MnStep) { LearnRateV[p] *= (1.0/0.95); printf("*");}
				if (MxStep > 3*MnStep) { MxStep *= 0.95; }
			}
			NewProbMtx.At(p) = ProbMtx.At(p) + LearnRateV[p]*CurGradV[p];
			if (NewProbMtx.At(p) > 0.9999) { NewProbMtx.At(p)=0.9999; }
			if (NewProbMtx.At(p) < 0.0001) { NewProbMtx.At(p)=0.0001; }
			LearnRateV[p] *= 0.95;
		}
		printf("  trueE0: %.2f (%u from %u),  estE0: %.2f (%u from %u),  ERR: %f\n", EZero, RealEdges(), RealNodes(), ProbMtx.GetMtxSum(), Graph->GetEdges(), Graph->GetNodes(), fabs(EZero-ProbMtx.GetMtxSum()));
		printf("      currLL: %.4f, deltaLL: %.4f\n", CurLL, CurLL-OldLL); //
		for (int p = 0; p < GetParams(); p++) {
			printf("      %d]  %f  <--  %f + %9f   Grad: %9.1f   Rate: %g\n", p, NewProbMtx.At(p),
			ProbMtx.At(p), (double)(LearnRateV[p]*CurGradV[p]), CurGradV[p](), LearnRateV[p]());
		}

		OldLL=CurLL;
		if(Iter == GradIter - 1) {
			break;
		}

		CurLL = 0;
		CurGradV.PutAll(0.0);
		TFltV OneDLL;

		CalcApxGraphLL();
		CalcApxGraphDLL();

		for(int s = 0; s < NSamples; s++) {
			ProbMtx = OldProbMtx;  ProbMtx.GetLLMtx(LLMtx);
			MetroGibbsSampleNext(10, true);
			ProbMtx = NewProbMtx;  ProbMtx.GetLLMtx(LLMtx);
			if(s % ReCalcLen == ReCalcLen/2) {
				CurLL += CalcApxGraphLL();
				OneDLL = CalcApxGraphDLL();
			} else {
				CurLL += LogLike;
				OneDLL = GradV;
			}
			for(int p = 0; p < GetParams(); p++) {
				CurGradV[p] += OneDLL[p];
			}
		}
		CurLL /= NSamples;

		if(MaxLL < CurLL) {
			MaxLL = CurLL;	MaxProbMtx = ProbMtx;
		}

		printf("    Time: %s\n", IterTm.GetTmStr());
		printf("\n");  fflush(stdout);
	}
	ProbMtx = MaxProbMtx;	ProbMtx.GetLLMtx(LLMtx);

	printf("    FinalLL : %f,   TotalExeTm: %s\n", MaxLL, TotalTm.GetTmStr());
	ProbMtx.Dump("    FITTED PARAMS", false);

	return MaxLL;
}

//
void TKroneckerLL::RunKronEM(const int& EMIter, const int& GradIter, double LrnRate, double MnStep, double MxStep, const int& GibbsWarmUp, const int& WarmUp, const int& NSamples, const TKronEMType& Type, const int& NMissing) {
	printf("\n----------------------------------------------------------------------\n");
	printf("Fitting graph on %d nodes, %d edges\n", int(RealNodes), int(RealEdges));
	printf("Kron iters:  %d (== %d nodes)\n\n", KronIters(), ProbMtx.GetNodes(KronIters()));

	TFltV LLV(NSamples);
	TVec<TFltV> DLLV(NSamples);
	//

	EMType = Type;
	MissEdges = NMissing;
	AppendIsoNodes();
	SetRndPerm();

	if(DebugMode) {
		LLV.Gen(EMIter, 0);
		MtxV.Gen(EMIter, 0);
	}

	for(int i = 0; i < EMIter; i++) {
		printf("\n----------------------------------------------------------------------\n");
		printf("%03d EM-iter] E-Step\n", i+1);
		RunEStep(GibbsWarmUp, WarmUp, NSamples, LLV, DLLV);
		printf("\n\n");

		printf("%03d EM-iter] M-Step\n", i+1);
		double CurLL = RunMStep(LLV, DLLV, GradIter, LrnRate, MnStep, MxStep);
		printf("\n\n");

		if(DebugMode) {
			LLV.Add(CurLL);
			MtxV.Add(ProbMtx);
		}
	}

	RestoreGraph();
}



void GetMinMax(const TFltPrV& XYValV, double& Min, double& Max, const bool& ResetMinMax) {
  if (ResetMinMax) { Min = TFlt::Mx;  Max = TFlt::Mn; }
  for (int i = 0; i < XYValV.Len(); i++) {
    Min = TMath::Mn(Min, XYValV[i].Val2.Val);
    Max = TMath::Mx(Max, XYValV[i].Val2.Val);
  }
}

void PlotGrad(const TFltPrV& EstLLV, const TFltPrV& TrueLLV, const TVec<TFltPrV>& GradVV, const TFltPrV& AcceptV, const TStr& OutFNm, const TStr& Desc) {
  double Min, Max, Min1, Max1;
  //
  { TGnuPlot GP("sLL-"+OutFNm, TStr::Fmt("Log-likelihood (avg 1k samples). %s", Desc.CStr()), true);
  GP.AddPlot(EstLLV, gpwLines, "Esimated LL", "linewidth 1");
  if (! TrueLLV.Empty()) { GP.AddPlot(TrueLLV, gpwLines, "TRUE LL", "linewidth 1"); }
  //
  //
  GP.SetXYLabel("Sample Index (time)", "Log-likelihood");
  GP.SavePng(); }
  //
  { TGnuPlot GP("sAcc-"+OutFNm, TStr::Fmt("Pct. accepted rnd moves (over 1k samples). %s", Desc.CStr()), true);
  GP.AddPlot(AcceptV, gpwLines, "Pct accepted swaps", "linewidth 1");
  GP.SetXYLabel("Sample Index (time)", "Pct accept permutation swaps");
  GP.SavePng(); }
  //
  TGnuPlot GPAll("sGradAll-"+OutFNm, TStr::Fmt("Gradient (avg 1k samples). %s", Desc.CStr()), true);
  GetMinMax(GradVV[0], Min1, Max1, true);
  for (int g = 0; g < GradVV.Len(); g++) {
    GPAll.AddPlot(GradVV[g], gpwLines, TStr::Fmt("param %d", g+1), "linewidth 1");
    GetMinMax(GradVV[g], Min1, Max1, false);
    TGnuPlot GP(TStr::Fmt("sGrad%02d-", g+1)+OutFNm, TStr::Fmt("Gradient (avg 1k samples). %s", Desc.CStr()), true);
    GP.AddPlot(GradVV[g], gpwLines, TStr::Fmt("param id %d", g+1), "linewidth 1");
    GetMinMax(GradVV[g], Min, Max, true);
    GP.SetYRange((int)floor(Min-1), (int)ceil(Max+1));
    GP.SetXYLabel("Sample Index (time)", "Gradient");
    GP.SavePng();
  }
  GPAll.SetYRange((int)floor(Min1-1), (int)ceil(Max1+1));
  GPAll.SetXYLabel("Sample Index (time)", "Gradient");
  GPAll.SavePng();
}

void PlotAutoCorrelation(const TFltV& ValV, const int& MaxK, const TStr& OutFNm, const TStr& Desc) {
  double Avg=0.0, Var=0.0;
  for (int i = 0; i < ValV.Len(); i++) { Avg += ValV[i]; }
  Avg /= (double) ValV.Len();
  for (int i = 0; i < ValV.Len(); i++) { Var += TMath::Sqr(ValV[i]-Avg); }
  TFltPrV ACorrV;
  for (int k = 0; k < TMath::Mn(ValV.Len(), MaxK); k++) {
    double corr = 0.0;
    for (int i = 0; i < ValV.Len() - k; i++) {
      corr += (ValV[i]-Avg)*(ValV[i+k]-Avg);
    }
    ACorrV.Add(TFltPr(k, corr/Var));
  }
  //
  TGnuPlot GP("sAutoCorr-"+OutFNm, TStr::Fmt("AutoCorrelation (%d samples). %s", ValV.Len(), Desc.CStr()), true);
  GP.AddPlot(ACorrV, gpwLines, "", "linewidth 1");
  GP.SetXYLabel("Lag, k", "Autocorrelation, r_k");
  GP.SavePng();
}

//
//
TFltV TKroneckerLL::TestSamplePerm(const TStr& OutFNm, const int& WarmUp, const int& NSamples, const TKronMtx& TrueMtx, const bool& DoPlot) {
  printf("Sample permutations: %s (warm-up: %s)\n", TInt::GetMegaStr(NSamples).CStr(), TInt::GetMegaStr(WarmUp).CStr());
  int NId1=0, NId2=0, NAccept=0;
  TExeTm ExeTm;
  const int PlotLen = NSamples/1000+1;
  double TrueLL=-1, AvgLL=0.0;
  TFltV AvgGradV(GetParams());
  TFltPrV TrueLLV(PlotLen, 0); //
  TFltPrV EstLLV(PlotLen, 0);  //
  TFltPrV AcceptV;             //
  TFltV SampleLLV(NSamples, 0);
  TVec<TFltPrV> GradVV(GetParams());
  for (int g = 0; g < GetParams(); g++) {
    GradVV[g].Gen(PlotLen, 0); }
  if (! TrueMtx.Empty()) {
    TIntV PermV=NodePerm;  TKronMtx CurMtx=ProbMtx;  ProbMtx.Dump();
    InitLL(TrueMtx);  SetOrderPerm();  CalcApxGraphLL();  printf("TrueLL: %f\n", LogLike());
    TrueLL=LogLike;  InitLL(CurMtx); NodePerm=PermV;
  }
  CalcApxGraphLL();
  printf("LogLike at start:       %f\n", LogLike());
  if (WarmUp > 0) {
    EstLLV.Add(TFltPr(0, LogLike));
    if (TrueLL != -1) { TrueLLV.Add(TFltPr(0, TrueLL)); }
    for (int s = 0; s < WarmUp; s++) { SampleNextPerm(NId1, NId2); }
    printf("  warm-up:%s,", ExeTm.GetTmStr());  ExeTm.Tick();
  }
  printf("LogLike afterm warm-up: %f\n", LogLike());
  CalcApxGraphLL(); //
  CalcApxGraphDLL();
  EstLLV.Add(TFltPr(WarmUp, LogLike));
  if (TrueLL != -1) { TrueLLV.Add(TFltPr(WarmUp, TrueLL)); }
  printf("  recalculated:         %f\n", LogLike());
  //
  printf("  sampling (average per 1000 samples)\n");
  TVec<TFltV> SamplVV(5);
  for (int s = 0; s < NSamples; s++) {
    if (SampleNextPerm(NId1, NId2)) { //
      UpdateGraphDLL(NId1, NId2);  NAccept++; }
    for (int m = 0; m < AvgGradV.Len(); m++) { AvgGradV[m] += GradV[m]; }
    AvgLL += GetLL();
    SampleLLV.Add(GetLL());

    if (s > 0 && s % 1000 == 0) {
      printf(".");
      for (int g = 0; g < AvgGradV.Len(); g++) {
        GradVV[g].Add(TFltPr(WarmUp+s, AvgGradV[g] / 1000.0)); }
      EstLLV.Add(TFltPr(WarmUp+s, AvgLL / 1000.0));
      if (TrueLL != -1) { TrueLLV.Add(TFltPr(WarmUp+s, TrueLL)); }
      AcceptV.Add(TFltPr(WarmUp+s, NAccept/1000.0));
      //

      if (s % 100000 == 0 && DoPlot) {
        const TStr Desc = TStr::Fmt("P(NodeSwap)=%g. Nodes: %d, Edges: %d, Params: %d, WarmUp: %s, Samples: %s", PermSwapNodeProb(),
          Graph->GetNodes(), Graph->GetEdges(), GetParams(), TInt::GetMegaStr(WarmUp).CStr(), TInt::GetMegaStr(NSamples).CStr());
        PlotGrad(EstLLV, TrueLLV, GradVV, AcceptV, OutFNm, Desc);
        for (int n = 0; n < SamplVV.Len(); n++) {
          PlotAutoCorrelation(SamplVV[n], 1000, TStr::Fmt("%s-n%d", OutFNm.CStr(), n), Desc); }
        printf("  samples: %d, time: %s, samples/s: %.1f\n", s, ExeTm.GetTmStr(), double(s+1)/ExeTm.GetSecs());
      }
      AvgLL = 0;  AvgGradV.PutAll(0);  NAccept=0;
    }
  }
  if (DoPlot) {
    const TStr Desc = TStr::Fmt("P(NodeSwap)=%g. Nodes: %d, Edges: %d, Params: %d, WarmUp: %s, Samples: %s", PermSwapNodeProb(),
      Graph->GetNodes(), Graph->GetEdges(), GetParams(), TInt::GetMegaStr(WarmUp).CStr(), TInt::GetMegaStr(NSamples).CStr());
    PlotGrad(EstLLV, TrueLLV, GradVV, AcceptV, OutFNm, Desc);
    for (int n = 0; n < SamplVV.Len(); n++) {
      PlotAutoCorrelation(SamplVV[n], 1000, TStr::Fmt("%s-n%d", OutFNm.CStr(), n), Desc); }
  }
  return SampleLLV; //
}

void McMcGetAvgAvg(const TFltV& AvgJV, double& AvgAvg) {
  AvgAvg = 0.0;
  for (int j = 0; j < AvgJV.Len(); j++) {
    AvgAvg += AvgJV[j]; }
  AvgAvg /= AvgJV.Len();
}

void McMcGetAvgJ(const TVec<TFltV>& ChainLLV, TFltV& AvgJV) {
  for (int j = 0; j < ChainLLV.Len(); j++) {
    const TFltV& ChainV = ChainLLV[j];
    double Avg = 0;
    for (int i = 0; i < ChainV.Len(); i++) {
      Avg += ChainV[i];
    }
    AvgJV.Add(Avg/ChainV.Len());
  }
}

//
double TKroneckerLL::CalcChainR2(const TVec<TFltV>& ChainLLV) {
  const double J = ChainLLV.Len();
  const double K = ChainLLV[0].Len();
  TFltV AvgJV;    McMcGetAvgJ(ChainLLV, AvgJV);
  double AvgAvg;  McMcGetAvgAvg(AvgJV, AvgAvg);
  IAssert(AvgJV.Len() == ChainLLV.Len());
  double InChainVar=0, OutChainVar=0;
  //
  for (int j = 0; j < AvgJV.Len(); j++) {
    OutChainVar += TMath::Sqr(AvgJV[j] - AvgAvg); }
  OutChainVar = OutChainVar * (K/double(J-1));
  printf("*** %g chains of len %g\n", J, K);
  printf("  ** between chain var: %f\n", OutChainVar);
  //
  for (int j = 0; j < AvgJV.Len(); j++) {
    const TFltV& ChainV = ChainLLV[j];
    for (int k = 0; k < ChainV.Len(); k++) {
      InChainVar += TMath::Sqr(ChainV[k] - AvgJV[j]); }
  }
  InChainVar = InChainVar * 1.0/double(J*(K-1));
  printf("  ** within chain var: %f\n", InChainVar);
  const double PostVar = (K-1)/K * InChainVar + 1.0/K * OutChainVar;
  printf("  ** posterior var: %f\n", PostVar);
  const double ScaleRed = sqrt(PostVar/InChainVar);
  printf("  ** scale reduction (< 1.2): %f\n\n", ScaleRed);
  return ScaleRed;
}

//
void TKroneckerLL::ChainGelmapRubinPlot(const TVec<TFltV>& ChainLLV, const TStr& OutFNm, const TStr& Desc) {
  TFltPrV LenR2V; //
  TVec<TFltV> SmallLLV(ChainLLV.Len());
  const int K = ChainLLV[0].Len();
  const int Buckets=1000;
  const int BucketSz = K/Buckets;
  for (int b = 1; b < Buckets; b++) {
    const int End = TMath::Mn(BucketSz*b, K-1);
    for (int c = 0; c < ChainLLV.Len(); c++) {
      ChainLLV[c].GetSubValV(0, End, SmallLLV[c]); }
    LenR2V.Add(TFltPr(End, TKroneckerLL::CalcChainR2(SmallLLV)));
  }
  LenR2V.Add(TFltPr(K, TKroneckerLL::CalcChainR2(ChainLLV)));
  TGnuPlot::PlotValV(LenR2V, TStr::Fmt("gelman-%s", OutFNm.CStr()), TStr::Fmt("%s. %d chains of len %d. BucketSz: %d.",
    Desc.CStr(), ChainLLV.Len(), ChainLLV[0].Len(), BucketSz), "Chain length", "Potential scale reduction");
}

//
TFltQu TKroneckerLL::TestKronDescent(const bool& DoExact, const bool& TruePerm, double LearnRate, const int& WarmUp, const int& NSamples, const TKronMtx& TrueParam) {
  printf("Test gradient descent on a synthetic kronecker graphs:\n");
  if (DoExact) { printf("  -- Exact gradient calculations\n"); }
  else { printf("  -- Approximate gradient calculations\n"); }
  if (TruePerm) { printf("  -- No permutation sampling (use true permutation)\n"); }
  else { printf("  -- Sample permutations (start with degree permutation)\n"); }
  TExeTm IterTm;
  int Iter;
  double OldLL=0, MyLL=0, AvgAbsErr, AbsSumErr;
  TFltV MyGradV, SDevV;
  TFltV LearnRateV(GetParams());  LearnRateV.PutAll(LearnRate);
  if (TruePerm) {
    SetOrderPerm();
  }
  else {

    //
    printf("DEGREE  PERMUTATION\n");  SetDegPerm();
  }
  for (Iter = 0; Iter < 100; Iter++) {
    if (TruePerm) {
      //
      if (DoExact) { CalcGraphDLL();  CalcGraphLL(); } //
      else { CalcApxGraphDLL();  CalcApxGraphLL(); }   //
      MyLL = LogLike;  MyGradV = GradV;
    } else {
      printf(".");
      //
      SampleGradient(WarmUp, NSamples, MyLL, MyGradV);
    }
    printf("%d] LL: %g, ", Iter, MyLL);
    AvgAbsErr = TKronMtx::GetAvgAbsErr(ProbMtx, TrueParam);
    AbsSumErr = fabs(ProbMtx.GetMtxSum() - TrueParam.GetMtxSum());
    printf("  avgAbsErr: %.4f, absSumErr: %.4f, newLL: %.2f, deltaLL: %.2f\n", AvgAbsErr, AbsSumErr, MyLL, OldLL-MyLL);
    for (int p = 0; p < GetParams(); p++) {
      //
      LearnRateV[p] *= 0.9;
      //
      while (fabs(LearnRateV[p]*MyGradV[p]) > 0.1) { LearnRateV[p] *= 0.9; }
      //
      while (fabs(LearnRateV[p]*MyGradV[p]) < 0.001) { LearnRateV[p] *= (1.0/0.9); }
      //
      printf("    %d]  %f  <--  %f + %f    lrnRate:%g\n", p, ProbMtx.At(p) + LearnRateV[p]*MyGradV[p],
        ProbMtx.At(p), (double)(LearnRateV[p]*MyGradV[p]), LearnRateV[p]());
      ProbMtx.At(p) = ProbMtx.At(p) + LearnRateV[p]*MyGradV[p];
      //
      if (ProbMtx.At(p) > 0.99) { ProbMtx.At(p)=0.99; }
      if (ProbMtx.At(p) < 0.01) { ProbMtx.At(p)=0.01; }
    }
    ProbMtx.GetLLMtx(LLMtx);  OldLL = MyLL;
    if (AvgAbsErr < 0.01) { printf("***CONVERGED!\n");  break; }
    printf("\n");  fflush(stdout);
  }
  TrueParam.Dump("True  Thetas", true);
  ProbMtx.Dump("Final Thetas", true);
  printf("  AvgAbsErr: %f\n  AbsSumErr: %f\n  Iterations: %d\n", AvgAbsErr, AbsSumErr, Iter);
  printf("Iteration run time: %s, sec: %g\n\n", IterTm.GetTmStr(), IterTm.GetSecs());
  return TFltQu(AvgAbsErr, AbsSumErr, Iter, IterTm.GetSecs());
}

void PlotTrueAndEst(const TStr& OutFNm, const TStr& Desc, const TStr& YLabel, const TFltPrV& EstV, const TFltPrV& TrueV) {
  TGnuPlot GP(OutFNm, Desc.CStr(), true);
  GP.AddPlot(EstV, gpwLinesPoints, YLabel, "linewidth 1 pointtype 6 pointsize 1");
  if (! TrueV.Empty()) { GP.AddPlot(TrueV, gpwLines, "TRUE"); }
  GP.SetXYLabel("Gradient descent iterations", YLabel);
  GP.SavePng();
}

void TKroneckerLL::GradDescentConvergence(const TStr& OutFNm, const TStr& Desc1, const bool& SamplePerm, const int& NIters,
 double LearnRate, const int& WarmUp, const int& NSamples, const int& AvgKGraphs, const TKronMtx& TrueParam) {
  TExeTm IterTm;
  int Iter;
  double OldLL=0, MyLL=0, AvgAbsErr=0, AbsSumErr=0;
  TFltV MyGradV, SDevV;
  TFltV LearnRateV(GetParams());  LearnRateV.PutAll(LearnRate);
  TFltPrV EZeroV, DiamV, Lambda1V, Lambda2V, AvgAbsErrV, AvgLLV;
  TFltPrV TrueEZeroV, TrueDiamV, TrueLambda1V, TrueLambda2V, TrueLLV;
  TFltV SngValV;  TSnap::GetSngVals(Graph, 2, SngValV);  SngValV.Sort(false);
  const double TrueEZero = pow((double) Graph->GetEdges(), 1.0/double(KronIters));
  const double TrueEffDiam = TSnap::GetAnfEffDiam(Graph, false, 10);
  const double TrueLambda1 = SngValV[0];
  const double TrueLambda2 = SngValV[1];
  if (! TrueParam.Empty()) {
    const TKronMtx CurParam = ProbMtx;  ProbMtx.Dump();
    InitLL(TrueParam);  SetOrderPerm();  CalcApxGraphLL(); printf("TrueLL: %f\n", LogLike());
    OldLL = LogLike;  InitLL(CurParam);
  }
  const double TrueLL = OldLL;
  if (! SamplePerm) { SetOrderPerm(); } else { SetDegPerm(); }
  for (Iter = 0; Iter < NIters; Iter++) {
    if (! SamplePerm) {
      //
      CalcApxGraphDLL();  CalcApxGraphLL();   //
      MyLL = LogLike;  MyGradV = GradV;
    } else {
      //
      SampleGradient(WarmUp, NSamples, MyLL, MyGradV);
    }
    double SumDiam=0, SumSngVal1=0, SumSngVal2=0;
    for (int trial = 0; trial < AvgKGraphs; trial++) {
      //
      PNGraph KronGraph = TKronMtx::GenFastKronecker(ProbMtx, KronIters, true, 0); //
      //
      SngValV.Clr(true);  TSnap::GetSngVals(KronGraph, 2, SngValV);  SngValV.Sort(false);
      SumDiam += TSnap::GetAnfEffDiam(KronGraph, false, 10);
      SumSngVal1 += SngValV[0];  SumSngVal2 += SngValV[1];
    }
    //
    AvgLLV.Add(TFltPr(Iter, MyLL));
    EZeroV.Add(TFltPr(Iter, ProbMtx.GetMtxSum()));
    DiamV.Add(TFltPr(Iter, SumDiam/double(AvgKGraphs)));
    Lambda1V.Add(TFltPr(Iter, SumSngVal1/double(AvgKGraphs)));
    Lambda2V.Add(TFltPr(Iter, SumSngVal2/double(AvgKGraphs)));
    TrueLLV.Add(TFltPr(Iter, TrueLL));
    TrueEZeroV.Add(TFltPr(Iter, TrueEZero));
    TrueDiamV.Add(TFltPr(Iter, TrueEffDiam));
    TrueLambda1V.Add(TFltPr(Iter, TrueLambda1));
    TrueLambda2V.Add(TFltPr(Iter, TrueLambda2));
    if (Iter % 10 == 0) {
      const TStr Desc = TStr::Fmt("%s. Iter: %d, G(%d, %d)  K(%d, %d)", Desc1.Empty()?OutFNm.CStr():Desc1.CStr(),
        Iter, Graph->GetNodes(), Graph->GetEdges(), ProbMtx.GetNodes(KronIters), ProbMtx.GetEdges(KronIters));
      PlotTrueAndEst("LL."+OutFNm, Desc, "Average LL", AvgLLV, TrueLLV);
      PlotTrueAndEst("E0."+OutFNm, Desc, "E0 (expected number of edges)", EZeroV, TrueEZeroV);
      PlotTrueAndEst("Diam."+OutFNm+"-Diam", Desc, "Effective diameter", DiamV, TrueDiamV);
      PlotTrueAndEst("Lambda1."+OutFNm, Desc, "Lambda 1", Lambda1V, TrueLambda1V);
      PlotTrueAndEst("Lambda2."+OutFNm, Desc, "Lambda 2", Lambda2V, TrueLambda2V);
      if (! TrueParam.Empty()) {
        PlotTrueAndEst("AbsErr."+OutFNm, Desc, "Average Absolute Error", AvgAbsErrV, TFltPrV()); }
    }
    if (! TrueParam.Empty()) {
      AvgAbsErr = TKronMtx::GetAvgAbsErr(ProbMtx, TrueParam);
      AvgAbsErrV.Add(TFltPr(Iter, AvgAbsErr));
    } else { AvgAbsErr = 1.0; }
    //
    AbsSumErr = fabs(ProbMtx.GetMtxSum() - TrueEZero);
    //
    for (int p = 0; p < GetParams(); p++) {
      LearnRateV[p] *= 0.99;
      while (fabs(LearnRateV[p]*MyGradV[p]) > 0.1) { LearnRateV[p] *= 0.99; printf(".");}
      while (fabs(LearnRateV[p]*MyGradV[p]) < 0.002) { LearnRateV[p] *= (1.0/0.95); printf("*");}
      printf("    %d]  %f  <--  %f + %9f   Grad: %9.1f,  Rate:%g\n", p, ProbMtx.At(p) + LearnRateV[p]*MyGradV[p],
        ProbMtx.At(p), (double)(LearnRateV[p]*MyGradV[p]), MyGradV[p](), LearnRateV[p]());
      ProbMtx.At(p) = ProbMtx.At(p) + LearnRateV[p]*MyGradV[p];
      //
      if (ProbMtx.At(p) > 1.0) { ProbMtx.At(p)=1.0; }
      if (ProbMtx.At(p) < 0.001) { ProbMtx.At(p)=0.001; }
    }
    printf("%d] LL: %g, ", Iter, MyLL);
    printf("  avgAbsErr: %.4f, absSumErr: %.4f, newLL: %.2f, deltaLL: %.2f\n", AvgAbsErr, AbsSumErr, MyLL, OldLL-MyLL);
    if (AvgAbsErr < 0.001) { printf("***CONVERGED!\n");  break; }
    printf("\n");  fflush(stdout);
    ProbMtx.GetLLMtx(LLMtx);  OldLL = MyLL;
  }
  TrueParam.Dump("True  Thetas", true);
  ProbMtx.Dump("Final Thetas", true);
  printf("  AvgAbsErr: %f\n  AbsSumErr: %f\n  Iterations: %d\n", AvgAbsErr, AbsSumErr, Iter);
  printf("Iteration run time: %s, sec: %g\n\n", IterTm.GetTmStr(), IterTm.GetSecs());
}

//
void TKroneckerLL::TestBicCriterion(const TStr& OutFNm, const TStr& Desc1, const PNGraph& G, const int& GradIters,
 double LearnRate, const int& WarmUp, const int& NSamples, const int& TrueN0) {
  TFltPrV BicV, MdlV, LLV;
  const double rndGP = G->GetEdges()/TMath::Sqr(double(G->GetNodes()));
  const double RndGLL = G->GetEdges()*log(rndGP )+ (TMath::Sqr(double(G->GetNodes()))-G->GetEdges())*log(1-rndGP);
  LLV.Add(TFltPr(1, RndGLL));
  BicV.Add(TFltPr(1, -RndGLL + 0.5*TMath::Sqr(1)*log(TMath::Sqr(G->GetNodes()))));
  MdlV.Add(TFltPr(1, -RndGLL + 32*TMath::Sqr(1)+2*(log((double)1)+log((double)G->GetNodes()))));
  for (int NZero = 2; NZero < 10; NZero++) {
    const TKronMtx InitKronMtx = TKronMtx::GetInitMtx(NZero, G->GetNodes(), G->GetEdges());
    InitKronMtx.Dump("INIT PARAM", true);
    TKroneckerLL KronLL(G, InitKronMtx);
    KronLL.SetPerm('d'); //
    const double LastLL = KronLL.GradDescent(GradIters, LearnRate, 0.001, 0.01, WarmUp, NSamples);
    LLV.Add(TFltPr(NZero, LastLL));
    BicV.Add(TFltPr(NZero, -LastLL + 0.5*TMath::Sqr(NZero)*log(TMath::Sqr(G->GetNodes()))));
    MdlV.Add(TFltPr(NZero, -LastLL + 32*TMath::Sqr(NZero)+2*(log((double)NZero)+log((double)KronLL.GetKronIters()))));
    { TGnuPlot GP("LL-"+OutFNm, Desc1);
    GP.AddPlot(LLV, gpwLinesPoints, "Log-likelihood", "linewidth 1 pointtype 6 pointsize 2");
    GP.SetXYLabel("NZero", "Log-Likelihood");  GP.SavePng(); }
    { TGnuPlot GP("BIC-"+OutFNm, Desc1);
    GP.AddPlot(BicV, gpwLinesPoints, "BIC", "linewidth 1 pointtype 6 pointsize 2");
    GP.SetXYLabel("NZero", "BIC");  GP.SavePng(); }
    { TGnuPlot GP("MDL-"+OutFNm, Desc1);
    GP.AddPlot(MdlV, gpwLinesPoints, "MDL", "linewidth 1 pointtype 6 pointsize 2");
    GP.SetXYLabel("NZero", "MDL");  GP.SavePng(); }
  }
}

void TKroneckerLL::TestGradDescent(const int& KronIters, const int& KiloSamples, const TStr& Permutation) {
  const TStr OutFNm = TStr::Fmt("grad-%s%d-%dk", Permutation.CStr(), KronIters, KiloSamples);
  TKronMtx KronParam = TKronMtx::GetMtx("0.8 0.6; 0.6 0.4");
  PNGraph Graph  = TKronMtx::GenFastKronecker(KronParam, KronIters, true, 0);
  TKroneckerLL KronLL(Graph, KronParam);
  TVec<TFltV> GradVV(4), SDevVV(4);  TFltV XValV;
  int NId1 = 0, NId2 = 0, NAccept = 0;
  TVec<TMom> GradMomV(4);
  TExeTm ExeTm;
  if (Permutation == "r") KronLL.SetRndPerm();
  else if (Permutation == "d") KronLL.SetDegPerm();
  else if (Permutation == "o") KronLL.SetOrderPerm();
  else FailR("Unknown permutation (r,d,o)");
  KronLL.CalcApxGraphLL();
  KronLL.CalcApxGraphDLL();
  for (int s = 0; s < 1000*KiloSamples; s++) {
    if (KronLL.SampleNextPerm(NId1, NId2)) { //
      KronLL.UpdateGraphDLL(NId1, NId2);  NAccept++; }
    if (s > 50000) { //
      for (int m = 0; m < 4; m++) { GradVV[m].Add(KronLL.GradV[m]); }
      if ((s+1) % 1000 == 0) {
        printf(".");
        for (int m = 0; m < 4; m++) { GradVV[m].Add(KronLL.GradV[m]); }
        XValV.Add((s+1));
        if ((s+1) % 100000 == 0) {
          TGnuPlot GP(OutFNm, TStr::Fmt("Gradient vs. samples. %d nodes, %d edges", Graph->GetNodes(), Graph->GetEdges()), true);
          for (int g = 0; g < GradVV.Len(); g++) {
            GP.AddPlot(XValV, GradVV[g], gpwLines, TStr::Fmt("grad %d", g)); }
          GP.SetXYLabel("sample index","log Gradient");
          GP.SavePng();
        }
      }
    }
  }
  printf("\n");
  for (int m = 0; m < 4; m++) {
    GradMomV[m].Def();
    printf("grad %d: mean: %12f  sDev: %12f  median: %12f\n", m,
      GradMomV[m].GetMean(), GradMomV[m].GetSDev(), GradMomV[m].GetMedian());
  }
}



//
//
//
int TKronNoise::RemoveNodeNoise(PNGraph& Graph, const int& NNodes, const bool Random) {
	IAssert(NNodes > 0 && NNodes < (Graph->GetNodes() / 2));

	int i = 0;
	TIntV ShufflePerm;
	Graph->GetNIdV(ShufflePerm);
	if(Random) {
		ShufflePerm.Shuffle(TKronMtx::Rnd);
		for(i = 0; i < NNodes; i++) {
			Graph->DelNode(int(ShufflePerm[i]));
		}
	} else {
		for(i = 0; i < NNodes; i++) {
			Graph->DelNode(int(ShufflePerm[ShufflePerm.Len() - 1 - i]));
		}
	}

	return Graph->GetNodes();
}

int TKronNoise::RemoveNodeNoise(PNGraph& Graph, const double& Rate, const bool Random) {
	IAssert(Rate > 0 && Rate < 0.5);
	return TKronNoise::RemoveNodeNoise(Graph, (int) floor(Rate * double(Graph->GetNodes())), Random);
}

int TKronNoise::FlipEdgeNoise(PNGraph& Graph, const int& NEdges, const bool Random) {
	IAssert(NEdges > 0 && NEdges < Graph->GetEdges());

	const int Nodes = Graph->GetNodes();
	const int Edges = Graph->GetEdges();
	int Src, Dst;

	TIntV NIdV, TempV;
	TIntPrV ToAdd, ToDel;
	Graph->GetNIdV(NIdV);

	ToAdd.Gen(NEdges / 2, 0);
	for(int i = 0; i < NEdges / 2; i++) {
		Src = NIdV[TKronMtx::Rnd.GetUniDevInt(Nodes)];
		Dst = NIdV[TKronMtx::Rnd.GetUniDevInt(Nodes)];
		if(Graph->IsEdge(Src, Dst)) {	i--;	continue;	}

		ToAdd.Add(TIntPr(Src, Dst));
	}

	ToDel.Gen(Edges, 0);
	for(TNGraph::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
		ToDel.Add(TIntPr(EI.GetSrcNId(), EI.GetDstNId()));
	}
	ToDel.Shuffle(TKronMtx::Rnd);

	for(int i = 0; i < NEdges / 2; i++) {
		Graph->DelEdge(ToDel[i].Val1, ToDel[i].Val2);
		Graph->AddEdge(ToAdd[i].Val1, ToAdd[i].Val2);
	}

	return Graph->GetEdges();
}

int TKronNoise::FlipEdgeNoise(PNGraph& Graph, const double& Rate, const bool Random) {
	IAssert(Rate > 0 && Rate < 0.5);
	return TKronNoise::FlipEdgeNoise(Graph, (int) floor(Rate * double(Graph->GetEdges())), Random);
}

int TKronNoise::RemoveEdgeNoise(PNGraph& Graph, const int& NEdges) {
	IAssert(NEdges > 0 && NEdges < Graph->GetEdges());

	TIntPrV ToDel;

	ToDel.Gen(Graph->GetEdges(), 0);
	for(TNGraph::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
		if(EI.GetSrcNId() != EI.GetDstNId()) {
			ToDel.Add(TIntPr(EI.GetSrcNId(), EI.GetDstNId()));
		}
	}
	ToDel.Shuffle(TKronMtx::Rnd);

	for(int i = 0; i < NEdges; i++) {
		Graph->DelEdge(ToDel[i].Val1, ToDel[i].Val2);
	}

	return Graph->GetEdges();
}

int TKronNoise::RemoveEdgeNoise(PNGraph& Graph, const double& Rate) {
	IAssert(Rate > 0 && Rate < 0.5);
	return TKronNoise::RemoveEdgeNoise(Graph, (int) floor(Rate * double(Graph->GetEdges())));
}



//
//
void TKronMaxLL::SetPerm(const char& PermId) {
  if (PermId == 'o') KronLL.SetOrderPerm();
  else if (PermId == 'd') KronLL.SetDegPerm();
  else if (PermId == 'r') KronLL.SetRndPerm();
  else FailR("Unknown permutation type (o,d,r)");
}



//
void TKronMaxLL::RoundTheta(const TFltV& ThetaV, TFltV& NewThetaV) {
  NewThetaV.Gen(ThetaV.Len());
  for (int i = 0; i < ThetaV.Len(); i++) {
    NewThetaV[i] = TMath::Round(ThetaV[i], 3); }
}

//
void TKronMaxLL::RoundTheta(const TFltV& ThetaV, TKronMtx& Kronecker) {
  Kronecker.GenMtx((int)sqrt((double)ThetaV.Len()));
  for (int i = 0; i < ThetaV.Len(); i++) {
    Kronecker.At(i) = TMath::Round(ThetaV[i], 3); }
}

void TKronMaxLL::Test() {
  TKronMtx::PutRndSeed(1);
  TKronMtx KronParam = TKronMtx::GetMtx("0.8 0.7; 0.6 0.5");
  PNGraph Graph  = TKronMtx::GenFastKronecker(KronParam, 8, true, 1);

  TKronMaxLL KronMaxLL(Graph, TFltV::GetV(0.9, 0.7, 0.5, 0.3));
  KronMaxLL.SetPerm('d');
  //
}

