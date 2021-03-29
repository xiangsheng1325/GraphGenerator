#ifndef snap_kronecker_h
#define snap_kronecker_h

#include "Snap.h"

//
//
class TKroneckerLL;
typedef TPt<TKroneckerLL> PKroneckerLL;

class TKronMtx {
public:
  static const double NInf;
  static TRnd Rnd;
private:
  TInt MtxDim;
  TFltV SeedMtx;
public:
  TKronMtx() : MtxDim(-1), SeedMtx() { }
  TKronMtx(const int& Dim) : MtxDim(Dim), SeedMtx(Dim*Dim) { }
  TKronMtx(const TFltV& SeedMatrix);
  TKronMtx(const TKronMtx& Kronecker) : MtxDim(Kronecker.MtxDim), SeedMtx(Kronecker.SeedMtx) { }
  void SaveTxt(const TStr& OutFNm) const;
  TKronMtx& operator = (const TKronMtx& Kronecker);
  bool operator == (const TKronMtx& Kronecker) const { return SeedMtx==Kronecker.SeedMtx; }
  int GetPrimHashCd() const { return SeedMtx.GetPrimHashCd(); }
  int GetSecHashCd() const { return SeedMtx.GetSecHashCd(); }

  //
  int GetDim() const { return MtxDim; }
  int Len() const { return SeedMtx.Len(); }
  bool Empty() const { return SeedMtx.Empty(); }
  bool IsProbMtx() const; //

  TFltV& GetMtx() { return SeedMtx; }
  const TFltV& GetMtx() const { return SeedMtx; }
  void SetMtx(const TFltV& ParamV) { SeedMtx = ParamV; }
  void SetRndMtx(const int& MtxDim, const double& MinProb = 0.0);
  void PutAllMtx(const double& Val) { SeedMtx.PutAll(Val); }
  void GenMtx(const int& Dim) { MtxDim=Dim;  SeedMtx.Gen(Dim*Dim); }
  void SetEpsMtx(const double& Eps1, const double& Eps0, const int& Eps1Val=1, const int& Eps0Val=0);
  void SetForEdges(const int& Nodes, const int& Edges); //
  void AddRndNoise(const double& SDev);
  TStr GetMtxStr() const;

  const double& At(const int& Row, const int& Col) const { return SeedMtx[MtxDim*Row+Col].Val; }
  double& At(const int& Row, const int& Col) { return SeedMtx[MtxDim*Row+Col].Val; }
  const double& At(const int& ValN) const { return SeedMtx[ValN].Val; }
  double& At(const int& ValN) { return SeedMtx[ValN].Val; }

  int GetNodes(const int& NIter) const;
  int GetEdges(const int& NIter) const;
  int GetKronIter(const int& Nodes) const;
  int GetNZeroK(const PNGraph& Graph) const; //
  double GetEZero(const int& Edges, const int& KronIter) const;
  double GetMtxSum() const;
  double GetRowSum(const int& RowId) const;
  double GetColSum(const int& ColId) const;

  void ToOneMinusMtx();
  void GetLLMtx(TKronMtx& LLMtx);
  void GetProbMtx(TKronMtx& ProbMtx);
  void Swap(TKronMtx& KronMtx);

  //
  double GetEdgeProb(int NId1, int NId2, const int& NKronIters) const; //
  double GetNoEdgeProb(int NId1, int NId2, const int& NKronIters) const; //
  double GetEdgeLL(int NId1, int NId2, const int& NKronIters) const; //
  double GetNoEdgeLL(int NId1, int NId2, const int& NKronIters) const; //
  double GetApxNoEdgeLL(int NId1, int NId2, const int& NKronIters) const; //
  bool IsEdgePlace(int NId1, int NId2, const int& NKronIters, const double& ProbTresh) const; //
  //
  double GetEdgeDLL(const int& ParamId, int NId1, int NId2, const int& NKronIters) const; //
  double GetNoEdgeDLL(const int& ParamId, int NId1, int NId2, const int& NKronIters) const; //
  double GetApxNoEdgeDLL(const int& ParamId, int NId1, int NId2, const int& NKronIters) const; //

  //
  static uint GetNodeSig(const double& OneProb = 0.5);
  double GetEdgeProb(const uint& NId1Sig, const uint& NId2Sig, const int& NIter) const;

  //
  PNGraph GenThreshGraph(const double& Thresh) const;
  PNGraph GenRndGraph(const double& RndFact=1.0) const;

  static int GetKronIter(const int& GNodes, const int& SeedMtxSz);
  //
  static PNGraph GenKronecker(const TKronMtx& SeedMtx, const int& NIter, const bool& IsDir, const int& Seed=0);
  static PNGraph GenFastKronecker(const TKronMtx& SeedMtx, const int& NIter, const bool& IsDir, const int& Seed=0);
  static PNGraph GenFastKronecker(const TKronMtx& SeedMtx, const int& NIter, const int& Edges, const bool& IsDir, const int& Seed=0);
  static PNGraph GenDetKronecker(const TKronMtx& SeedMtx, const int& NIter, const bool& IsDir);
  static void PlotCmpGraphs(const TKronMtx& SeedMtx, const PNGraph& Graph, const TStr& OutFNm, const TStr& Desc);
  static void PlotCmpGraphs(const TKronMtx& SeedMtx1, const TKronMtx& SeedMtx2, const PNGraph& Graph, const TStr& OutFNm, const TStr& Desc);
  static void PlotCmpGraphs(const TVec<TKronMtx>& SeedMtxV, const PNGraph& Graph, const TStr& FNmPref, const TStr& Desc);

  static void KronMul(const TKronMtx& LeftPt, const TKronMtx& RightPt, TKronMtx& OutMtx);
  static void KronSum(const TKronMtx& LeftPt, const TKronMtx& RightPt, TKronMtx& OutMtx); //
  static void KronPwr(const TKronMtx& KronPt, const int& NIter, TKronMtx& OutMtx);

  void Dump(const TStr& MtxNm = TStr(), const bool& Sort = false) const;
  static double GetAvgAbsErr(const TKronMtx& Kron1, const TKronMtx& Kron2); //
  static double GetAvgFroErr(const TKronMtx& Kron1, const TKronMtx& Kron2); //
  static TKronMtx GetMtx(TStr MatlabMtxStr);
  static TKronMtx GetRndMtx(const int& Dim, const double& MinProb);
  static TKronMtx GetInitMtx(const int& Dim, const int& Nodes, const int& Edges);
  static TKronMtx GetInitMtx(const TStr& MtxStr, const int& Dim, const int& Nodes, const int& Edges);
  static TKronMtx GetMtxFromNm(const TStr& MtxNm);
  static TKronMtx LoadTxt(const TStr& MtxFNm);
  static void PutRndSeed(const int& Seed) { TKronMtx::Rnd.PutSeed(Seed); }
};

//
//

enum TKronEMType {  kronNodeMiss = 0, kronFutureLink, kronEdgeMiss }; //

class TKroneckerLL {
public:
private:
  TCRef CRef;
  PNGraph Graph;         //
  TInt Nodes, KronIters;

  TFlt PermSwapNodeProb; //
//
  TIntTrV GEdgeV;        //
  TIntTrV LEdgeV;        //
  TInt LSelfEdge;        //
  TIntV NodePerm;        //
  TIntV InvertPerm;      //

  TInt RealNodes;	//
  TInt RealEdges;	//

  TKronMtx ProbMtx, LLMtx; //
  TFlt LogLike; //
  TFltV GradV;  //

  TKronEMType EMType;	//
  TInt MissEdges;		//

  TBool DebugMode;		//
  TFltV LLV;			//
  TVec<TKronMtx> MtxV;	//

public:
  //
  //
  //
  //
  TKroneckerLL() : Nodes(-1), KronIters(-1), PermSwapNodeProb(0.2), RealNodes(-1), RealEdges(-1), LogLike(TKronMtx::NInf), EMType(kronNodeMiss), MissEdges(-1), DebugMode(false) { }
  TKroneckerLL(const PNGraph& GraphPt, const TFltV& ParamV, const double& PermPSwapNd=0.2);
  TKroneckerLL(const PNGraph& GraphPt, const TKronMtx& ParamMtx, const double& PermPSwapNd=0.2);
  TKroneckerLL(const PNGraph& GraphPt, const TKronMtx& ParamMtx, const TIntV& NodeIdPermV, const double& PermPSwapNd=0.2);
  static PKroneckerLL New() { return new TKroneckerLL(); }
  static PKroneckerLL New(const PNGraph& GraphPt, const TKronMtx& ParamMtx, const double& PermPSwapNd=0.1);
  static PKroneckerLL New(const PNGraph& GraphPt, const TKronMtx& ParamMtx, const TIntV& NodeIdPermV, const double& PermPSwapNd=0.2);

  int GetNodes() const { return Nodes; }
  int GetKronIters() const { return KronIters; }
  PNGraph GetGraph() const { return Graph; }
  void SetGraph(const PNGraph& GraphPt);
  const TKronMtx& GetProbMtx() const { return ProbMtx; }
  const TKronMtx& GetLLMtx() const { return LLMtx; }
  int GetParams() const { return ProbMtx.Len(); }
  int GetDim() const { return ProbMtx.GetDim(); }

  void SetDebug(const bool Debug) { DebugMode = Debug; }
  const TFltV& GetLLHist() const { return LLV; }
  const TVec<TKronMtx>& GetParamHist() const { return MtxV; }

  //
  bool IsObsNode(const int& NId) const { IAssert(RealNodes > 0);	return (NId < RealNodes);	}
  bool IsObsEdge(const int& NId1, const int& NId2) const { IAssert(RealNodes > 0);	return ((NId1 < RealNodes) && (NId2 < RealNodes));	}
  bool IsLatentNode(const int& NId) const { return !IsObsNode(NId);	}
  bool IsLatentEdge(const int& NId1, const int& NId2) const { return !IsObsEdge(NId1, NId2);	}

  //
  void SetPerm(const char& PermId);
  void SetOrderPerm(); //
  void SetRndPerm();   //
  void SetDegPerm();   //
  void SetBestDegPerm();	//
  void SetPerm(const TIntV& NodePermV) { NodePerm = NodePermV; SetIPerm(NodePerm); }
  void SetIPerm(const TIntV& Perm);	//
  const TIntV& GetPermV() const { return NodePerm; }

  //
  void AppendIsoNodes();
  void RestoreGraph(const bool RestoreNodes = true);

  //
  double GetFullGraphLL() const;
  double GetFullRowLL(int RowId) const;
  double GetFullColLL(int ColId) const;
  //
  double GetEmptyGraphLL() const;
  double GetApxEmptyGraphLL() const;
  //
  void InitLL(const TFltV& ParamV);
  void InitLL(const TKronMtx& ParamMtx);
  void InitLL(const PNGraph& GraphPt, const TKronMtx& ParamMtx);
  double CalcGraphLL();
  double CalcApxGraphLL();
  double GetLL() const { return LogLike; }
  double GetAbsErr() const { return fabs(pow((double)Graph->GetEdges(), 1.0/double(KronIters)) - ProbMtx.GetMtxSum()); }
  double NodeLLDelta(const int& NId) const;
  double SwapNodesLL(const int& NId1, const int& NId2);
  bool SampleNextPerm(int& NId1, int& NId2); //

  //
  double GetEmptyGraphDLL(const int& ParamId) const;
  double GetApxEmptyGraphDLL(const int& ParamId) const;
  const TFltV& CalcGraphDLL();
  const TFltV& CalcFullApxGraphDLL();
  const TFltV& CalcApxGraphDLL();
  double NodeDLLDelta(const int ParamId, const int& NId) const;
  void UpdateGraphDLL(const int& SwapNId1, const int& SwapNId2);
  const TFltV& GetDLL() const { return GradV; }
  double GetDLL(const int& ParamId) const { return GradV[ParamId]; }

  //
  void SampleGradient(const int& WarmUp, const int& NSamples, double& AvgLL, TFltV& GradV);
  double GradDescent(const int& NIter, const double& LrnRate, double MnStep, double MxStep, const int& WarmUp, const int& NSamples);
  double GradDescent2(const int& NIter, const double& LrnRate, double MnStep, double MxStep, const int& WarmUp, const int& NSamples);

  //
  void SetRandomEdges(const int& NEdges, const bool isDir = true);
  void MetroGibbsSampleSetup(const int& WarmUp);
  void MetroGibbsSampleNext(const int& WarmUp, const bool DLLUpdate = false);
  void RunEStep(const int& GibbsWarmUp, const int& WarmUp, const int& NSamples, TFltV& LLV, TVec<TFltV>& DLLV);
  double RunMStep(const TFltV& LLV, const TVec<TFltV>& DLLV, const int& GradIter, const double& LrnRate, double MnStep, double MxStep);
  void RunKronEM(const int& EMIter, const int& GradIter, double LrnRate, double MnStep, double MxStep, const int& GibbsWarmUp, const int& WarmUp, const int& NSamples, const TKronEMType& Type = kronNodeMiss, const int& NMissing = -1);



  TFltV TestSamplePerm(const TStr& OutFNm, const int& WarmUp, const int& NSamples, const TKronMtx& TrueMtx, const bool& DoPlot=true);
  static double CalcChainR2(const TVec<TFltV>& ChainLLV);
  static void ChainGelmapRubinPlot(const TVec<TFltV>& ChainLLV, const TStr& OutFNm, const TStr& Desc);
  TFltQu TestKronDescent(const bool& DoExact, const bool& TruePerm, double LearnRate, const int& WarmUp, const int& NSamples, const TKronMtx& TrueParam);
  void GradDescentConvergence(const TStr& OutFNm, const TStr& Desc1, const bool& SamplePerm, const int& NIters,
    double LearnRate, const int& WarmUp, const int& NSamples, const int& AvgKGraphs, const TKronMtx& TrueParam);
  static void TestBicCriterion(const TStr& OutFNm, const TStr& Desc1, const PNGraph& G, const int& GradIters,
    double LearnRate, const int& WarmUp, const int& NSamples, const int& TrueN0);
  static void TestGradDescent(const int& KronIters, const int& KiloSamples, const TStr& Permutation);
  friend class TPt<TKroneckerLL>;
};


//
//
class TKronNoise {
public:
	TKronNoise() {};
	static int RemoveNodeNoise(PNGraph& Graph, const int& NNodes, const bool Random = true);
	static int RemoveNodeNoise(PNGraph& Graph, const double& Rate, const bool Random = true);
	static int FlipEdgeNoise(PNGraph& Graph, const int& NEdges, const bool Random = true);
	static int FlipEdgeNoise(PNGraph& Graph, const double& Rate, const bool Random = true);
	static int RemoveEdgeNoise(PNGraph& Graph, const int& NEdges);
	static int RemoveEdgeNoise(PNGraph& Graph, const double& Rate);
};


//
//
class TKronMaxLL {
public:
  class TFEval {
  public:
    TFlt LogLike;
    TFltV GradV;
  public:
    TFEval() : LogLike(0), GradV() { }
    TFEval(const TFlt& LL, const TFltV& DLL) : LogLike(LL), GradV(DLL) { }
    TFEval(const TFEval& FEval) : LogLike(FEval.LogLike), GradV(FEval.GradV) { }
    TFEval& operator = (const TFEval& FEval) { if (this!=&FEval) {
      LogLike=FEval.LogLike; GradV=FEval.GradV; } return *this; }
  };
private:
  //
  THash<TKronMtx, TFEval> FEvalH; //
  TKroneckerLL KronLL;
public:
  TKronMaxLL(const PNGraph& GraphPt, const TKronMtx& StartParam) : KronLL(GraphPt, StartParam) { }
  void SetPerm(const char& PermId);

  void GradDescent(const int& NIter, const double& LrnRate, const double& MnStep, const double& MxStep,
    const double& WarmUp, const double& NSamples);



  static void RoundTheta(const TFltV& ThetaV, TFltV& NewThetaV);
  static void RoundTheta(const TFltV& ThetaV, TKronMtx& Kronecker);

  static void Test();
};

//
//
class TKronMomentsFit {
public:
  double Edges, Hairpins, Tripins, Triads;
public:
  TKronMomentsFit(const PUNGraph& G) {
    Edges=0; Hairpins=0; Tripins=0; Triads=0;
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++) {
      const int d = NI.GetOutDeg();
      Edges += d;
      Hairpins += d*(d-1.0);
      Tripins += d*(d-1.0)*(d-2.0);
    }
    Edges /= 2.0;
    Hairpins /= 2.0;
    Tripins /= 6.0;
    int64 ot,ct;
    Triads = (int) TSnap::GetTriads(G, ot, ct)/3.0;
    printf("E:%g\tH:%g\tT:%g\tD:%g\n", Edges, Hairpins, Tripins, Triads);
  }

  TFltQu EstABC(const int& R) {
    const double Step = 0.01;
    double MinScore=TFlt::Mx;
    double A=0, B=0, C=0;
    //
    for (double a = 1.0; a > Step; a-=Step) {
      for (double b = Step; b <= 1.0; b+=Step) {
        for (double c = Step; c <= a; c+=Step) {
          double EE = ( pow(a+2*b+c, R) - pow(a+c, R) ) / 2.0;
          double EH = ( pow(pow(a+b,2) + pow(b+c,2), R)
                             -2*pow(a*(a+b)+c*(c+b), R)
                             -pow(a*a + 2*b*b + c*c, R)
                             +2*pow(a*a + c*c, R) ) / 2.0;
          double ET = ( pow(pow(a+b,3)+pow(b+c,3), R)
                             -3*pow(a*pow(a+b,2)+c*pow(b+c,2), R)
                             -3*pow(a*a*a + c*c*c + b*(a*a+c*c) + b*b*(a+c) + 2*b*b*b ,R)
                             +2*pow(a*a*a + 2*b*b*b + c*c*c, R)
                             +5*pow(a*a*a + c*c*c + b*b*(a+c), R)
                             +4*pow(a*a*a + c*c*c + b*(a*a+c*c), R)
                             -6*pow(a*a*a + c*c*c, R) ) / 6.0;
          double ED = ( pow(a*a*a + 3*b*b*(a+c) + c*c*c, R)
                             -3*pow(a*(a*a+b*b) + c*(b*b+c*c), R)
                             +2*pow(a*a*a+c*c*c, R) ) / 6.0;
          if (EE < 0) { EE = 1; }
          if (EH < 0) { EH = 1; }
          if (ET < 0) { ET = 1; }
          if (ED < 0) { ED = 1; }
          //
          double Score = pow(Edges-EE,2)/EE + pow(Hairpins-EH ,2)/EH + pow(Tripins-ET, 2)/ET + pow(Triads-ED, 2)/ED;
          //
          //
          if (MinScore > Score || (a==0.9 && b==0.6 && c==0.2) || (TMath::IsInEps(a-0.99,1e-6) && TMath::IsInEps(b-0.57,1e-6) && TMath::IsInEps(c-0.05,1e-6)))
          {
            printf("%.03f %.03f %0.03f %10.4f  %10.10g\t%10.10g\t%10.10g\t%10.10g\n", a,b,c, log10(Score), EE, EH, ET, ED);
            //
            A=a; B=b; C=c; MinScore=Score;
          }
        }
      }
    }
    printf("\t\t\t      %10.10g\t%10.10g\t%10.10g\t%10.10g\n", Edges, Hairpins, Tripins, Triads);
    return TFltQu(A,B,C,MinScore);
  }

  static void Test() {
    TFIn FIn("as20.ngraph");
    PUNGraph G = TSnap::ConvertGraph<PUNGraph>(TNGraph::Load(FIn));
    //
    //
    TSnap::PrintInfo(G);
    TSnap::DelSelfEdges(G);
    TSnap::PrintInfo(G);
    TKronMomentsFit Fit(G);
    printf("iter %d\n", TKronMtx::GetKronIter(G->GetNodes(), 2));
    Fit.EstABC(TKronMtx::GetKronIter(G->GetNodes(), 2)); //
  }
};



#endif
