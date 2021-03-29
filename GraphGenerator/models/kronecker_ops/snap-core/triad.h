#ifndef TRIAD_H
#define TRIAD_H
namespace TSnap {
template <class PGraph> double GetClustCf(const PGraph& Graph, int SampleNodes=-1);
template <class PGraph> double GetClustCf(const PGraph& Graph, TFltPrV& DegToCCfV, int SampleNodes=-1);
template <class PGraph> double GetClustCf(const PGraph& Graph, TFltPrV& DegToCCfV, int64& ClosedTriads, int64& OpenTriads, int SampleNodes=-1);
template <class PGraph> double GetClustCfAll(const PGraph& Graph, TFltPrV& DegToCCfV, int64& ClosedTriadsX, int64& OpenTriadsX, int SampleNodes=-1);
template <class PGraph> double GetNodeClustCf(const PGraph& Graph, const int& NId);
template <class PGraph> void GetNodeClustCf(const PGraph& Graph, TIntFltH& NIdCCfH);
template <class PGraph> int64 GetTriads(const PGraph& Graph, int SampleNodes=-1);
template <class PGraph> int64 GetTriads(const PGraph& Graph, int64& ClosedTriadsX, int64& OpenTriadsX, int SampleNodes);
template <class PGraph> int64 GetTriadsAll(const PGraph& Graph, int64& ClosedTriadsX, int64& OpenTriadsX, int SampleNodes=-1);
template <class PGraph> void GetTriads(const PGraph& Graph, TIntTrV& NIdCOTriadV, int SampleNodes=-1);
template <class PGraph> int GetTriadEdges(const PGraph& Graph, int SampleEdges=-1);
template <class PGraph> int GetNodeTriads(const PGraph& Graph, const int& NId);
template <class PGraph> int GetNodeTriads(const PGraph& Graph, const int& NId, int& ClosedNTriadsX, int& OpenNTriadsX);
template <class PGraph> int GetNodeTriadsAll(const PGraph& Graph, const int& NId, int& ClosedNTriadsX, int& OpenNTriadsX);
template <class PGraph>
int GetNodeTriads(const PGraph& Graph, const int& NId, const TIntSet& GroupSet, int& InGroupEdgesX, int& InOutGroupEdgesX, int& OutGroupEdgesX);
template <class PGraph> void GetTriadParticip(const PGraph& Graph, TIntPrV& TriadCntV);
template<class PGraph> int GetCmnNbrs(const PGraph& Graph, const int& NId1, const int& NId2);
template<class PGraph> int GetCmnNbrs(const PGraph& Graph, const int& NId1, const int& NId2, TIntV& NbrV);
template<class PGraph> int GetLen2Paths(const PGraph& Graph, const int& NId1, const int& NId2);
template<class PGraph> int GetLen2Paths(const PGraph& Graph, const int& NId1, const int& NId2, TIntV& NbrV);
template<class PGraph> int64 GetTriangleCnt(const PGraph& Graph);
template<class PGraph> void MergeNbrs(TIntV& NeighbourV, const typename PGraph::TObj::TNodeI& NI);
template <class PGraph> void GetUniqueNbrV(const PGraph& Graph, const int& NId, TIntV& NbrV);
int GetCommon(TIntV& A, TIntV& B);
template <class PGraph> double GetClustCf(const PGraph& Graph, int SampleNodes) {
  TIntTrV NIdCOTriadV;
  GetTriads(Graph, NIdCOTriadV, SampleNodes);
  if (NIdCOTriadV.Empty()) { return 0.0; }
  double SumCcf = 0.0;
  for (int i = 0; i < NIdCOTriadV.Len(); i++) {
    const double OpenCnt = NIdCOTriadV[i].Val2()+NIdCOTriadV[i].Val3();
    if (OpenCnt > 0) {
      SumCcf += NIdCOTriadV[i].Val2() / OpenCnt; }
  }
  IAssert(SumCcf>=0);
  return SumCcf / double(NIdCOTriadV.Len());
}
template <class PGraph> double GetClustCf(const PGraph& Graph, TFltPrV& DegToCCfV, int SampleNodes) {
  TIntTrV NIdCOTriadV;
  GetTriads(Graph, NIdCOTriadV, SampleNodes);
  if (NIdCOTriadV.Empty()) {
    DegToCCfV.Clr(false);
    return 0.0;
  }
  THash<TInt, TFltPr> DegSumCnt;
  double SumCcf = 0.0;
  for (int i = 0; i < NIdCOTriadV.Len(); i++) {
    const int D = NIdCOTriadV[i].Val2()+NIdCOTriadV[i].Val3();
    const double Ccf = D!=0 ? NIdCOTriadV[i].Val2() / double(D) : 0.0;
    TFltPr& SumCnt = DegSumCnt.AddDat(Graph->GetNI(NIdCOTriadV[i].Val1).GetDeg());
    SumCnt.Val1 += Ccf;
    SumCnt.Val2 += 1;
    SumCcf += Ccf;
  }

  DegToCCfV.Gen(DegSumCnt.Len(), 0);
  for (int d = 0; d  < DegSumCnt.Len(); d++) {
    DegToCCfV.Add(TFltPr(DegSumCnt.GetKey(d).Val, double(DegSumCnt[d].Val1()/DegSumCnt[d].Val2())));
  }
  DegToCCfV.Sort();
  return SumCcf / double(NIdCOTriadV.Len());
}
template <class PGraph>
double GetClustCf(const PGraph& Graph, TFltPrV& DegToCCfV, int64& ClosedTriads, int64& OpenTriads, int SampleNodes) {
  TIntTrV NIdCOTriadV;
  GetTriads(Graph, NIdCOTriadV, SampleNodes);
  if (NIdCOTriadV.Empty()) {
    DegToCCfV.Clr(false);
    ClosedTriads = 0;
    OpenTriads = 0;
    return 0.0;
  }
  THash<TInt, TFltPr> DegSumCnt;
  double SumCcf = 0.0;
  int64 closedTriads = 0;
  int64 openTriads = 0;
  for (int i = 0; i < NIdCOTriadV.Len(); i++) {
    const int D = NIdCOTriadV[i].Val2()+NIdCOTriadV[i].Val3();
    const double Ccf = D!=0 ? NIdCOTriadV[i].Val2() / double(D) : 0.0;
    closedTriads += NIdCOTriadV[i].Val2;
    openTriads += NIdCOTriadV[i].Val3;
    TFltPr& SumCnt = DegSumCnt.AddDat(Graph->GetNI(NIdCOTriadV[i].Val1).GetDeg());
    SumCnt.Val1 += Ccf;
    SumCnt.Val2 += 1;
    SumCcf += Ccf;
  }

  DegToCCfV.Gen(DegSumCnt.Len(), 0);
  for (int d = 0; d  < DegSumCnt.Len(); d++) {
    DegToCCfV.Add(TFltPr(DegSumCnt.GetKey(d).Val, DegSumCnt[d].Val1()/DegSumCnt[d].Val2()));
  }


  ClosedTriads = closedTriads/int64(3);
  OpenTriads = openTriads;
  DegToCCfV.Sort();
  return SumCcf / double(NIdCOTriadV.Len());
}
template <class PGraph>
double GetClustCfAll(const PGraph& Graph, TFltPrV& DegToCCfV, int64& ClosedTriads, int64& OpenTriads, int SampleNodes) {
  return GetClustCf(Graph, DegToCCfV, ClosedTriads, OpenTriads, SampleNodes);
}
template <class PGraph>
double GetNodeClustCf(const PGraph& Graph, const int& NId) {
  int Open, Closed;
  GetNodeTriads(Graph, NId, Open, Closed);

  return (Open+Closed)==0 ? 0 : double(Open)/double(Open+Closed);
}
template <class PGraph>
void GetNodeClustCf(const PGraph& Graph, TIntFltH& NIdCCfH) {
  TIntTrV NIdCOTriadV;
  GetTriads(Graph, NIdCOTriadV);
  NIdCCfH.Clr(false);
  for (int i = 0; i < NIdCOTriadV.Len(); i++) {
    const int D = NIdCOTriadV[i].Val2()+NIdCOTriadV[i].Val3();
    const double CCf = D!=0 ? NIdCOTriadV[i].Val2() / double(D) : 0.0;
    NIdCCfH.AddDat(NIdCOTriadV[i].Val1, CCf);
  }
}
template <class PGraph>
int64 GetTriads(const PGraph& Graph, int SampleNodes) {
  int64 OpenTriads, ClosedTriads;
  return GetTriads(Graph, ClosedTriads, OpenTriads, SampleNodes);
}
template <class PGraph>
int64 GetTriads(const PGraph& Graph, int64& ClosedTriads, int64& OpenTriads, int SampleNodes) {
  TIntTrV NIdCOTriadV;
  GetTriads(Graph, NIdCOTriadV, SampleNodes);
  uint64 closedTriads = 0;
  uint64 openTriads = 0;
  for (int i = 0; i < NIdCOTriadV.Len(); i++) {
    closedTriads += NIdCOTriadV[i].Val2;
    openTriads += NIdCOTriadV[i].Val3;
  }


  ClosedTriads = int64(closedTriads/3);
  OpenTriads = int64(openTriads);
  return ClosedTriads;
}
template <class PGraph>
int64 GetTriadsAll(const PGraph& Graph, int64& ClosedTriads, int64& OpenTriads, int SampleNodes) {
  return GetTriads(Graph, ClosedTriads, OpenTriads, SampleNodes);
}
template <class PGraph>
void GetTriads_v0(const PGraph& Graph, TIntTrV& NIdCOTriadV, int SampleNodes) {
  const bool IsDir = Graph->HasFlag(gfDirected);
  TIntSet NbrH;
  TIntV NIdV;
  TRnd Rnd(0);
  Graph->GetNIdV(NIdV);
  NIdV.Shuffle(Rnd);
  if (SampleNodes == -1) {
    SampleNodes = Graph->GetNodes(); }
  NIdCOTriadV.Clr(false);
  NIdCOTriadV.Reserve(SampleNodes);
  for (int node = 0; node < SampleNodes; node++) {
    typename PGraph::TObj::TNodeI NI = Graph->GetNI(NIdV[node]);
    if (NI.GetDeg() < 2) {
      NIdCOTriadV.Add(TIntTr(NI.GetId(), 0, 0));
      continue;
    }

    NbrH.Clr(false);
    for (int e = 0; e < NI.GetOutDeg(); e++) {
      if (NI.GetOutNId(e) != NI.GetId()) {
        NbrH.AddKey(NI.GetOutNId(e)); }
    }
    if (IsDir) {
      for (int e = 0; e < NI.GetInDeg(); e++) {
        if (NI.GetInNId(e) != NI.GetId()) {
          NbrH.AddKey(NI.GetInNId(e)); }
      }
    }

    int OpenCnt=0, CloseCnt=0;
    for (int srcNbr = 0; srcNbr < NbrH.Len(); srcNbr++) {
      const typename PGraph::TObj::TNodeI SrcNode = Graph->GetNI(NbrH.GetKey(srcNbr));
      for (int dstNbr = srcNbr+1; dstNbr < NbrH.Len(); dstNbr++) {
        const int dstNId = NbrH.GetKey(dstNbr);
        if (SrcNode.IsNbrNId(dstNId)) { CloseCnt++; }
        else { OpenCnt++; }
      }
    }
    IAssert(2*(OpenCnt+CloseCnt) == NbrH.Len()*(NbrH.Len()-1));
    NIdCOTriadV.Add(TIntTr(NI.GetId(), CloseCnt, OpenCnt));
  }
}
template <class PGraph>
void GetTriads(const PGraph& Graph, TIntTrV& NIdCOTriadV, int SampleNodes) {
  const bool IsDir = Graph->HasFlag(gfDirected);
  TIntSet NbrH;
  TIntV NIdV;

  TRnd Rnd(1);
  int NNodes;
  TIntV Nbrs;
  int NId;
  int64 hcount;
  hcount = 0;
  NNodes = Graph->GetNodes();
  Graph->GetNIdV(NIdV);
  NIdV.Shuffle(Rnd);
  if (SampleNodes == -1) {
    SampleNodes = NNodes;
  }
  int MxId = -1;
  for (int i = 0; i < NNodes; i++) {
    if (NIdV[i] > MxId) {
      MxId = NIdV[i];
    }
  }
  TVec<TIntV> NbrV(MxId + 1);
  if (IsDir) {

    for (int node = 0; node < NNodes; node++) {
      int NId = NIdV[node];
      NbrV[NId] = TIntV();
      GetUniqueNbrV(Graph, NId, NbrV[NId]);
    }
  } else {

    for (int node = 0; node < NNodes; node++) {
      int NId = NIdV[node];
      typename PGraph::TObj::TNodeI NI = Graph->GetNI(NId);
      NbrV[NId] = TIntV();
      NbrV[NId].Reserve(NI.GetOutDeg());
      NbrV[NId].Reduce(0);
      for (int i = 0; i < NI.GetOutDeg(); i++) {
        NbrV[NId].Add(NI.GetOutNId(i));
      }
    }
  }
  NIdCOTriadV.Clr(false);
  NIdCOTriadV.Reserve(SampleNodes);
  for (int node = 0; node < SampleNodes; node++) {
    typename PGraph::TObj::TNodeI NI = Graph->GetNI(NIdV[node]);
    int NLen;
    NId = NI.GetId();
    hcount++;
    if (NI.GetDeg() < 2) {
      NIdCOTriadV.Add(TIntTr(NId, 0, 0));
      continue;
    }
    Nbrs = NbrV[NId];
    NLen = Nbrs.Len();

    int OpenCnt1 = 0, CloseCnt1 = 0;
    for (int srcNbr = 0; srcNbr < NLen; srcNbr++) {
      int Count = GetCommon(NbrV[NbrV[NId][srcNbr]],Nbrs);
      CloseCnt1 += Count;
    }
    CloseCnt1 /= 2;
    OpenCnt1 = (NLen*(NLen-1))/2 - CloseCnt1;
    NIdCOTriadV.Add(TIntTr(NId, CloseCnt1, OpenCnt1));
  }
}
#if 0
template<class PGraph>
int64 CountTriangles(const PGraph& Graph) {
  THash<TInt, TInt> H;
  TIntV MapV;
  int ind = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++)   {
    H.AddDat(NI.GetId(), ind);
    MapV.Add(NI.GetId());
    ind += 1;
  }
  TVec<TIntV> HigherDegNbrV(ind);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 0; i < ind; i++) {
    typename PGraph::TObj::TNodeI NI = Graph->GetNI(MapV[i]);
    TIntV NbrV;
    MergeNbrs<PGraph>(NbrV, NI);
    TIntV V;
    for (int j = 0; j < NbrV.Len(); j++) {
      TInt Vert = NbrV[j];
      TInt Deg = Graph->GetNI(Vert).GetDeg();
      if (Deg > NI.GetDeg() ||
         (Deg == NI.GetDeg() && Vert > NI.GetId())) {
        V.Add(Vert);
      }
    }
    HigherDegNbrV[i] = V;
  }
  int64 cnt = 0;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+:cnt)
#endif
  for (int i = 0; i < HigherDegNbrV.Len(); i++) {
    for (int j = 0; j < HigherDegNbrV[i].Len(); j++) {
      TInt NbrInd = H.GetDat(HigherDegNbrV[i][j]);
      int64 num = GetCommon(HigherDegNbrV[i], HigherDegNbrV[NbrInd]);
      cnt += num;
    }
  }
  return cnt;
}
#endif
template<class PGraph>
int64 GetTriangleCnt(const PGraph& Graph) {
  const int NNodes = Graph->GetNodes();
  TIntV MapV(NNodes);
  TVec<typename PGraph::TObj::TNodeI> NV(NNodes);
  NV.Reduce(0);
  int MxId = -1;
  int ind = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++)   {
    NV.Add(NI);
    int Id = NI.GetId();
    if (Id > MxId) {
      MxId = Id;
    }
    MapV[ind] = Id;
    ind++;
  }
  TIntV IndV(MxId+1);
  for (int j = 0; j < NNodes; j++) {
    IndV[MapV[j]] = j;
  }
  ind = MapV.Len();
  TVec<TIntV> HigherDegNbrV(ind);
  for (int i = 0; i < ind; i++) {
    HigherDegNbrV[i] = TVec<TInt>();
    HigherDegNbrV[i].Reserve(NV[i].GetDeg());
    HigherDegNbrV[i].Reduce(0);
  }
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 0; i < ind; i++) {
    typename PGraph::TObj::TNodeI NI = NV[i];
    MergeNbrs<PGraph>(HigherDegNbrV[i], NI);
    int k = 0;
    for (int j = 0; j < HigherDegNbrV[i].Len(); j++) {
      TInt Vert = HigherDegNbrV[i][j];
      TInt Deg = NV[IndV[Vert]].GetDeg();
      if (Deg > NI.GetDeg() ||
         (Deg == NI.GetDeg() && Vert > NI.GetId())) {
        HigherDegNbrV[i][k] = Vert;
        k++;
      }
    }
    HigherDegNbrV[i].Reduce(k);
  }
  int64 cnt = 0;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+:cnt)
#endif
  for (int i = 0; i < HigherDegNbrV.Len(); i++) {
    for (int j = 0; j < HigherDegNbrV[i].Len(); j++) {
      TInt NbrInd = IndV[HigherDegNbrV[i][j]];
      int64 num = GetCommon(HigherDegNbrV[i], HigherDegNbrV[NbrInd]);
      cnt += num;
    }
  }
  return cnt;
}
template<class PGraph>
void MergeNbrs(TIntV& NeighbourV, const typename PGraph::TObj::TNodeI& NI) {
  int j = 0;
  int k = 0;
  int prev = -1;
  int indeg = NI.GetInDeg();
  int outdeg = NI.GetOutDeg();
  if (indeg > 0  &&  outdeg > 0) {
    int v1 = NI.GetInNId(j);
    int v2 = NI.GetOutNId(k);
    while (1) {
      if (v1 <= v2) {
        if (prev != v1) {
          NeighbourV.Add(v1);
          prev = v1;
        }
        j += 1;
        if (j >= indeg) {
          break;
        }
        v1 = NI.GetInNId(j);
      } else {
        if (prev != v2) {
          NeighbourV.Add(v2);
          prev = v2;
        }
        k += 1;
        if (k >= outdeg) {
          break;
        }
        v2 = NI.GetOutNId(k);
      }
    }
  }
  while (j < indeg) {
    int v = NI.GetInNId(j);
    if (prev != v) {
      NeighbourV.Add(v);
      prev = v;
    }
    j += 1;
  }
  while (k < outdeg) {
    int v = NI.GetOutNId(k);
    if (prev != v) {
      NeighbourV.Add(v);
      prev = v;
    }
    k += 1;
  }
}
template <class PGraph>
int GetTriadEdges(const PGraph& Graph, int SampleEdges) {
  const bool IsDir = Graph->HasFlag(gfDirected);
  TIntSet NbrH;
  int TriadEdges = 0;
  for(typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    NbrH.Clr(false);
    for (int e = 0; e < NI.GetOutDeg(); e++) {
      if (NI.GetOutNId(e) != NI.GetId()) {
        NbrH.AddKey(NI.GetOutNId(e)); }
    }
    if (IsDir) {
      for (int e = 0; e < NI.GetInDeg(); e++) {
        if (NI.GetInNId(e) != NI.GetId()) {
          NbrH.AddKey(NI.GetInNId(e)); }
      }
    }
    for (int e = 0; e < NI.GetOutDeg(); e++) {
      if (!IsDir && NI.GetId()<NI.GetOutNId(e)) { continue; }
      const typename PGraph::TObj::TNodeI SrcNode = Graph->GetNI(NI.GetOutNId(e));
      bool Triad=false;
      for (int e1 = 0; e1 < SrcNode.GetOutDeg(); e1++) {
        if (NbrH.IsKey(SrcNode.GetOutNId(e1))) { Triad=true; break; }
      }
      if (IsDir && ! Triad) {
        for (int e1 = 0; e1 < SrcNode.GetInDeg(); e1++) {
          if (NbrH.IsKey(SrcNode.GetInNId(e1))) { Triad=true; break; }
        }
      }
      if (Triad) { TriadEdges++; }
    }
  }
  return TriadEdges;
}
template <class PGraph>
int GetNodeTriads(const PGraph& Graph, const int& NId) {
  int ClosedTriads=0, OpenTriads=0;
  return GetNodeTriads(Graph, NId, ClosedTriads, OpenTriads);
}
template <class PGraph>
int GetNodeTriads(const PGraph& Graph, const int& NId, int& ClosedTriads, int& OpenTriads) {
  const typename PGraph::TObj::TNodeI NI = Graph->GetNI(NId);
  ClosedTriads=0;  OpenTriads=0;
  if (NI.GetDeg() < 2) { return 0; }

  TIntSet NbrSet(NI.GetDeg());
  for (int e = 0; e < NI.GetOutDeg(); e++) {
    if (NI.GetOutNId(e) != NI.GetId()) {
      NbrSet.AddKey(NI.GetOutNId(e)); }
  }
  if (Graph->HasFlag(gfDirected)) {
    for (int e = 0; e < NI.GetInDeg(); e++) {
      if (NI.GetInNId(e) != NI.GetId()) {
        NbrSet.AddKey(NI.GetInNId(e)); }
    }
  }

  for (int srcNbr = 0; srcNbr < NbrSet.Len(); srcNbr++) {
    const typename PGraph::TObj::TNodeI SrcNode = Graph->GetNI(NbrSet.GetKey(srcNbr));
    for (int dstNbr = srcNbr+1; dstNbr < NbrSet.Len(); dstNbr++) {
      const int dstNId = NbrSet.GetKey(dstNbr);
      if (SrcNode.IsNbrNId(dstNId)) { ClosedTriads++; }
      else { OpenTriads++; }
    }
  }
  return ClosedTriads;
}
template <class PGraph>
int GetNodeTriadsAll(const PGraph& Graph, const int& NId, int& ClosedTriads, int& OpenTriads) {
  return GetNodeTriads(Graph, NId, ClosedTriads, OpenTriads);
}
template <class PGraph>
int GetNodeTriads(const PGraph& Graph, const int& NId, const TIntSet& GroupSet, int& InGroupEdges, int& InOutGroupEdges, int& OutGroupEdges) {
  const typename PGraph::TObj::TNodeI NI = Graph->GetNI(NId);
  const bool IsDir = Graph->HasFlag(gfDirected);
  InGroupEdges=0;  InOutGroupEdges=0;  OutGroupEdges=0;
  if (NI.GetDeg() < 2) { return 0; }

  TIntSet NbrSet(NI.GetDeg());
  for (int e = 0; e < NI.GetOutDeg(); e++) {
    if (NI.GetOutNId(e) != NI.GetId()) {
      NbrSet.AddKey(NI.GetOutNId(e)); }
  }
  if (IsDir) {
    for (int e = 0; e < NI.GetInDeg(); e++) {
      if (NI.GetInNId(e) != NI.GetId()) {
        NbrSet.AddKey(NI.GetInNId(e)); }
    }
  }

  for (int srcNbr = 0; srcNbr < NbrSet.Len(); srcNbr++) {
    const int NbrId = NbrSet.GetKey(srcNbr);
    const bool NbrIn = GroupSet.IsKey(NbrId);
    const typename PGraph::TObj::TNodeI SrcNode = Graph->GetNI(NbrId);
    for (int dstNbr = srcNbr+1; dstNbr < NbrSet.Len(); dstNbr++) {
      const int DstNId = NbrSet.GetKey(dstNbr);
      if (SrcNode.IsNbrNId(DstNId)) {
        bool DstIn = GroupSet.IsKey(DstNId);
        if (NbrIn && DstIn) { InGroupEdges++; }
        else if (NbrIn || DstIn) { InOutGroupEdges++; }
        else { OutGroupEdges++; }
      }
    }
  }
  return InGroupEdges;
}
template <class PGraph>
void GetTriadParticip(const PGraph& Graph, TIntPrV& TriadCntV) {
  TIntH TriadCntH;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    const int Triads = GetNodeTriads(Graph, NI.GetId());
    TriadCntH.AddDat(Triads) += 1;
  }
  TriadCntH.GetKeyDatPrV(TriadCntV);
  TriadCntV.Sort();
}
template<class PGraph>
int GetCmnNbrs(const PGraph& Graph, const int& NId1, const int& NId2) {
  TIntV NbrV;
  return GetCmnNbrs(Graph, NId1, NId2, NbrV);
}
template<class PGraph>
int GetCmnNbrs(const PGraph& Graph, const int& NId1, const int& NId2, TIntV& NbrV) {
  if (! Graph->IsNode(NId1) || ! Graph->IsNode(NId2)) { NbrV.Clr(false); return 0; }
  typename PGraph::TObj::TNodeI NI1 = Graph->GetNI(NId1);
  typename PGraph::TObj::TNodeI NI2 = Graph->GetNI(NId2);
  NbrV.Clr(false);
  NbrV.Reserve(TMath::Mn(NI1.GetDeg(), NI2.GetDeg()));
  TIntSet NSet1(NI1.GetDeg()), NSet2(NI2.GetDeg());
  for (int i = 0; i < NI1.GetDeg(); i++) {
    const int nid = NI1.GetNbrNId(i);
    if (nid!=NId1 && nid!=NId2) {
      NSet1.AddKey(nid); }
  }
  for (int i = 0; i < NI2.GetDeg(); i++) {
    const int nid = NI2.GetNbrNId(i);
    if (NSet1.IsKey(nid)) {
      NSet2.AddKey(nid);
    }
  }
  NSet2.GetKeyV(NbrV);
  return NbrV.Len();
}
template<>
inline int GetCmnNbrs<PUNGraph>(const PUNGraph& Graph, const int& NId1, const int& NId2, TIntV& NbrV) {
  if (! Graph->IsNode(NId1) || ! Graph->IsNode(NId2)) { NbrV.Clr(false); return 0; }
  const TUNGraph::TNodeI NI1 = Graph->GetNI(NId1);
  const TUNGraph::TNodeI NI2 = Graph->GetNI(NId2);
  int i=0, j=0;
  NbrV.Clr(false);
  NbrV.Reserve(TMath::Mn(NI1.GetDeg(), NI2.GetDeg()));
  while (i < NI1.GetDeg() && j < NI2.GetDeg()) {
    const int nid = NI1.GetNbrNId(i);
    while (j < NI2.GetDeg() && NI2.GetNbrNId(j) < nid) { j++; }
    if (j < NI2.GetDeg() && nid==NI2.GetNbrNId(j) && nid!=NId1 && nid!=NId2) {
      IAssert(NbrV.Empty() || NbrV.Last() < nid);
      NbrV.Add(nid);
      j++;
    }
    i++;
  }
  return NbrV.Len();
}
template<class PGraph>
int GetLen2Paths(const PGraph& Graph, const int& NId1, const int& NId2) {
  TIntV NbrV;
  return GetLen2Paths(Graph, NId1, NId2, NbrV);
}
template<class PGraph>
int GetLen2Paths(const PGraph& Graph, const int& NId1, const int& NId2, TIntV& NbrV) {
  const typename PGraph::TObj::TNodeI NI = Graph->GetNI(NId1);
  NbrV.Clr(false);
  NbrV.Reserve(NI.GetOutDeg());
  for (int e = 0; e < NI.GetOutDeg(); e++) {
    const typename PGraph::TObj::TNodeI MidNI = Graph->GetNI(NI.GetOutNId(e));
    if (MidNI.IsOutNId(NId2)) {
      NbrV.Add(MidNI.GetId());
    }
  }
  return NbrV.Len();
}
template <class PGraph>
void GetUniqueNbrV(const PGraph& Graph, const int& NId, TIntV& NbrV) {
  typename PGraph::TObj::TNodeI NI = Graph->GetNI(NId);
  NbrV.Reserve(NI.GetDeg());
  NbrV.Reduce(0);
  int j = 0;
  int k = 0;
  int Prev = -1;
  int InDeg = NI.GetInDeg();
  int OutDeg = NI.GetOutDeg();
  if (InDeg > 0  &&  OutDeg > 0) {
    int v1 = NI.GetInNId(j);
    int v2 = NI.GetOutNId(k);
    while (1) {
      if (v1 <= v2) {
        if (Prev != v1) {
          if (v1 != NId) {
            NbrV.Add(v1);
            Prev = v1;
          }
        }
        j += 1;
        if (j >= InDeg) {
          break;
        }
        v1 = NI.GetInNId(j);
      } else {
        if (Prev != v2) {
          if (v2 != NId) {
            NbrV.Add(v2);
          }
          Prev = v2;
        }
        k += 1;
        if (k >= OutDeg) {
          break;
        }
        v2 = NI.GetOutNId(k);
      }
    }
  }
  while (j < InDeg) {
    int v = NI.GetInNId(j);
    if (Prev != v) {
      if (v != NId) {
        NbrV.Add(v);
      }
      Prev = v;
    }
    j += 1;
  }
  while (k < OutDeg) {
    int v = NI.GetOutNId(k);
    if (Prev != v) {
      if (v != NId) {
        NbrV.Add(v);
      }
      Prev = v;
    }
    k += 1;
  }
}
};
template <class PGraph>
class TNetConstraint {
public:
  PGraph Graph;
  THash<TIntPr, TFlt> NodePrCH;
public:
  TNetConstraint(const PGraph& GraphPt, const bool& CalcaAll=true);
  int Len() const { return NodePrCH.Len(); }
  double GetC(const int& ConstraintN) const { return NodePrCH[ConstraintN]; }
  TIntPr GetNodePr(const int& ConstraintN) const { return NodePrCH.GetKey(ConstraintN); }
  double GetEdgeC(const int& NId1, const int& NId2) const;
  double GetNodeC(const int& NId) const;
  void AddConstraint(const int& NId1, const int& NId2);
  void CalcConstraints();
  void CalcConstraints(const int& NId);
  void Dump() const;
  static void Test();
};
template <class PGraph>
TNetConstraint<PGraph>::TNetConstraint(const PGraph& GraphPt, const bool& CalcaAll) : Graph(GraphPt) {
  CAssert(! HasGraphFlag(typename PGraph::TObj, gfMultiGraph));
  if (CalcaAll) {
    CalcConstraints();
  }
}
template <class PGraph>
double TNetConstraint<PGraph>::GetEdgeC(const int& NId1, const int& NId2) const {
  if (NodePrCH.IsKey(TIntPr(NId1, NId2))) {
    return NodePrCH.GetDat(TIntPr(NId1, NId2)); }
  else {
    return 0.0; }
}
template <class PGraph>
double TNetConstraint<PGraph>::GetNodeC(const int& NId) const {
  typename PGraph::TObj::TNodeI NI1 = Graph->GetNI(NId);
  if (NI1.GetOutDeg() == 0) { return 0.0; }
  int KeyId = -1;
  for (int k = 0; k<NI1.GetOutDeg(); k++) {
    KeyId = NodePrCH.GetKeyId(TIntPr(NI1.GetId(), NI1.GetOutNId(k)));
    if (KeyId > -1) { break; }
  }
  if (KeyId < 0) { return 0.0; }
  double Constraint = NodePrCH[KeyId];
  for (int i = KeyId-1; i >-1 && NodePrCH.GetKey(i).Val1()==NId; i--) {
    Constraint += NodePrCH[i];
  }
  for (int i = KeyId+1; i < NodePrCH.Len() && NodePrCH.GetKey(i).Val1()==NId; i++) {
    Constraint += NodePrCH[i];
  }
  return Constraint;
}
template <class PGraph>
void TNetConstraint<PGraph>::AddConstraint(const int& NId1, const int& NId2) {
  if (NId1==NId2 || NodePrCH.IsKey(TIntPr(NId1, NId2))) {
    return;
  }
  typename PGraph::TObj::TNodeI NI1 = Graph->GetNI(NId1);
  double Constraint = 0.0;
  if (NI1.IsOutNId(NId2)) {
    Constraint += 1.0/(double) NI1.GetOutDeg();
  }
  const double SrcC = 1.0/(double) NI1.GetOutDeg();
  for (int e = 0; e < NI1.GetOutDeg(); e++) {
    const int MidNId = NI1.GetOutNId(e);
    if (MidNId == NId1 || MidNId == NId2) { continue; }
    const typename PGraph::TObj::TNodeI MidNI = Graph->GetNI(MidNId);
    if (MidNI.IsOutNId(NId2)) {
      Constraint += SrcC * (1.0/(double)MidNI.GetOutDeg());
    }
  }
  if (Constraint==0) { return; }
  Constraint = TMath::Sqr(Constraint);
  NodePrCH.AddDat(TIntPr(NId1, NId2), Constraint);
}
template <class PGraph>
void TNetConstraint<PGraph>::CalcConstraints() {

  for (typename PGraph::TObj::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
    AddConstraint(EI.GetSrcNId(), EI.GetDstNId());
    AddConstraint(EI.GetDstNId(), EI.GetSrcNId());
  }

  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    for (int i = 0; i < NI.GetDeg();  i++) {
      const int NId1 = NI.GetNbrNId(i);
      for (int j = 0; j < NI.GetDeg();  j++) {
        const int NId2 = NI.GetNbrNId(j);
        AddConstraint(NId1, NId2);
      }
    }
  }
  NodePrCH.SortByKey();
}
template <class PGraph>
void TNetConstraint<PGraph>::CalcConstraints(const int& NId) {
  typename PGraph::TObj::TNodeI StartNI = Graph->GetNI(NId);
  TIntSet SeenSet;
  for (int e = 0; e < StartNI.GetOutDeg(); e++) {
    typename PGraph::TObj::TNodeI MidNI = Graph->GetNI(StartNI.GetOutNId(e));
    AddConstraint(NId, MidNI.GetId());
    for (int i = 0; i < MidNI.GetOutDeg();  i++) {
      const int EndNId = MidNI.GetOutNId(i);
      if (! SeenSet.IsKey(EndNId)) {
        AddConstraint(NId, EndNId);
        SeenSet.AddKey(EndNId);
      }
    }
  }
}
template <class PGraph>
void TNetConstraint<PGraph>::Dump() const {
  printf("Edge network constraint: (%d, %d)\n", Graph->GetNodes(), Graph->GetEdges());
  for (int e = 0; e < NodePrCH.Len(); e++) {
    printf("  %4d %4d  :  %f\n", NodePrCH.GetKey(e).Val1(), NodePrCH.GetKey(e).Val2(), NodePrCH[e].Val);
  }
  printf("\n");
}
template <class PGraph>
void TNetConstraint<PGraph>::Test() {
  PUNGraph G = TUNGraph::New();
  G->AddNode(0); G->AddNode(1); G->AddNode(2); G->AddNode(3);
  G->AddNode(4); G->AddNode(5); G->AddNode(6);
  G->AddEdge(0,1); G->AddEdge(0,2); G->AddEdge(0,3); G->AddEdge(0,4); G->AddEdge(0,5); G->AddEdge(0,6);
  G->AddEdge(1,2); G->AddEdge(1,5);  G->AddEdge(1,6);
  G->AddEdge(2,4);
  TNetConstraint<PUNGraph> NetConstraint(G, true);

  NetConstraint.Dump();
  printf("middle node network constraint: %f\n", NetConstraint.GetNodeC(0));
}
#endif
