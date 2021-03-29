namespace TSnap {





template <class PGraph> int CntInDegNodes(const PGraph& Graph, const int& NodeInDeg);

template <class PGraph> int CntOutDegNodes(const PGraph& Graph, const int& NodeOutDeg);

template <class PGraph> int CntDegNodes(const PGraph& Graph, const int& NodeDeg);

template <class PGraph> int CntNonZNodes(const PGraph& Graph);

template <class PGraph> int CntEdgesToSet(const PGraph& Graph, const int& NId, const TIntSet& NodeSet);


template <class PGraph> int GetMxDegNId(const PGraph& Graph);

template <class PGraph> int GetMxInDegNId(const PGraph& Graph);

template <class PGraph> int GetMxOutDegNId(const PGraph& Graph);



template <class PGraph> void GetInDegCnt(const PGraph& Graph, TIntPrV& DegToCntV);

template <class PGraph> void GetInDegCnt(const PGraph& Graph, TFltPrV& DegToCntV);

template <class PGraph> void GetOutDegCnt(const PGraph& Graph, TIntPrV& DegToCntV);

template <class PGraph> void GetOutDegCnt(const PGraph& Graph, TFltPrV& DegToCntV);

template <class PGraph> void GetDegCnt(const PGraph& Graph, TIntPrV& DegToCntV);

template <class PGraph> void GetDegCnt(const PGraph& Graph, TFltPrV& DegToCntV);

template <class PGraph> void GetDegSeqV(const PGraph& Graph, TIntV& DegV);

template <class PGraph> void GetDegSeqV(const PGraph& Graph, TIntV& InDegV, TIntV& OutDegV);


template <class PGraph> void GetNodeInDegV(const PGraph& Graph, TIntPrV& NIdInDegV);

template <class PGraph> void GetNodeOutDegV(const PGraph& Graph, TIntPrV& NIdOutDegV);


template <class PGraph> int CntUniqUndirEdges(const PGraph& Graph);

template <class PGraph> int CntUniqDirEdges(const PGraph& Graph);

template <class PGraph> int CntUniqBiDirEdges(const PGraph& Graph);

template <class PGraph> int CntSelfEdges(const PGraph& Graph);




template <class PGraph> PGraph GetUnDir(const PGraph& Graph);

template <class PGraph> void MakeUnDir(const PGraph& Graph);

template <class PGraph> void AddSelfEdges(const PGraph& Graph);

template <class PGraph> void DelSelfEdges(const PGraph& Graph);





template <class PGraph> void DelNodes(PGraph& Graph, const TIntV& NIdV);

template <class PGraph> void DelZeroDegNodes(PGraph& Graph);

template <class PGraph> void DelDegKNodes(PGraph& Graph, const int& OutDegK, const int& InDegK);



template <class PGraph> bool IsTree(const PGraph& Graph, int& RootNIdX);
template <class PGraph> int GetTreeRootNId(const PGraph& Graph) { int RootNId; bool Tree; Tree = IsTree(Graph, RootNId);  Assert(Tree);  return RootNId; }
template <class PGraph> void GetTreeSig(const PGraph& Graph, const int& RootNId, TIntV& Sig);
template <class PGraph> void GetTreeSig(const PGraph& Graph, const int& RootNId, TIntV& Sig, TIntPrV& NodeMap);



template <class PGraph>
int CntInDegNodes(const PGraph& Graph, const int& NodeInDeg) {
  int Cnt = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (NI.GetInDeg() == NodeInDeg) Cnt++;
  }
  return Cnt;
}

template <class PGraph>
int CntOutDegNodes(const PGraph& Graph, const int& NodeOutDeg) {
  int Cnt = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (NI.GetOutDeg() == NodeOutDeg) Cnt++;
  }
  return Cnt;
}

template <class PGraph>
int CntDegNodes(const PGraph& Graph, const int& NodeDeg) {
  int Cnt = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (NI.GetDeg() == NodeDeg) Cnt++;
  }
  return Cnt;
}

template <class PGraph>
int CntNonZNodes(const PGraph& Graph) {
  int Cnt = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (NI.GetDeg() > 0) Cnt++;
  }
  return Cnt;
}

template <class PGraph>
int CntEdgesToSet(const PGraph& Graph, const int& NId, const TIntSet& NodeSet) {
  if (! Graph->IsNode(NId)) { return 0; }
  const bool IsDir = Graph->HasFlag(gfDirected);
  const typename PGraph::TObj::TNodeI NI = Graph->GetNI(NId);
  if (! IsDir) {
    int EdgesToSet = 0;
    for (int e = 0; e < NI.GetOutDeg(); e++) {
      if (NodeSet.IsKey(NI.GetOutNId(e))) { EdgesToSet++; } }
    return EdgesToSet;
  } else {
    TIntSet Set(NI.GetDeg());
    for (int e = 0; e < NI.GetOutDeg(); e++) {
      if (NodeSet.IsKey(NI.GetOutNId(e))) { Set.AddKey(NI.GetOutNId(e)); } }
    for (int e = 0; e < NI.GetInDeg(); e++) {
      if (NodeSet.IsKey(NI.GetInNId(e))) { Set.AddKey(NI.GetInNId(e)); } }
    return Set.Len();
  }
}

template <class PGraph>
int GetMxDegNId(const PGraph& Graph) {
  TIntV MxDegV;
  int MxDeg=-1;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (MxDeg < NI.GetDeg()) { MxDegV.Clr(); MxDeg = NI.GetDeg(); }
    if (MxDeg == NI.GetDeg()) { MxDegV.Add(NI.GetId()); }
  }
  EAssertR(! MxDegV.Empty(), "Input graph is empty!");
  return MxDegV[TInt::Rnd.GetUniDevInt(MxDegV.Len())];
}

template <class PGraph>
int GetMxInDegNId(const PGraph& Graph) {
  TIntV MxDegV;
  int MxDeg=-1;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (MxDeg < NI.GetInDeg()) { MxDegV.Clr(); MxDeg = NI.GetInDeg(); }
    if (MxDeg == NI.GetInDeg()) { MxDegV.Add(NI.GetId()); }
  }
  EAssertR(! MxDegV.Empty(), "Input graph is empty!");
  return MxDegV[TInt::Rnd.GetUniDevInt(MxDegV.Len())];
}

template <class PGraph>
int GetMxOutDegNId(const PGraph& Graph) {
  TIntV MxDegV;
  int MxDeg=-1;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (MxDeg < NI.GetOutDeg()) { MxDegV.Clr(); MxDeg = NI.GetOutDeg(); }
    if (MxDeg == NI.GetOutDeg()) { MxDegV.Add(NI.GetId()); }
  }
  EAssertR(! MxDegV.Empty(), "Input graph is empty!");
  return MxDegV[TInt::Rnd.GetUniDevInt(MxDegV.Len())];
}

template <class PGraph>
void GetInDegCnt(const PGraph& Graph, TIntPrV& DegToCntV) {
  TIntH DegToCntH;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    DegToCntH.AddDat(NI.GetInDeg())++; }
  DegToCntV.Gen(DegToCntH.Len(), 0);
  for (int i = 0; i < DegToCntH.Len(); i++) {
    DegToCntV.Add(TIntPr(DegToCntH.GetKey(i), DegToCntH[i])); }
  DegToCntV.Sort();
}

template <class PGraph>
void GetInDegCnt(const PGraph& Graph, TFltPrV& DegToCntV) {
  TIntH DegToCntH;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    DegToCntH.AddDat(NI.GetInDeg())++; }
  DegToCntV.Gen(DegToCntH.Len(), 0);
  for (int i = 0; i < DegToCntH.Len(); i++) {
    DegToCntV.Add(TFltPr(DegToCntH.GetKey(i).Val, DegToCntH[i].Val)); }
  DegToCntV.Sort();
}

template <class PGraph>
void GetOutDegCnt(const PGraph& Graph, TIntPrV& DegToCntV) {
  TIntH DegToCntH;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    DegToCntH.AddDat(NI.GetOutDeg())++; }
  DegToCntV.Gen(DegToCntH.Len(), 0);
  for (int i = 0; i < DegToCntH.Len(); i++) {
    DegToCntV.Add(TIntPr(DegToCntH.GetKey(i), DegToCntH[i])); }
  DegToCntV.Sort();
}

template <class PGraph>
void GetOutDegCnt(const PGraph& Graph, TFltPrV& DegToCntV) {
  TIntH DegToCntH;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    DegToCntH.AddDat(NI.GetOutDeg())++; }
  DegToCntV.Gen(DegToCntH.Len(), 0);
  for (int i = 0; i < DegToCntH.Len(); i++) {
    DegToCntV.Add(TFltPr(DegToCntH.GetKey(i).Val, DegToCntH[i].Val)); }
  DegToCntV.Sort();
}

template <class PGraph>
void GetDegCnt(const PGraph& Graph, TIntPrV& DegToCntV) {
  TIntH DegToCntH;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    DegToCntH.AddDat(NI.GetDeg())++; }
  DegToCntV.Gen(DegToCntH.Len(), 0);
  for (int i = 0; i < DegToCntH.Len(); i++) {
    DegToCntV.Add(TIntPr(DegToCntH.GetKey(i), DegToCntH[i])); }
  DegToCntV.Sort();
}

template <class PGraph>
void GetDegCnt(const PGraph& Graph, TFltPrV& DegToCntV) {
  TIntH DegToCntH;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    DegToCntH.AddDat(NI.GetDeg())++; }
  DegToCntV.Gen(DegToCntH.Len(), 0);
  for (int i = 0; i < DegToCntH.Len(); i++) {
    DegToCntV.Add(TFltPr(DegToCntH.GetKey(i).Val, DegToCntH[i].Val)); }
  DegToCntV.Sort();
}

template <class PGraph>
void GetDegSeqV(const PGraph& Graph, TIntV& DegV) {
  DegV.Gen(Graph->GetNodes(), 0);
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    DegV.Add(NI.GetDeg());
  }
}

template <class PGraph>
void GetDegSeqV(const PGraph& Graph, TIntV& InDegV, TIntV& OutDegV) {
  InDegV.Gen(Graph->GetNodes(), 0);
  OutDegV.Gen(Graph->GetNodes(), 0);
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    InDegV.Add(NI.GetInDeg());
    OutDegV.Add(NI.GetOutDeg());
  }
}

template <class PGraph>
void GetNodeInDegV(const PGraph& Graph, TIntPrV& NIdInDegV) {
  NIdInDegV.Reserve(Graph->GetNodes(), 0);
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    NIdInDegV.Add(TIntPr(NI.GetId(), NI.GetInDeg()));
  }
}

template <class PGraph>
void GetNodeOutDegV(const PGraph& Graph, TIntPrV& NIdOutDegV) {
  NIdOutDegV.Reserve(Graph->GetNodes(), 0);
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    NIdOutDegV.Add(TIntPr(NI.GetId(), NI.GetOutDeg()));
  }
}

template <class PGraph>
int CntUniqUndirEdges(const PGraph& Graph) {
  TIntSet NbrSet;
  TIntSet SelfNbrSet;
  int Cnt = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    NbrSet.Clr(false);
    for (int e = 0; e < NI.GetDeg(); e++) {
      const int NbrId = NI.GetNbrNId(e);
      if (NbrId == NI.GetId()) {
        SelfNbrSet.AddKey(NbrId);
      } else {
        NbrSet.AddKey(NbrId);
      }
    }
    Cnt += NbrSet.Len();
  }


  return Cnt / 2;
}

template <class PGraph>
int CntUniqDirEdges(const PGraph& Graph) {
  TIntSet NbrSet;
  int Cnt = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    NbrSet.Clr(false);
    for (int e = 0; e < NI.GetOutDeg(); e++) {
      if (NI.GetOutNId(e) != NI.GetId()) {
        NbrSet.AddKey(NI.GetOutNId(e)); }
    }
    Cnt += NbrSet.Len();
  }
  return Cnt;
}

template <class PGraph>
int CntUniqBiDirEdges(const PGraph& Graph) {
  if (! Graph->HasFlag(gfDirected)) {
    return CntUniqUndirEdges(Graph);
  }
  TIntSet NbrSet;
  int Cnt = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    const int SrcId = NI.GetId();
    for (int e = 0; e < NI.GetOutDeg(); e++) {
      const int DstId = NI.GetOutNId(e);
      if (DstId <= SrcId) { continue; }
      if (Graph->IsEdge(DstId, SrcId)) { Cnt++; }
    }
  }
  return Cnt;
}

template <class PGraph>
int CntSelfEdges(const PGraph& Graph) {
  int Cnt = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    for (int e = 0; e < NI.GetOutDeg(); e++) {
      if (NI.GetId() == NI.GetOutNId(e)) { Cnt++; }
    }
  }
  return Cnt;
}

template <class PGraph>
PGraph GetUnDir(const PGraph& Graph) {
  PGraph NewGraphPt = PGraph::New();
  *NewGraphPt = *Graph;
  MakeUnDir(NewGraphPt);
  return NewGraphPt;
}

template <class PGraph>
void MakeUnDir(const PGraph& Graph) {
  CAssert(HasGraphFlag(typename PGraph::TObj, gfDirected));
  TIntPrV EdgeV;
  for (typename PGraph::TObj::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
    const int SrcNId = EI.GetSrcNId();
    const int DstNId = EI.GetDstNId();
    if (! Graph->IsEdge(DstNId, SrcNId)) {
      EdgeV.Add(TIntPr(DstNId, SrcNId));
    }
  }
  for (int i = 0; i < EdgeV.Len(); i++) {
    Graph->AddEdge(EdgeV[i].Val1, EdgeV[i].Val2);
  }
}

template <class PGraph>
void AddSelfEdges(const PGraph& Graph) {
  TIntV EdgeV;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    const int NId = NI.GetId();
    if (! Graph->IsEdge(NId, NId)) {
      EdgeV.Add(NId);
    }
  }
  for (int i = 0; i < EdgeV.Len(); i++) {
    Graph->AddEdge(EdgeV[i], EdgeV[i]);
  }
}

namespace TSnapDetail {

template <class PGraph, bool IsMultiGraph>
struct TDelSelfEdges {
  static void Do(const PGraph& Graph) {
    TIntV EdgeV;

    for (typename PGraph::TObj::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
      if (EI.GetSrcNId() == EI.GetDstNId()) {
        EdgeV.Add(EI.GetSrcNId());
      }
    }
    for (int i = 0; i < EdgeV.Len(); i++) {
      Graph->DelEdge(EdgeV[i], EdgeV[i]);
    }
  }
};

template <class PGraph>
struct TDelSelfEdges<PGraph, true> {
  static void Do(const PGraph& Graph) {
    TIntV EdgeV;
    for (typename PGraph::TObj::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
      if (EI.GetSrcNId() == EI.GetDstNId()) {
        IAssert(EI.GetId() >= 0);
        EdgeV.Add(EI.GetId());
      }
    }
    for (int i = 0; i < EdgeV.Len(); i++) {
      Graph->DelEdge(EdgeV[i]);
    }
  }
};

}

template <class PGraph>
void DelSelfEdges(const PGraph& Graph) {
  TSnapDetail::TDelSelfEdges<PGraph, HasGraphFlag(typename PGraph::TObj, gfMultiGraph)>
    ::Do(Graph);
}

template <class PGraph>
void DelNodes(PGraph& Graph, const TIntV& NIdV) {
  for (int n = 0; n < NIdV.Len(); n++) {
    Graph->DelNode(NIdV[n]);
  }
}

template <class PGraph>
void DelZeroDegNodes(PGraph& Graph) {
  TIntV DelNIdV;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (NI.GetDeg() == 0) {
      DelNIdV.Add(NI.GetId());
    }
  }
  for (int i = 0; i < DelNIdV.Len(); i++) {
    Graph->DelNode(DelNIdV[i]);
  }
}

template <class PGraph>
void DelDegKNodes(PGraph& Graph, const int& OutDegK, const int& InDegK) {
  TIntV DelNIdV;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (NI.GetOutDeg() == OutDegK || NI.GetInDeg() == InDegK) {
      DelNIdV.Add(NI.GetId());
    }
  }
  for (int i = 0; i < DelNIdV.Len(); i++) {
    Graph->DelNode(DelNIdV[i]);
  }
}



template <class PGraph>
bool IsTree(const PGraph& Graph, int& RootNId) {
  if (Graph->GetNodes() == 1 && Graph->GetEdges() == 0) {
    RootNId = Graph->BegNI().GetId();
    return true;
  }
  RootNId = -1;
  if (Graph->GetNodes() != Graph->GetEdges()+1) { return false; }
  int NZeroOutDeg = 0;
  int ZeroOutDegN = -1;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (NI.GetOutDeg() == 0) {
      ZeroOutDegN = NI.GetId();  NZeroOutDeg++;
    }
    if (NI.GetDeg() == 0) { return false; }
  }
  if (NZeroOutDeg==1) {
    if (! TSnap::IsConnected(Graph)) { return false; }
    RootNId = ZeroOutDegN;  return true;
  }
  return false;
}


template <class PGraph>
void GetTreeSig(const PGraph& Graph, const int& RootNId, TIntV& Sig) {
  CAssert(HasGraphFlag(typename PGraph::TObj, gfDirected));
  Sig.Gen(Graph->GetNodes(), 0);
  TSnapQueue<int> NIdQ(Graph->GetNodes());
  NIdQ.Push(RootNId);
  int LastPos = 0, NodeCnt = 1;
  while (! NIdQ.Empty()) {
    const typename PGraph::TObj::TNodeI Node = Graph->GetNI(NIdQ.Top());  NIdQ.Pop();
    IAssert(Node.GetInDeg()==0 || Node.GetOutDeg()==0);
    if (Node.GetInDeg() != 0) {
      for (int e = 0; e < Node.GetInDeg(); e++) {
        NIdQ.Push(Node.GetInNId(e)); }
    } else if (Node.GetOutDeg() != 0) {
      for (int e = 0; e < Node.GetOutDeg(); e++) {
        NIdQ.Push(Node.GetOutNId(e)); }
    }
    Sig.Add(Node.GetInDeg());
    if (--NodeCnt == 0) {
      for (int i = LastPos; i < Sig.Len(); i++) NodeCnt += Sig[i];
      Sig.QSort(LastPos, Sig.Len()-1, false);
      LastPos = Sig.Len();
    }
  }
}


template <class PGraph>
void GetTreeSig(const PGraph& Graph, const int& RootNId, TIntV& Sig, TIntPrV& NodeMap) {
  CAssert(HasGraphFlag(typename PGraph::TObj, gfDirected));
  NodeMap.Gen(Graph->GetNodes(), 0);
  Sig.Gen(Graph->GetNodes(), 0);
  TSnapQueue<int> NIdQ(Graph->GetNodes());
  NIdQ.Push(RootNId);
  int LastPos = 0, NodeCnt = 1;
  while (! NIdQ.Empty()) {
    const typename PGraph::TObj::TNodeI Node = Graph->GetNI(NIdQ.Top());  NIdQ.Pop();
    IAssert(Node.GetInDeg()==0 || Node.GetOutDeg()==0);
    if (Node.GetInDeg() != 0) {
      for (int e = 0; e < Node.GetInDeg(); e++) {
        NIdQ.Push(Node.GetInNId(e)); }
      NodeMap.Add(TIntPr(Node.GetInDeg(), Node.GetId()));
    } else if (Node.GetOutDeg() != 0) {
      for (int e = 0; e < Node.GetOutDeg(); e++) {
        NIdQ.Push(Node.GetOutNId(e)); }
      NodeMap.Add(TIntPr(Node.GetOutDeg(), Node.GetId()));
    }
    if (--NodeCnt == 0) {
      for (int i = LastPos; i < NodeMap.Len(); i++) {
        NodeCnt += NodeMap[i].Val1; }
      NodeMap.QSort(LastPos, NodeMap.Len()-1, false);
      LastPos = NodeMap.Len();
    }
  }
  for (int i = 0; i < NodeMap.Len(); i++) {
    Sig.Add(NodeMap[i].Val1);
    NodeMap[i].Val1 = NodeMap[i].Val2;
    NodeMap[i].Val2 = i;
  }
}

};
