#ifndef NETWORK_H
#define NETWORK_H
template <class TNodeData>
class TNodeNet {
public:
  typedef TNodeData TNodeDat;
  typedef TNodeNet<TNodeData> TNet;
  typedef TPt<TNet> PNet;
public:
  class TNode {
  private:
    TInt Id;
    TNodeData NodeDat;
    TIntV InNIdV, OutNIdV;
  public:
    TNode() : Id(-1), NodeDat(), InNIdV(), OutNIdV() { }
    TNode(const int& NId) : Id(NId), NodeDat(), InNIdV(), OutNIdV() { }
    TNode(const int& NId, const TNodeData& NodeData) : Id(NId), NodeDat(NodeData), InNIdV(), OutNIdV() { }
    TNode(const TNode& Node) : Id(Node.Id), NodeDat(Node.NodeDat), InNIdV(Node.InNIdV), OutNIdV(Node.OutNIdV) { }
    TNode(TSIn& SIn) : Id(SIn), NodeDat(SIn), InNIdV(SIn), OutNIdV(SIn) { }
    void Save(TSOut& SOut) const { Id.Save(SOut);  NodeDat.Save(SOut);  InNIdV.Save(SOut);  OutNIdV.Save(SOut); }
    int GetId() const { return Id; }
    int GetDeg() const { return GetInDeg() + GetOutDeg(); }
    int GetInDeg() const { return InNIdV.Len(); }
    int GetOutDeg() const { return OutNIdV.Len(); }
    const TNodeData& GetDat() const { return NodeDat; }
    TNodeData& GetDat() { return NodeDat; }
    int GetInNId(const int& NodeN) const { return InNIdV[NodeN]; }
    int GetOutNId(const int& NodeN) const { return OutNIdV[NodeN]; }
    int GetNbrNId(const int& NodeN) const { return NodeN<GetOutDeg() ? GetOutNId(NodeN):GetInNId(NodeN-GetOutDeg()); }
    bool IsInNId(const int& NId) const { return InNIdV.SearchBin(NId) != -1; }
    bool IsOutNId(const int& NId) const { return OutNIdV.SearchBin(NId) != -1; }
    bool IsNbrNId(const int& NId) const { return IsOutNId(NId) || IsInNId(NId); }
    void LoadShM(TShMIn& MStream) {
      Id = TInt(MStream);
      NodeDat = TNodeData(MStream);
      InNIdV.LoadShM(MStream);
      OutNIdV.LoadShM(MStream);
    }
    bool operator < (const TNode& Node) const { return NodeDat < Node.NodeDat; }
    friend class TNodeNet<TNodeData>;
  };

  class TNodeI {
  private:
    typedef typename THash<TInt, TNode>::TIter THashIter;
    THashIter NodeHI;
    TNodeNet *Net;
  public:
    TNodeI() : NodeHI(), Net(NULL) { }
    TNodeI(const THashIter& NodeHIter, const TNodeNet* NetPt) : NodeHI(NodeHIter), Net((TNodeNet *) NetPt) { }
    TNodeI(const TNodeI& NodeI) : NodeHI(NodeI.NodeHI), Net(NodeI.Net) { }
    TNodeI& operator = (const TNodeI& NodeI) { NodeHI=NodeI.NodeHI; Net=NodeI.Net; return *this; }

    TNodeI& operator++ (int) { NodeHI++;  return *this; }
    bool operator < (const TNodeI& NodeI) const { return NodeHI < NodeI.NodeHI; }
    bool operator == (const TNodeI& NodeI) const { return NodeHI == NodeI.NodeHI; }

    int GetId() const { return NodeHI.GetDat().GetId(); }

    int GetDeg() const { return NodeHI.GetDat().GetDeg(); }

    int GetInDeg() const { return NodeHI.GetDat().GetInDeg(); }

    int GetOutDeg() const { return NodeHI.GetDat().GetOutDeg(); }

    int GetInNId(const int& NodeN) const { return NodeHI.GetDat().GetInNId(NodeN); }

    int GetOutNId(const int& NodeN) const { return NodeHI.GetDat().GetOutNId(NodeN); }

    int GetNbrNId(const int& NodeN) const { return NodeHI.GetDat().GetNbrNId(NodeN); }

    bool IsInNId(const int& NId) const { return NodeHI.GetDat().IsInNId(NId); }

    bool IsOutNId(const int& NId) const { return NodeHI.GetDat().IsOutNId(NId); }

    bool IsNbrNId(const int& NId) const { return IsOutNId(NId) || IsInNId(NId); }
    const TNodeData& operator () () const { return NodeHI.GetDat().NodeDat; }
    TNodeData& operator () () { return NodeHI.GetDat().GetDat(); }
    const TNodeData& GetDat() const { return NodeHI.GetDat().GetDat(); }
    TNodeData& GetDat() { return NodeHI.GetDat().GetDat(); }
    const TNodeData& GetInNDat(const int& NodeN) const { return Net->GetNDat(GetInNId(NodeN)); }
    TNodeData& GetInNDat(const int& NodeN) { return Net->GetNDat(GetInNId(NodeN)); }
    const TNodeData& GetOutNDat(const int& NodeN) const { return Net->GetNDat(GetOutNId(NodeN)); }
    TNodeData& GetOutNDat(const int& NodeN) { return Net->GetNDat(GetOutNId(NodeN)); }
    const TNodeData& GetNbrNDat(const int& NodeN) const { return Net->GetNDat(GetNbrNId(NodeN)); }
    TNodeData& GetNbrNDat(const int& NodeN) { return Net->GetNDat(GetNbrNId(NodeN)); }
    friend class TNodeNet<TNodeData>;
  };

  class TEdgeI {
  private:
    TNodeI CurNode, EndNode;
    int CurEdge;
  public:
    TEdgeI() : CurNode(), EndNode(), CurEdge(0) { }
    TEdgeI(const TNodeI& NodeI, const TNodeI& EndNodeI, const int& EdgeN=0) : CurNode(NodeI), EndNode(EndNodeI), CurEdge(EdgeN) { }
    TEdgeI(const TEdgeI& EdgeI) : CurNode(EdgeI.CurNode), EndNode(EdgeI.EndNode), CurEdge(EdgeI.CurEdge) { }
    TEdgeI& operator = (const TEdgeI& EdgeI) { if (this!=&EdgeI) { CurNode=EdgeI.CurNode;  EndNode=EdgeI.EndNode;  CurEdge=EdgeI.CurEdge; }  return *this; }

    TEdgeI& operator++ (int) { CurEdge++; if (CurEdge >= CurNode.GetOutDeg()) { CurEdge=0;  CurNode++;
      while (CurNode < EndNode && CurNode.GetOutDeg()==0) { CurNode++; } }  return *this; }
    bool operator < (const TEdgeI& EdgeI) const { return CurNode<EdgeI.CurNode || (CurNode==EdgeI.CurNode && CurEdge<EdgeI.CurEdge); }
    bool operator == (const TEdgeI& EdgeI) const { return CurNode == EdgeI.CurNode && CurEdge == EdgeI.CurEdge; }

    int GetId() const { return -1; }

    int GetSrcNId() const { return CurNode.GetId(); }

    int GetDstNId() const { return CurNode.GetOutNId(CurEdge); }
    const TNodeData& GetSrcNDat() const { return CurNode.GetDat(); }
    TNodeData& GetDstNDat() { return CurNode.GetOutNDat(CurEdge); }
    friend class TNodeNet<TNodeData>;
  };
protected:
  TNode& GetNode(const int& NId) { return NodeH.GetDat(NId); }
protected:
  TCRef CRef;
  TInt MxNId;
  THash<TInt, TNode> NodeH;
private:
  class TNodeFunctor {
  public:
    TNodeFunctor() {}
    void operator() (TNode* n, TShMIn& ShMIn) { n->LoadShM(ShMIn);}
  };
private:
  void LoadNetworkShM(TShMIn& ShMIn) {
    MxNId = TInt(ShMIn);
    TNodeFunctor f;
    NodeH.LoadShM(ShMIn, f);
  }
public:
  TNodeNet() : CRef(), MxNId(0), NodeH() { }

  explicit TNodeNet(const int& Nodes, const int& Edges) : MxNId(0) { Reserve(Nodes, Edges); }
  TNodeNet(const TNodeNet& NodeNet) : MxNId(NodeNet.MxNId), NodeH(NodeNet.NodeH) { }

  TNodeNet(TSIn& SIn) : MxNId(SIn), NodeH(SIn) { }
  virtual ~TNodeNet() { }

  virtual void Save(TSOut& SOut) const { MxNId.Save(SOut);  NodeH.Save(SOut); }

  static PNet New() { return PNet(new TNodeNet()); }

  static PNet Load(TSIn& SIn) { return PNet(new TNodeNet(SIn)); }

  static PNet LoadShM(TShMIn& ShMIn) {
    TNodeNet* Network = new TNodeNet();
    Network->LoadNetworkShM(ShMIn);
    return PNet(Network);
  }

  bool HasFlag(const TGraphFlag& Flag) const;
  TNodeNet& operator = (const TNodeNet& NodeNet) {
    if (this!=&NodeNet) { NodeH=NodeNet.NodeH;  MxNId=NodeNet.MxNId; }  return *this; }


  int GetNodes() const { return NodeH.Len(); }

  int AddNode(int NId = -1);

  int AddNodeUnchecked(int NId = -1);

  int AddNode(int NId, const TNodeData& NodeDat);

  int AddNode(const TNodeI& NodeI) { return AddNode(NodeI.GetId(), NodeI.GetDat()); }

  virtual void DelNode(const int& NId);

  void DelNode(const TNode& NodeI) { DelNode(NodeI.GetId()); }

  bool IsNode(const int& NId) const { return NodeH.IsKey(NId); }

  TNodeI BegNI() const { return TNodeI(NodeH.BegI(), this); }

  TNodeI EndNI() const { return TNodeI(NodeH.EndI(), this); }

  TNodeI GetNI(const int& NId) const { return TNodeI(NodeH.GetI(NId), this); }

  const TNode& GetNode(const int& NId) const { return NodeH.GetDat(NId); }

  void SetNDat(const int& NId, const TNodeData& NodeDat);

  TNodeData& GetNDat(const int& NId) { return NodeH.GetDat(NId).NodeDat; }

  const TNodeData& GetNDat(const int& NId) const { return NodeH.GetDat(NId).NodeDat; }

  int GetMxNId() const { return MxNId; }


  int GetEdges() const;

  int AddEdge(const int& SrcNId, const int& DstNId);

  int AddEdge(const TEdgeI& EdgeI) { return AddEdge(EdgeI.GetSrcNId(), EdgeI.GetDstNId()); }

  void DelEdge(const int& SrcNId, const int& DstNId, const bool& IsDir = true);

  bool IsEdge(const int& SrcNId, const int& DstNId, const bool& IsDir = true) const;

  TEdgeI BegEI() const { TNodeI NI=BegNI();  while(NI<EndNI() && NI.GetOutDeg()==0) NI++;  return TEdgeI(NI, EndNI()); }

  TEdgeI EndEI() const { return TEdgeI(EndNI(), EndNI()); }

  TEdgeI GetEI(const int& EId) const;

  TEdgeI GetEI(const int& SrcNId, const int& DstNId) const;

  int GetRndNId(TRnd& Rnd=TInt::Rnd) { return NodeH.GetKey(NodeH.GetRndKeyId(Rnd, 0.8)); }

  TNodeI GetRndNI(TRnd& Rnd=TInt::Rnd) { return GetNI(GetRndNId(Rnd)); }

  void GetNIdV(TIntV& NIdV) const;

  bool Empty() const { return GetNodes()==0; }

  void Clr(const bool& DoDel=true, const bool& ResetDat=true) {
    MxNId = 0;  NodeH.Clr(DoDel, -1, ResetDat); }

  void Reserve(const int& Nodes, const int& Edges) { if (Nodes>0) { NodeH.Gen(Nodes/2); } }

  void SortNIdById(const bool& Asc=true) { NodeH.SortByKey(Asc); }

  void SortNIdByDat(const bool& Asc=true) { NodeH.SortByDat(Asc); }

  void Defrag(const bool& OnlyNodeLinks=false);

  bool IsOk(const bool& ThrowExcept=true) const;
  friend class TPt<TNodeNet<TNodeData> >;
};
namespace TSnap {
template <class TNodeData> struct IsDirected<TNodeNet<TNodeData> > { enum { Val = 1 }; };
template <class TNodeData> struct IsNodeDat<TNodeNet<TNodeData> > { enum { Val = 1 }; };
}
template <class TNodeData>
bool TNodeNet<TNodeData>::HasFlag(const TGraphFlag& Flag) const {
  return HasGraphFlag(typename TNet, Flag);
}
template <class TNodeData>
int TNodeNet<TNodeData>::AddNode(int NId) {
  if (NId == -1) {
    NId = MxNId;  MxNId++;
  } else {
    IAssertR(!IsNode(NId), TStr::Fmt("NodeId %d already exists", NId));
    MxNId = TMath::Mx(NId+1, MxNId());
  }
  NodeH.AddDat(NId, TNode(NId));
  return NId;
}
template <class TNodeData>
int TNodeNet<TNodeData>::AddNodeUnchecked(int NId) {
  if (NId == -1) {
    NId = MxNId;  MxNId++;
  } else {
    if (IsNode(NId)) { return -1;}
    MxNId = TMath::Mx(NId+1, MxNId());
  }
  NodeH.AddDat(NId, TNode(NId));
  return NId;
}
template <class TNodeData>
int TNodeNet<TNodeData>::AddNode(int NId, const TNodeData& NodeDat) {
  if (NId == -1) {
    NId = MxNId;  MxNId++;
  } else {
    IAssertR(!IsNode(NId), TStr::Fmt("NodeId %d already exists", NId));
    MxNId = TMath::Mx(NId+1, MxNId());
  }
  NodeH.AddDat(NId, TNode(NId, NodeDat));
  return NId;
}
template <class TNodeData>
void TNodeNet<TNodeData>::DelNode(const int& NId) {
  { TNode& Node = GetNode(NId);
  for (int e = 0; e < Node.GetOutDeg(); e++) {
  const int nbr = Node.GetOutNId(e);
  if (nbr == NId) { continue; }
    TNode& N = GetNode(nbr);
    int n = N.InNIdV.SearchBin(NId);
    if (n!= -1) { N.InNIdV.Del(n); }
  }
  for (int e = 0; e < Node.GetInDeg(); e++) {
  const int nbr = Node.GetInNId(e);
  if (nbr == NId) { continue; }
    TNode& N = GetNode(nbr);
    int n = N.OutNIdV.SearchBin(NId);
    if (n!= -1) { N.OutNIdV.Del(n); }
  } }
  NodeH.DelKey(NId);
}
template <class TNodeData>
void TNodeNet<TNodeData>::SetNDat(const int& NId, const TNodeData& NodeDat) {
  IAssertR(IsNode(NId), TStr::Fmt("NodeId %d does not exist.", NId).CStr());
  NodeH.GetDat(NId).NodeDat = NodeDat;
}
template <class TNodeData>
int TNodeNet<TNodeData>::GetEdges() const {
  int edges=0;
  for (int N=NodeH.FFirstKeyId(); NodeH.FNextKeyId(N);) {
    edges+=NodeH[N].GetOutDeg(); }
  return edges;
}
template <class TNodeData>
int TNodeNet<TNodeData>::AddEdge(const int& SrcNId, const int& DstNId) {
  IAssertR(IsNode(SrcNId) && IsNode(DstNId), TStr::Fmt("%d or %d not a node.", SrcNId, DstNId).CStr());
  if (IsEdge(SrcNId, DstNId)) { return -2; }
  GetNode(SrcNId).OutNIdV.AddSorted(DstNId);
  GetNode(DstNId).InNIdV.AddSorted(SrcNId);
  return -1;
}
template <class TNodeData>
void TNodeNet<TNodeData>::DelEdge(const int& SrcNId, const int& DstNId, const bool& IsDir) {
  IAssertR(IsNode(SrcNId) && IsNode(DstNId), TStr::Fmt("%d or %d not a node.", SrcNId, DstNId).CStr());
  GetNode(SrcNId).OutNIdV.DelIfIn(DstNId);
  GetNode(DstNId).InNIdV.DelIfIn(SrcNId);
  if (! IsDir) {
    GetNode(DstNId).OutNIdV.DelIfIn(SrcNId);
    GetNode(SrcNId).InNIdV.DelIfIn(DstNId);
  }
}
template <class TNodeData>
bool TNodeNet<TNodeData>::IsEdge(const int& SrcNId, const int& DstNId, const bool& IsDir) const {
  if (! IsNode(SrcNId) || ! IsNode(DstNId)) { return false; }
  if (IsDir) { return GetNode(SrcNId).IsOutNId(DstNId); }
  else { return GetNode(SrcNId).IsOutNId(DstNId) || GetNode(DstNId).IsOutNId(SrcNId); }
}
template <class TNodeData>
void TNodeNet<TNodeData>::GetNIdV(TIntV& NIdV) const {
  NIdV.Reserve(GetNodes(), 0);
  for (int N=NodeH.FFirstKeyId(); NodeH.FNextKeyId(N); ) {
    NIdV.Add(NodeH.GetKey(N)); }
}
template <class TNodeData>
typename TNodeNet<TNodeData>::TEdgeI  TNodeNet<TNodeData>::GetEI(const int& SrcNId, const int& DstNId) const {
  const TNodeI SrcNI = GetNI(SrcNId);
  const int NodeN = SrcNI.NodeHI.GetDat().OutNIdV.SearchBin(DstNId);
  if (NodeN == -1) { return EndEI(); }
  return TEdgeI(SrcNI, EndNI(), NodeN);
}
template <class TNodeData>
void TNodeNet<TNodeData>::Defrag(const bool& OnlyNodeLinks) {
  for (int n = NodeH.FFirstKeyId(); NodeH.FNextKeyId(n); ) {
    TNode& Node = NodeH[n];
    Node.InNIdV.Pack();  Node.OutNIdV.Pack();
  }
  if (! OnlyNodeLinks && ! NodeH.IsKeyIdEqKeyN()) {
    NodeH.Defrag(); }
}
template <class TNodeData>
bool TNodeNet<TNodeData>::IsOk(const bool& ThrowExcept) const {
  bool RetVal = true;
  for (int N = NodeH.FFirstKeyId(); NodeH.FNextKeyId(N); ) {
    const TNode& Node = NodeH[N];
    if (! Node.OutNIdV.IsSorted()) {
      const TStr Msg = TStr::Fmt("Out-neighbor list of node %d is not sorted.", Node.GetId());
      if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
    }
    if (! Node.InNIdV.IsSorted()) {
      const TStr Msg = TStr::Fmt("In-neighbor list of node %d is not sorted.", Node.GetId());
      if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
    }

    int prevNId = -1;
    for (int e = 0; e < Node.GetOutDeg(); e++) {
      if (! IsNode(Node.GetOutNId(e))) {
        const TStr Msg = TStr::Fmt("Out-edge %d --> %d: node %d does not exist.",
          Node.GetId(), Node.GetOutNId(e), Node.GetOutNId(e));
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      if (e > 0 && prevNId == Node.GetOutNId(e)) {
        const TStr Msg = TStr::Fmt("Node %d has duplidate out-edge %d --> %d.",
          Node.GetId(), Node.GetId(), Node.GetOutNId(e));
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      prevNId = Node.GetOutNId(e);
    }

    prevNId = -1;
    for (int e = 0; e < Node.GetInDeg(); e++) {
      if (! IsNode(Node.GetInNId(e))) {
        const TStr Msg = TStr::Fmt("In-edge %d <-- %d: node %d does not exist.",
          Node.GetId(), Node.GetInNId(e), Node.GetInNId(e));
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      if (e > 0 && prevNId == Node.GetInNId(e)) {
        const TStr Msg = TStr::Fmt("Node %d has duplidate in-edge %d <-- %d.",
          Node.GetId(), Node.GetId(), Node.GetInNId(e));
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      prevNId = Node.GetInNId(e);
    }
  }
  return RetVal;
}
typedef TNodeNet<TInt> TIntNNet;
typedef TPt<TIntNNet> PIntNNet;
typedef TNodeNet<TFlt> TFltNNet;
typedef TPt<TFltNNet> PFltNNet;
typedef TNodeNet<TStr> TStrNNet;
typedef TPt<TStrNNet> PStrNNet;
template <class TNodeData, class TEdgeData>
class TNodeEDatNet {
public:
  typedef TNodeData TNodeDat;
  typedef TEdgeData TEdgeDat;
  typedef TNodeEDatNet<TNodeData, TEdgeData> TNet;
  typedef TPt<TNet> PNet;
  typedef TVec<TPair<TInt, TEdgeData> > TNIdDatPrV;
public:
  class TNode {
  private:
    TInt  Id;
    TNodeData NodeDat;
    TIntV InNIdV;
    TNIdDatPrV OutNIdV;
  public:
    TNode() : Id(-1), NodeDat(), InNIdV(), OutNIdV() { }
    TNode(const int& NId) : Id(NId), NodeDat(), InNIdV(), OutNIdV() { }
    TNode(const int& NId, const TNodeData& NodeData) : Id(NId), NodeDat(NodeData), InNIdV(), OutNIdV() { }
    TNode(const TNode& Node) : Id(Node.Id), NodeDat(Node.NodeDat), InNIdV(Node.InNIdV), OutNIdV(Node.OutNIdV) { }
    TNode(TSIn& SIn) : Id(SIn), NodeDat(SIn), InNIdV(SIn), OutNIdV(SIn) { }
    void Save(TSOut& SOut) const { Id.Save(SOut);  NodeDat.Save(SOut);  InNIdV.Save(SOut);  OutNIdV.Save(SOut); }
    int GetId() const { return Id; }
    int GetDeg() const { return GetInDeg() + GetOutDeg(); }
    int GetInDeg() const { return InNIdV.Len(); }
    int GetOutDeg() const { return OutNIdV.Len(); }
    const TNodeData& GetDat() const { return NodeDat; }
    TNodeData& GetDat() { return NodeDat; }
    int GetInNId(const int& EdgeN) const { return InNIdV[EdgeN]; }
    int GetOutNId(const int& EdgeN) const { return OutNIdV[EdgeN].Val1; }
    int GetNbrNId(const int& EdgeN) const { return EdgeN<GetOutDeg() ? GetOutNId(EdgeN):GetInNId(EdgeN-GetOutDeg()); }
    TEdgeData& GetOutEDat(const int& EdgeN) { return OutNIdV[EdgeN].Val2; }
    const TEdgeData& GetOutEDat(const int& EdgeN) const { return OutNIdV[EdgeN].Val2; }
    bool IsInNId(const int& NId) const { return InNIdV.SearchBin(NId)!=-1; }
    bool IsOutNId(const int& NId) const { return TNodeEDatNet::GetNIdPos(OutNIdV, NId)!=-1; }
    bool IsNbrNId(const int& NId) const { return IsOutNId(NId) || IsInNId(NId); }
    void LoadShM(TShMIn& MStream) {
      Id = TInt(MStream);
      NodeDat = TNodeData(MStream);
      InNIdV.LoadShM(MStream);
      OutNIdV.LoadShM(MStream);
    }
    bool operator < (const TNode& Node) const { return NodeDat < Node.NodeDat; }
    friend class TNodeEDatNet<TNodeData, TEdgeData>;
  };

  class TNodeI {
  private:
    typedef typename THash<TInt, TNode>::TIter THashIter;
    THashIter NodeHI;
    TNodeEDatNet *Net;
  public:
    TNodeI() : NodeHI(), Net(NULL) { }
    TNodeI(const THashIter& NodeHIter, const TNodeEDatNet* NetPt) : NodeHI(NodeHIter), Net((TNodeEDatNet *) NetPt) { }
    TNodeI(const TNodeI& NodeI) : NodeHI(NodeI.NodeHI), Net(NodeI.Net) { }
    TNodeI& operator = (const TNodeI& NodeI) { NodeHI=NodeI.NodeHI; Net=NodeI.Net; return *this; }

    TNodeI& operator++ (int) { NodeHI++;  return *this; }
    bool operator < (const TNodeI& NodeI) const { return NodeHI < NodeI.NodeHI; }
    bool operator == (const TNodeI& NodeI) const { return NodeHI == NodeI.NodeHI; }

    int GetId() const { return NodeHI.GetDat().GetId(); }

    int GetDeg() const { return NodeHI.GetDat().GetDeg(); }

    int GetInDeg() const { return NodeHI.GetDat().GetInDeg(); }

    int GetOutDeg() const { return NodeHI.GetDat().GetOutDeg(); }

    int GetInNId(const int& NodeN) const { return NodeHI.GetDat().GetInNId(NodeN); }

    int GetOutNId(const int& NodeN) const { return NodeHI.GetDat().GetOutNId(NodeN); }

    int GetNbrNId(const int& NodeN) const { return NodeHI.GetDat().GetNbrNId(NodeN); }

    bool IsInNId(const int& NId) const { return NodeHI.GetDat().IsInNId(NId); }

    bool IsOutNId(const int& NId) const { return NodeHI.GetDat().IsOutNId(NId); }

    bool IsNbrNId(const int& NId) const { return IsOutNId(NId) || IsInNId(NId); }

    const TNodeData& operator () () const { return NodeHI.GetDat().NodeDat; }
    TNodeData& operator () () { return NodeHI.GetDat().GetDat(); }
    const TNodeData& GetDat() const { return NodeHI.GetDat().GetDat(); }
    TNodeData& GetDat() { return NodeHI.GetDat().GetDat(); }
    const TNodeData& GetOutNDat(const int& NodeN) const { return Net->GetNDat(GetOutNId(NodeN)); }
    TNodeData& GetOutNDat(const int& NodeN) { return Net->GetNDat(GetOutNId(NodeN)); }
    const TNodeData& GetInNDat(const int& NodeN) const { return Net->GetNDat(GetInNId(NodeN)); }
    TNodeData& GetInNDat(const int& NodeN) { return Net->GetNDat(GetInNId(NodeN)); }
    const TNodeData& GetNbrNDat(const int& NodeN) const { return Net->GetNDat(GetNbrNId(NodeN)); }
    TNodeData& GetNbrNDat(const int& NodeN) { return Net->GetNDat(GetNbrNId(NodeN)); }

    TEdgeData& GetOutEDat(const int& EdgeN) { return NodeHI.GetDat().GetOutEDat(EdgeN); }
    const TEdgeData& GetOutEDat(const int& EdgeN) const { return NodeHI.GetDat().GetOutEDat(EdgeN); }
    TEdgeData& GetInEDat(const int& EdgeN) { return Net->GetEDat(GetInNId(EdgeN), GetId()); }
    const TEdgeData& GetInEDat(const int& EdgeN) const { return Net->GetEDat(GetInNId(EdgeN), GetId()); }
    TEdgeData& GetNbrEDat(const int& EdgeN) { return EdgeN<GetOutDeg() ? GetOutEDat(EdgeN) : GetInEDat(EdgeN-GetOutDeg()); }
    const TEdgeData& GetNbrEDat(const int& EdgeN) const { return EdgeN<GetOutDeg() ? GetOutEDat(EdgeN) : GetInEDat(EdgeN-GetOutDeg()); }
    friend class TNodeEDatNet<TNodeData, TEdgeData>;
  };

  class TEdgeI {
  private:
    TNodeI CurNode, EndNode;
    int CurEdge;
  public:
    TEdgeI() : CurNode(), EndNode(), CurEdge(0) { }
    TEdgeI(const TNodeI& NodeI, const TNodeI& EndNodeI, const int& EdgeN=0) : CurNode(NodeI), EndNode(EndNodeI), CurEdge(EdgeN) { }
    TEdgeI(const TEdgeI& EdgeI) : CurNode(EdgeI.CurNode), EndNode(EdgeI.EndNode), CurEdge(EdgeI.CurEdge) { }
    TEdgeI& operator = (const TEdgeI& EdgeI) { if (this!=&EdgeI) { CurNode=EdgeI.CurNode;  EndNode=EdgeI.EndNode;  CurEdge=EdgeI.CurEdge; }  return *this; }

    TEdgeI& operator++ (int) { CurEdge++; if (CurEdge >= CurNode.GetOutDeg()) { CurEdge=0;  CurNode++;
      while (CurNode < EndNode && CurNode.GetOutDeg()==0) { CurNode++; } }  return *this; }
    bool operator < (const TEdgeI& EdgeI) const { return CurNode<EdgeI.CurNode || (CurNode==EdgeI.CurNode && CurEdge<EdgeI.CurEdge); }
    bool operator == (const TEdgeI& EdgeI) const { return CurNode == EdgeI.CurNode && CurEdge == EdgeI.CurEdge; }

    int GetId() const { return -1; }

    int GetSrcNId() const { return CurNode.GetId(); }

    int GetDstNId() const { return CurNode.GetOutNId(CurEdge); }
    TEdgeData& operator () () { return CurNode.GetOutEDat(CurEdge); }
    const TEdgeData& operator () () const { return CurNode.GetOutEDat(CurEdge); }
    TEdgeData& GetDat() { return CurNode.GetOutEDat(CurEdge); }
    const TEdgeData& GetDat() const { return CurNode.GetOutEDat(CurEdge); }
    TNodeData& GetSrcNDat() { return CurNode(); }
    const TNodeData& GetSrcNDat() const { return CurNode(); }
    TNodeData& GetDstNDat() { return CurNode.GetOutNDat(CurEdge); }
    const TNodeData& GetDstNDat() const { return CurNode.GetOutNDat(CurEdge); }
    friend class TNodeEDatNet<TNodeData, TEdgeData>;
  };
protected:
  TNode& GetNode(const int& NId) { return NodeH.GetDat(NId); }
  static int GetNIdPos(const TVec<TPair<TInt, TEdgeData> >& NIdV, const int& NId);
protected:
  TCRef CRef;
  TInt MxNId;
  THash<TInt, TNode> NodeH;
private:
  class TNodeFunctor {
  public:
    TNodeFunctor() {}
    void operator() (TNode* n, TShMIn& ShMIn) { n->LoadShM(ShMIn);}
  };
private:
  void LoadNetworkShM(TShMIn& ShMIn) {
    MxNId = TInt(ShMIn);
    TNodeFunctor f;
    NodeH.LoadShM(ShMIn, f);
  }
public:
  TNodeEDatNet() : CRef(), MxNId(0), NodeH() { }

  explicit TNodeEDatNet(const int& Nodes, const int& Edges) : MxNId(0) { Reserve(Nodes, Edges); }
  TNodeEDatNet(const TNodeEDatNet& NodeNet) : MxNId(NodeNet.MxNId), NodeH(NodeNet.NodeH) { }

  TNodeEDatNet(TSIn& SIn) : MxNId(SIn), NodeH(SIn) { }
  virtual ~TNodeEDatNet() { }

  virtual void Save(TSOut& SOut) const { MxNId.Save(SOut);  NodeH.Save(SOut); }

  static PNet New() { return PNet(new TNet()); }

  static PNet Load(TSIn& SIn) { return PNet(new TNet(SIn)); }

  static PNet LoadShM(TShMIn& ShMIn) {
    TNet* Network = new TNet();
    Network->LoadNetworkShM(ShMIn);
    return PNet(Network);
  }

  bool HasFlag(const TGraphFlag& Flag) const;
  TNodeEDatNet& operator = (const TNodeEDatNet& NodeNet) { if (this!=&NodeNet) {
    NodeH=NodeNet.NodeH;  MxNId=NodeNet.MxNId; }  return *this; }


  int GetNodes() const { return NodeH.Len(); }

  int AddNode(int NId = -1);

  int AddNodeUnchecked(int NId = -1);

  int AddNode(int NId, const TNodeData& NodeDat);

  int AddNode(const TNodeI& NodeI) { return AddNode(NodeI.GetId(), NodeI.GetDat()); }

  void DelNode(const int& NId);

  void DelNode(const TNode& NodeI) { DelNode(NodeI.GetId()); }

  bool IsNode(const int& NId) const { return NodeH.IsKey(NId); }

  TNodeI BegNI() const { return TNodeI(NodeH.BegI(), this); }

  TNodeI EndNI() const { return TNodeI(NodeH.EndI(), this); }

  TNodeI GetNI(const int& NId) const { return TNodeI(NodeH.GetI(NId), this); }

  const TNode& GetNode(const int& NId) const { return NodeH.GetDat(NId); }

  void SetNDat(const int& NId, const TNodeData& NodeDat);

  TNodeData& GetNDat(const int& NId) { return NodeH.GetDat(NId).NodeDat; }

  const TNodeData& GetNDat(const int& NId) const { return NodeH.GetDat(NId).NodeDat; }

  int GetMxNId() const { return MxNId; }


  int GetEdges() const;

  int AddEdge(const int& SrcNId, const int& DstNId);

  int AddEdge(const int& SrcNId, const int& DstNId, const TEdgeData& EdgeDat);

  int AddEdge(const TEdgeI& EdgeI) { return AddEdge(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), EdgeI()); }

  void DelEdge(const int& SrcNId, const int& DstNId, const bool& IsDir = true);

  bool IsEdge(const int& SrcNId, const int& DstNId, const bool& IsDir = true) const;

  TEdgeI BegEI() const { TNodeI NI=BegNI();  while(NI<EndNI() && NI.GetOutDeg()==0) NI++; return TEdgeI(NI, EndNI()); }

  TEdgeI EndEI() const { return TEdgeI(EndNI(), EndNI()); }

  TEdgeI GetEI(const int& EId) const;

  TEdgeI GetEI(const int& SrcNId, const int& DstNId) const;

  void SetEDat(const int& SrcNId, const int& DstNId, const TEdgeData& EdgeDat);

  bool GetEDat(const int& SrcNId, const int& DstNId, TEdgeData& EdgeDat) const;

  TEdgeData& GetEDat(const int& SrcNId, const int& DstNId);

  const TEdgeData& GetEDat(const int& SrcNId, const int& DstNId) const;

  void SetAllEDat(const TEdgeData& EdgeDat);

  int GetRndNId(TRnd& Rnd=TInt::Rnd) { return NodeH.GetKey(NodeH.GetRndKeyId(Rnd, 0.8)); }

  TNodeI GetRndNI(TRnd& Rnd=TInt::Rnd) { return GetNI(GetRndNId(Rnd)); }

  void GetNIdV(TIntV& NIdV) const;

  bool Empty() const { return GetNodes()==0; }

  void Clr(const bool& DoDel=true, const bool& ResetDat=true) {
    MxNId = 0;  NodeH.Clr(DoDel, -1, ResetDat); }

  void Reserve(const int& Nodes, const int& Edges) { if (Nodes>0) { NodeH.Gen(Nodes/2); } }

  void SortNIdById(const bool& Asc=true) { NodeH.SortByKey(Asc); }

  void SortNIdByDat(const bool& Asc=true) { NodeH.SortByDat(Asc); }

  void Defrag(const bool& OnlyNodeLinks=false);

  bool IsOk(const bool& ThrowExcept=true) const;
  friend class TPt<TNodeEDatNet<TNodeData, TEdgeData> >;
};
namespace TSnap {
template <class TNodeData, class TEdgeData> struct IsDirected<TNodeEDatNet<TNodeData, TEdgeData> > { enum { Val = 1 }; };
template <class TNodeData, class TEdgeData> struct IsNodeDat<TNodeEDatNet<TNodeData, TEdgeData> > { enum { Val = 1 }; };
template <class TNodeData, class TEdgeData> struct IsEdgeDat<TNodeEDatNet<TNodeData, TEdgeData> > { enum { Val = 1 }; };
}
template <class TNodeData, class TEdgeData>
bool TNodeEDatNet<TNodeData, TEdgeData>::HasFlag(const TGraphFlag& Flag) const {
  return HasGraphFlag(typename TNet, Flag);
}
template <class TNodeData, class TEdgeData>
int TNodeEDatNet<TNodeData, TEdgeData>::GetNIdPos(const TVec<TPair<TInt, TEdgeData> >& NIdV, const int& NId) {
  int LValN=0, RValN = NIdV.Len()-1;
  while (RValN >= LValN) {
    const int ValN = (LValN+RValN)/2;
    const int CurNId = NIdV[ValN].Val1;
    if (NId == CurNId) { return ValN; }
    if (NId < CurNId) { RValN=ValN-1; }
    else { LValN=ValN+1; }
  }
  return -1;
}
template <class TNodeData, class TEdgeData>
int TNodeEDatNet<TNodeData, TEdgeData>::AddNode(int NId) {
  if (NId == -1) {
    NId = MxNId;  MxNId++;
  } else {
    IAssertR(!IsNode(NId), TStr::Fmt("NodeId %d already exists", NId));
    MxNId = TMath::Mx(NId+1, MxNId());
  }
  NodeH.AddDat(NId, TNode(NId));
  return NId;
}
template <class TNodeData, class TEdgeData>
int TNodeEDatNet<TNodeData, TEdgeData>::AddNodeUnchecked(int NId) {
  if (NId == -1) {
    NId = MxNId;  MxNId++;
  } else {
    if (IsNode(NId)) { return -1;}
    MxNId = TMath::Mx(NId+1, MxNId());
  }
  NodeH.AddDat(NId, TNode(NId));
  return NId;
}
template <class TNodeData, class TEdgeData>
int TNodeEDatNet<TNodeData, TEdgeData>::AddNode(int NId, const TNodeData& NodeDat) {
  if (NId == -1) {
    NId = MxNId;  MxNId++;
  } else {
    IAssertR(!IsNode(NId), TStr::Fmt("NodeId %d already exists", NId));
    MxNId = TMath::Mx(NId+1, MxNId());
  }
  NodeH.AddDat(NId, TNode(NId, NodeDat));
  return NId;
}
template <class TNodeData, class TEdgeData>
void TNodeEDatNet<TNodeData, TEdgeData>::SetNDat(const int& NId, const TNodeData& NodeDat) {
  IAssertR(IsNode(NId), TStr::Fmt("NodeId %d does not exist.", NId).CStr());
  NodeH.GetDat(NId).NodeDat = NodeDat;
}
template <class TNodeData, class TEdgeData>
void TNodeEDatNet<TNodeData, TEdgeData>::DelNode(const int& NId) {
  const TNode& Node = GetNode(NId);
  for (int out = 0; out < Node.GetOutDeg(); out++) {
    const int nbr = Node.GetOutNId(out);
    if (nbr == NId) { continue; }
    TIntV& NIdV = GetNode(nbr).InNIdV;
    const int pos = NIdV.SearchBin(NId);
    if (pos != -1) { NIdV.Del(pos); }
  }
  for (int in = 0; in < Node.GetInDeg(); in++) {
    const int nbr = Node.GetInNId(in);
    if (nbr == NId) { continue; }
    TNIdDatPrV& NIdDatV = GetNode(nbr).OutNIdV;
    const int pos = GetNIdPos(NIdDatV, NId);
    if (pos != -1) { NIdDatV.Del(pos); }
  }
  NodeH.DelKey(NId);
}
template <class TNodeData, class TEdgeData>
int TNodeEDatNet<TNodeData, TEdgeData>::GetEdges() const {
  int edges=0;
  for (int N=NodeH.FFirstKeyId(); NodeH.FNextKeyId(N); ) {
    edges+=NodeH[N].GetOutDeg(); }
  return edges;
}
template <class TNodeData, class TEdgeData>
int TNodeEDatNet<TNodeData, TEdgeData>::AddEdge(const int& SrcNId, const int& DstNId) {
  return AddEdge(SrcNId, DstNId, TEdgeData());
}
template <class TNodeData, class TEdgeData>
int TNodeEDatNet<TNodeData, TEdgeData>::AddEdge(const int& SrcNId, const int& DstNId, const TEdgeData& EdgeDat) {
  IAssertR(IsNode(SrcNId) && IsNode(DstNId), TStr::Fmt("%d or %d not a node.", SrcNId, DstNId).CStr());

  if (IsEdge(SrcNId, DstNId)) {
    GetEDat(SrcNId, DstNId) = EdgeDat;
    return -2;
  }
  GetNode(SrcNId).OutNIdV.AddSorted(TPair<TInt, TEdgeData>(DstNId, EdgeDat));
  GetNode(DstNId).InNIdV.AddSorted(SrcNId);
  return -1;
}
template <class TNodeData, class TEdgeData>
void TNodeEDatNet<TNodeData, TEdgeData>::DelEdge(const int& SrcNId, const int& DstNId, const bool& IsDir) {
  IAssertR(IsNode(SrcNId) && IsNode(DstNId), TStr::Fmt("%d or %d not a node.", SrcNId, DstNId).CStr());
  int pos = GetNIdPos(GetNode(SrcNId).OutNIdV, DstNId);
  if (pos != -1) { GetNode(SrcNId).OutNIdV.Del(pos); }
  pos = GetNode(DstNId).InNIdV.SearchBin(SrcNId);
  if (pos != -1) { GetNode(DstNId).InNIdV.Del(pos); }
  if (! IsDir) {
    pos = GetNIdPos(GetNode(DstNId).OutNIdV, SrcNId);
    if (pos != -1) { GetNode(DstNId).OutNIdV.Del(pos); }
    pos = GetNode(SrcNId).InNIdV.SearchBin(DstNId);
    if (pos != -1) { GetNode(SrcNId).InNIdV.Del(pos); }
  }
}
template <class TNodeData, class TEdgeData>
bool TNodeEDatNet<TNodeData, TEdgeData>::IsEdge(const int& SrcNId, const int& DstNId, const bool& IsDir) const {
  if (! IsNode(SrcNId) || ! IsNode(DstNId)) { return false; }
  if (IsDir) { return GetNode(SrcNId).IsOutNId(DstNId); }
  else { return GetNode(SrcNId).IsOutNId(DstNId) || GetNode(DstNId).IsOutNId(SrcNId); }
}
template <class TNodeData, class TEdgeData>
void TNodeEDatNet<TNodeData, TEdgeData>::SetEDat(const int& SrcNId, const int& DstNId, const TEdgeData& EdgeDat) {
  IAssertR(IsNode(SrcNId) && IsNode(DstNId), TStr::Fmt("%d or %d not a node.", SrcNId, DstNId).CStr());
  IAssertR(IsEdge(SrcNId, DstNId), TStr::Fmt("Edge between %d and %d does not exist.", SrcNId, DstNId).CStr());
  GetEDat(SrcNId, DstNId) = EdgeDat;
}
template <class TNodeData, class TEdgeData>
bool TNodeEDatNet<TNodeData, TEdgeData>::GetEDat(const int& SrcNId, const int& DstNId, TEdgeData& EdgeDat) const {
  if (! IsEdge(SrcNId, DstNId)) { return false; }
  const TNode& N = GetNode(SrcNId);
  EdgeDat = N.GetOutEDat(GetNIdPos(N.OutNIdV, DstNId));
  return true;
}
template <class TNodeData, class TEdgeData>
TEdgeData& TNodeEDatNet<TNodeData, TEdgeData>::GetEDat(const int& SrcNId, const int& DstNId) {
  Assert(IsEdge(SrcNId, DstNId));
  TNode& N = GetNode(SrcNId);
  return N.GetOutEDat(GetNIdPos(N.OutNIdV, DstNId));
}
template <class TNodeData, class TEdgeData>
const TEdgeData& TNodeEDatNet<TNodeData, TEdgeData>::GetEDat(const int& SrcNId, const int& DstNId) const {
  Assert(IsEdge(SrcNId, DstNId));
  const TNode& N = GetNode(SrcNId);
  return N.GetOutEDat(GetNIdPos(N.OutNIdV, DstNId));
}
template <class TNodeData, class TEdgeData>
void TNodeEDatNet<TNodeData, TEdgeData>::SetAllEDat(const TEdgeData& EdgeDat) {
  for (TEdgeI EI = BegEI(); EI < EndEI(); EI++) {
    EI() = EdgeDat;
  }
}
template <class TNodeData, class TEdgeData>
typename TNodeEDatNet<TNodeData, TEdgeData>::TEdgeI  TNodeEDatNet<TNodeData, TEdgeData>::GetEI(const int& SrcNId, const int& DstNId) const {
  const TNodeI SrcNI = GetNI(SrcNId);
  int NodeN = -1;

  const TNIdDatPrV& NIdDatV = SrcNI.NodeHI.GetDat().OutNIdV;
  int LValN=0, RValN=NIdDatV.Len()-1;
  while (RValN>=LValN){
    int ValN=(LValN+RValN)/2;
    if (DstNId==NIdDatV[ValN].Val1){ NodeN=ValN; break; }
    if (DstNId<NIdDatV[ValN].Val1){RValN=ValN-1;} else {LValN=ValN+1;}
  }
  if (NodeN == -1) { return EndEI(); }
  else { return TEdgeI(SrcNI, EndNI(), NodeN); }
}
template <class TNodeData, class TEdgeData>
void TNodeEDatNet<TNodeData, TEdgeData>::GetNIdV(TIntV& NIdV) const {
  NIdV.Reserve(GetNodes(), 0);
  for (int N=NodeH.FFirstKeyId(); NodeH.FNextKeyId(N); ) {
    NIdV.Add(NodeH.GetKey(N)); }
}
template <class TNodeData, class TEdgeData>
void TNodeEDatNet<TNodeData, TEdgeData>::Defrag(const bool& OnlyNodeLinks) {
  for (int n = NodeH.FFirstKeyId(); NodeH.FNextKeyId(n);) {
    TNode& Node = NodeH[n];
    Node.InNIdV.Pack();  Node.OutNIdV.Pack();
  }
  if (! OnlyNodeLinks && ! NodeH.IsKeyIdEqKeyN()) {
    NodeH.Defrag();
  }
}
template <class TNodeData, class TEdgeData>
bool TNodeEDatNet<TNodeData, TEdgeData>::IsOk(const bool& ThrowExcept) const {
  bool RetVal = true;
  for (int N = NodeH.FFirstKeyId(); NodeH.FNextKeyId(N); ) {
    const TNode& Node = NodeH[N];
    if (! Node.OutNIdV.IsSorted()) {
      const TStr Msg = TStr::Fmt("Out-neighbor list of node %d is not sorted.", Node.GetId());
      if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
    }
    if (! Node.InNIdV.IsSorted()) {
      const TStr Msg = TStr::Fmt("In-neighbor list of node %d is not sorted.", Node.GetId());
      if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
    }

    int prevNId = -1;
    for (int e = 0; e < Node.GetOutDeg(); e++) {
      if (! IsNode(Node.GetOutNId(e))) {
        const TStr Msg = TStr::Fmt("Out-edge %d --> %d: node %d does not exist.",
          Node.GetId(), Node.GetOutNId(e), Node.GetOutNId(e));
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      if (e > 0 && prevNId == Node.GetOutNId(e)) {
        const TStr Msg = TStr::Fmt("Node %d has duplidate out-edge %d --> %d.",
          Node.GetId(), Node.GetId(), Node.GetOutNId(e));
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      prevNId = Node.GetOutNId(e);
    }

    prevNId = -1;
    for (int e = 0; e < Node.GetInDeg(); e++) {
      if (! IsNode(Node.GetInNId(e))) {
        const TStr Msg = TStr::Fmt("In-edge %d <-- %d: node %d does not exist.",
          Node.GetId(), Node.GetInNId(e), Node.GetInNId(e));
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      if (e > 0 && prevNId == Node.GetInNId(e)) {
        const TStr Msg = TStr::Fmt("Node %d has duplidate in-edge %d <-- %d.",
          Node.GetId(), Node.GetId(), Node.GetInNId(e));
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      prevNId = Node.GetInNId(e);
    }
  }
  return RetVal;
}
typedef TNodeEDatNet<TInt, TInt> TIntNEDNet;
typedef TPt<TIntNEDNet> PIntNEDNet;
typedef TNodeEDatNet<TInt, TFlt> TIntFltNEDNet;
typedef TPt<TIntFltNEDNet> PIntFltNEDNet;
typedef TNodeEDatNet<TStr, TInt> TStrIntNEDNet;
typedef TPt<TStrIntNEDNet> PStrIntNEDNet;
template <class TNodeData, class TEdgeData>
class TNodeEdgeNet {
public:
  typedef TNodeData TNodeDat;
  typedef TEdgeData TEdgeDat;
  typedef TNodeEdgeNet<TNodeData, TEdgeData> TNet;
  typedef TPt<TNet> PNet;
public:
  class TNode {
  private:
    TInt Id;
    TIntV InEIdV, OutEIdV;
    TNodeData NodeDat;
  public:
    TNode() : Id(-1), InEIdV(), OutEIdV(), NodeDat() { }
    TNode(const int& NId) : Id(NId), InEIdV(), OutEIdV(), NodeDat()  { }
    TNode(const int& NId, const TNodeData& NodeData) : Id(NId), InEIdV(), OutEIdV(), NodeDat(NodeData) { }
    TNode(const TNode& Node) : Id(Node.Id), InEIdV(Node.InEIdV), OutEIdV(Node.OutEIdV), NodeDat(Node.NodeDat) { }
    TNode(TSIn& SIn) : Id(SIn), InEIdV(SIn), OutEIdV(SIn), NodeDat(SIn) { }
    void Save(TSOut& SOut) const { Id.Save(SOut);  InEIdV.Save(SOut);  OutEIdV.Save(SOut);  NodeDat.Save(SOut); }
    bool operator < (const TNode& Node) const { return NodeDat < Node.NodeDat; }
    int GetId() const { return Id; }
    int GetDeg() const { return GetInDeg() + GetOutDeg(); }
    int GetInDeg() const { return InEIdV.Len(); }
    int GetOutDeg() const { return OutEIdV.Len(); }
    const TNodeData& GetDat() const { return NodeDat; }
    TNodeData& GetDat() { return NodeDat; }
    int GetInEId(const int& NodeN) const { return InEIdV[NodeN]; }
    int GetOutEId(const int& NodeN) const { return OutEIdV[NodeN]; }
    int GetNbrEId(const int& EdgeN) const { return EdgeN<GetOutDeg()?GetOutEId(EdgeN):GetInEId(EdgeN-GetOutDeg()); }
    bool IsInEId(const int& EId) const { return InEIdV.SearchBin(EId) != -1; }
    bool IsOutEId(const int& EId) const { return OutEIdV.SearchBin(EId) != -1; }
    bool IsNbrEId(const int& EId) const { return IsInEId(EId) || IsOutEId(EId); }
    void LoadShM(TShMIn& MStream) {
      Id = TInt(MStream);
      InEIdV.LoadShM(MStream);
      OutEIdV.LoadShM(MStream);
      NodeDat = TNodeData(MStream);
    }
    friend class TNodeEdgeNet<TNodeData, TEdgeData>;
  };
  class TEdge {
  private:
    TInt Id, SrcNId, DstNId;
    TEdgeData EdgeDat;
  public:
    TEdge() : Id(-1), SrcNId(-1), DstNId(-1), EdgeDat() { }
    TEdge(const int& EId, const int& SourceNId, const int& DestNId) : Id(EId), SrcNId(SourceNId), DstNId(DestNId), EdgeDat() { }
    TEdge(const int& EId, const int& SourceNId, const int& DestNId, const TEdgeData& EdgeData) : Id(EId), SrcNId(SourceNId), DstNId(DestNId), EdgeDat(EdgeData) { }
    TEdge(const TEdge& Edge) : Id(Edge.Id), SrcNId(Edge.SrcNId), DstNId(Edge.DstNId), EdgeDat(Edge.EdgeDat) { }
    TEdge(TSIn& SIn) : Id(SIn), SrcNId(SIn), DstNId(SIn), EdgeDat(SIn) { }
    void Save(TSOut& SOut) const { Id.Save(SOut);  SrcNId.Save(SOut);  DstNId.Save(SOut);  EdgeDat.Save(SOut); }
    bool operator < (const TEdge& Edge) const { return EdgeDat < Edge.EdgeDat; }
    int GetId() const { return Id; }
    int GetSrcNId() const { return SrcNId; }
    int GetDstNId() const { return DstNId; }
    void Load(TSIn& InStream) {
      Id = TInt(InStream);
      SrcNId = TInt(InStream);
      DstNId = TInt(InStream);
      EdgeDat = TEdgeData(InStream);
    }
    const TEdgeData& GetDat() const { return EdgeDat; }
    TEdgeData& GetDat() { return EdgeDat; }
    friend class TNodeEdgeNet;
  };

  class TNodeI {
  private:
    typedef typename THash<TInt, TNode>::TIter THashIter;
    THashIter NodeHI;
    TNodeEdgeNet *Net;
  public:
    TNodeI() : NodeHI(), Net(NULL) { }
    TNodeI(const THashIter& NodeHIter, const TNodeEdgeNet* NetPt) : NodeHI(NodeHIter), Net((TNodeEdgeNet *)NetPt) { }
    TNodeI(const TNodeI& NodeI) : NodeHI(NodeI.NodeHI), Net(NodeI.Net) { }
    TNodeI& operator = (const TNodeI& NodeI) { NodeHI = NodeI.NodeHI;  Net=NodeI.Net;  return *this; }

    TNodeI& operator++ (int) { NodeHI++;  return *this; }
    bool operator < (const TNodeI& NodeI) const { return NodeHI < NodeI.NodeHI; }
    bool operator == (const TNodeI& NodeI) const { return NodeHI == NodeI.NodeHI; }

    int GetId() const { return NodeHI.GetDat().GetId(); }

    int GetDeg() const { return NodeHI.GetDat().GetDeg(); }

    int GetInDeg() const { return NodeHI.GetDat().GetInDeg(); }

    int GetOutDeg() const { return NodeHI.GetDat().GetOutDeg(); }

    int GetInNId(const int& EdgeN) const { return Net->GetEdge(NodeHI.GetDat().GetInEId(EdgeN)).GetSrcNId(); }

    int GetOutNId(const int& EdgeN) const { return Net->GetEdge(NodeHI.GetDat().GetOutEId(EdgeN)).GetDstNId(); }

    int GetNbrNId(const int& EdgeN) const { const TEdge& E = Net->GetEdge(NodeHI.GetDat().GetNbrEId(EdgeN));
      return GetId()==E.GetSrcNId() ? E.GetDstNId():E.GetSrcNId(); }

    bool IsInNId(const int& NId) const;

    bool IsOutNId(const int& NId) const;

    bool IsNbrNId(const int& NId) const { return IsOutNId(NId) || IsInNId(NId); }

    const TNodeData& operator () () const { return NodeHI.GetDat().GetDat(); }
    TNodeData& operator () () { return NodeHI.GetDat().GetDat(); }
    const TNodeData& GetDat() const { return NodeHI.GetDat().GetDat(); }
    TNodeData& GetDat() { return NodeHI.GetDat().GetDat(); }
    const TNodeData& GetInNDat(const int& EdgeN) const { return Net->GetNDat(GetInNId(EdgeN)); }
    TNodeData& GetInNDat(const int& EdgeN) { return Net->GetNDat(GetInNId(EdgeN)); }
    const TNodeData& GetOutNDat(const int& EdgeN) const { return Net->GetNDat(GetOutNId(EdgeN)); }
    TNodeData& GetOutNDat(const int& EdgeN) { return Net->GetNDat(GetOutNId(EdgeN)); }
    const TNodeData& GetNbrNDat(const int& EdgeN) const { return Net->GetNDat(GetNbrNId(EdgeN)); }
    TNodeData& GetNbrNDat(const int& EdgeN) { return Net->GetNDat(GetNbrNId(EdgeN)); }


    int GetInEId(const int& EdgeN) const { return NodeHI.GetDat().GetInEId(EdgeN); }

    int GetOutEId(const int& EdgeN) const { return NodeHI.GetDat().GetOutEId(EdgeN); }

    int GetNbrEId(const int& EdgeN) const { return NodeHI.GetDat().GetNbrEId(EdgeN); }

    bool IsInEId(const int& EId) const { return NodeHI.GetDat().IsInEId(EId); }

    bool IsOutEId(const int& EId) const { return NodeHI.GetDat().IsOutEId(EId); }

    bool IsNbrEId(const int& EId) const { return NodeHI.GetDat().IsNbrEId(EId); }

    TEdgeDat& GetInEDat(const int& EdgeN) { return Net->GetEDat(GetInEId(EdgeN)); }
    const TEdgeDat& GetInEDat(const int& EdgeN) const { return Net->GetEDat(GetInEId(EdgeN)); }
    TEdgeDat& GetOutEDat(const int& EdgeN) { return Net->GetEDat(GetOutEId(EdgeN)); }
    const TEdgeDat& GetOutEDat(const int& EdgeN) const { return Net->GetEDat(GetOutEId(EdgeN)); }
    TEdgeDat& GetNbrEDat(const int& EdgeN) { return Net->GetEDat(GetNbrEId(EdgeN)); }
    const TEdgeDat& GetNbrEDat(const int& EdgeN) const { return Net->GetEDat(GetNbrEId(EdgeN)); }
    friend class TNodeEdgeNet;
  };

  class TEdgeI {
  private:
    typedef typename THash<TInt, TEdge>::TIter THashIter;
    THashIter EdgeHI;
    TNodeEdgeNet *Net;
  public:
    TEdgeI() : EdgeHI(), Net(NULL) { }
    TEdgeI(const THashIter& EdgeHIter, const TNodeEdgeNet *NetPt) : EdgeHI(EdgeHIter), Net((TNodeEdgeNet *) NetPt) { }
    TEdgeI(const TEdgeI& EdgeI) : EdgeHI(EdgeI.EdgeHI), Net(EdgeI.Net) { }
    TEdgeI& operator = (const TEdgeI& EdgeI) { if (this!=&EdgeI) { EdgeHI=EdgeI.EdgeHI;  Net=EdgeI.Net; }  return *this; }
    TEdgeI& operator++ (int) { EdgeHI++;  return *this; }
    bool operator < (const TEdgeI& EdgeI) const { return EdgeHI < EdgeI.EdgeHI; }
    bool operator == (const TEdgeI& EdgeI) const { return EdgeHI == EdgeI.EdgeHI; }

    int GetId() const { return EdgeHI.GetDat().GetId(); }

    int GetSrcNId() const { return EdgeHI.GetDat().GetSrcNId(); }

    int GetDstNId() const { return EdgeHI.GetDat().GetDstNId(); }
    const TEdgeData& operator () () const { return EdgeHI.GetDat().GetDat(); }
    TEdgeData& operator () () { return EdgeHI.GetDat().GetDat(); }
    const TEdgeData& GetDat() const { return EdgeHI.GetDat().GetDat(); }
    TEdgeData& GetDat() { return EdgeHI.GetDat().GetDat(); }
    const TNodeData& GetSrcNDat() const { return Net->GetNDat(GetSrcNId()); }
    TNodeData& GetSrcNDat() { return Net->GetNDat(GetSrcNId()); }
    const TNodeData& GetDstNDat() const { return Net->GetNDat(GetDstNId()); }
    TNodeData& GetDstNDat() { return Net->GetNDat(GetDstNId()); }
    friend class TNodeEdgeNet;
  };
private:
  TNode& GetNode(const int& NId) { return NodeH.GetDat(NId); }
  const TNode& GetNode(const int& NId) const { return NodeH.GetDat(NId); }
  const TNode& GetNodeKId(const int& NodeKeyId) const { return NodeH[NodeKeyId]; }
  TEdge& GetEdge(const int& EId) { return EdgeH.GetDat(EId); }
  const TEdge& GetEdge(const int& EId) const { return EdgeH.GetDat(EId); }
  const TEdge& GetEdgeKId(const int& EdgeKeyId) const { return EdgeH[EdgeKeyId]; }
protected:
  TCRef CRef;
  TInt MxNId, MxEId;
  THash<TInt, TNode> NodeH;
  THash<TInt, TEdge> EdgeH;
private:
  class LoadTNodeFunctor {
  public:
    LoadTNodeFunctor() {}
    void operator() (TNode* n, TShMIn& ShMIn) { n->LoadShM(ShMIn);}
  };
private:
  void LoadNetworkShM(TShMIn& ShMIn) {
    MxNId = TInt(ShMIn);
    MxEId = TInt(ShMIn);
    LoadTNodeFunctor fn;
    NodeH.LoadShM(ShMIn, fn);
    EdgeH.LoadShM(ShMIn);
  }
public:
  TNodeEdgeNet() : CRef(), MxNId(0), MxEId(0) { }

  explicit TNodeEdgeNet(const int& Nodes, const int& Edges) : CRef(), MxNId(0), MxEId(0) { Reserve(Nodes, Edges); }
  TNodeEdgeNet(const TNodeEdgeNet& Net) : MxNId(Net.MxNId), MxEId(Net.MxEId), NodeH(Net.NodeH), EdgeH(Net.EdgeH) { }

  TNodeEdgeNet(TSIn& SIn) : MxNId(SIn), MxEId(SIn), NodeH(SIn), EdgeH(SIn) { }
  virtual ~TNodeEdgeNet() { }

  virtual void Save(TSOut& SOut) const { MxNId.Save(SOut);  MxEId.Save(SOut);  NodeH.Save(SOut);  EdgeH.Save(SOut); }

  static PNet New() { return PNet(new TNet()); }

  static PNet Load(TSIn& SIn) { return PNet(new TNet(SIn)); }

  static PNet LoadShM(TShMIn& ShMIn) {
    TNet* Network = new TNet();
    Network->LoadNetworkShM(ShMIn);
    return PNet(Network);
  }

  bool HasFlag(const TGraphFlag& Flag) const;
  TNodeEdgeNet& operator = (const TNodeEdgeNet& Net) {
    if (this!=&Net) { NodeH=Net.NodeH; EdgeH=Net.EdgeH; MxNId=Net.MxNId; MxEId=Net.MxEId; }  return *this; }


  int GetNodes() const { return NodeH.Len(); }

  int AddNode(int NId = -1);

  int AddNodeUnchecked(int NId = -1);

  int AddNode(int NId, const TNodeData& NodeDat);

  friend class TCrossNet;
  int AddNode(const TNodeI& NodeI) { return AddNode(NodeI.GetId(), NodeI.GetDat()); }

  void DelNode(const int& NId);

  void DelNode(const TNode& NodeI) { DelNode(NodeI.GetId()); }

  bool IsNode(const int& NId) const { return NodeH.IsKey(NId); }

  TNodeI BegNI() const { return TNodeI(NodeH.BegI(), this); }

  TNodeI EndNI() const { return TNodeI(NodeH.EndI(), this); }

  TNodeI GetNI(const int& NId) const { return TNodeI(NodeH.GetI(NId), this); }

  void SetNDat(const int& NId, const TNodeData& NodeDat);

  TNodeData& GetNDat(const int& NId) { return NodeH.GetDat(NId).NodeDat; }

  const TNodeData& GetNDat(const int& NId) const { return NodeH.GetDat(NId).NodeDat; }

  int GetMxNId() const { return MxNId; }


  int GetEdges() const { return EdgeH.Len(); }

  int GetUniqEdges(const bool& IsDir=true) const;

  int AddEdge(const int& SrcNId, const int& DstNId, int EId = -1);

  int AddEdge(const int& SrcNId, const int& DstNId, int EId, const TEdgeData& EdgeDat);

  int AddEdge(const TEdgeI& EdgeI) { return AddEdge(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), EdgeI.GetId(), EdgeI.GetDat()); }

  void DelEdge(const int& EId);

  void DelEdge(const int& SrcNId, const int& DstNId, const bool& IsDir = true);

  bool IsEdge(const int& EId) const { return EdgeH.IsKey(EId); }

  bool IsEdge(const int& SrcNId, const int& DstNId, const bool& IsDir = true) const { int EId;  return IsEdge(SrcNId, DstNId, EId, IsDir); }

  bool IsEdge(const int& SrcNId, const int& DstNId, int& EId, const bool& IsDir = true) const;
  int GetEId(const int& SrcNId, const int& DstNId) const { int EId; return IsEdge(SrcNId, DstNId, EId)?EId:-1; }

  TEdgeI BegEI() const { return TEdgeI(EdgeH.BegI(), this); }

  TEdgeI EndEI() const { return TEdgeI(EdgeH.EndI(), this); }

  TEdgeI GetEI(const int& EId) const { return TEdgeI(EdgeH.GetI(EId), this); }

  TEdgeI GetEI(const int& SrcNId, const int& DstNId) const { return GetEI(GetEId(SrcNId, DstNId)); }

  void SetEDat(const int& EId, const TEdgeData& EdgeDat);

  TEdgeData& GetEDat(const int& EId) { return EdgeH.GetDat(EId).EdgeDat; }

  const TEdgeData& GetEDat(const int& EId) const { return EdgeH.GetDat(EId).EdgeDat; }

  void SetAllEDat(const TEdgeData& EdgeDat);

  int GetRndNId(TRnd& Rnd=TInt::Rnd) { return NodeH.GetKey(NodeH.GetRndKeyId(Rnd, 0.8)); }

  TNodeI GetRndNI(TRnd& Rnd=TInt::Rnd) { return GetNI(GetRndNId(Rnd)); }

  int GetRndEId(TRnd& Rnd=TInt::Rnd) { return EdgeH.GetKey(EdgeH.GetRndKeyId(Rnd, 0.8)); }

  TEdgeI GetRndEI(TRnd& Rnd=TInt::Rnd) { return GetEI(GetRndEId(Rnd)); }

  void GetNIdV(TIntV& NIdV) const;

  void GetEIdV(TIntV& EIdV) const;

  bool Empty() const { return GetNodes()==0; }

  void Clr() { MxNId=0;  MxEId=0;  NodeH.Clr();  EdgeH.Clr(); }

  void Reserve(const int& Nodes, const int& Edges) {
    if (Nodes>0) { NodeH.Gen(Nodes/2); }  if (Edges>0) { EdgeH.Gen(Edges/2); } }

  void SortNIdById(const bool& Asc=true) { NodeH.SortByKey(Asc); }

  void SortNIdByDat(const bool& Asc=true) { NodeH.SortByDat(Asc); }

  void SortEIdById(const bool& Asc=true) { EdgeH.SortByKey(Asc); }

  void SortEIdByDat(const bool& Asc=true) { EdgeH.SortByDat(Asc); }

  void Defrag(const bool& OnlyNodeLinks=false);

  bool IsOk(const bool& ThrowExcept=true) const;
  friend class TPt<TNodeEdgeNet<TNodeData, TEdgeData> >;
};
namespace TSnap {
template <class TNodeData, class TEdgeData> struct IsMultiGraph<TNodeEdgeNet<TNodeData, TEdgeData> > { enum { Val = 1 }; };
template <class TNodeData, class TEdgeData> struct IsDirected<TNodeEdgeNet<TNodeData, TEdgeData> > { enum { Val = 1 }; };
template <class TNodeData, class TEdgeData> struct IsNodeDat<TNodeEdgeNet<TNodeData, TEdgeData> > { enum { Val = 1 }; };
template <class TNodeData, class TEdgeData> struct IsEdgeDat<TNodeEdgeNet<TNodeData, TEdgeData> > { enum { Val = 1 }; };
}
template <class TNodeData, class TEdgeData>
bool TNodeEdgeNet<TNodeData, TEdgeData>::HasFlag(const TGraphFlag& Flag) const {
  return HasGraphFlag(typename TNet, Flag);
}
template <class TNodeData, class TEdgeData>
bool TNodeEdgeNet<TNodeData, TEdgeData>::TNodeI::IsInNId(const int& NId) const {
  const TNode& Node = NodeHI.GetDat();
  for (int edge = 0; edge < Node.GetInDeg(); edge++) {
    if (NId == Net->GetEdge(Node.GetInEId(edge)).GetSrcNId())
      return true;
  }
  return false;
}
template <class TNodeData, class TEdgeData>
bool TNodeEdgeNet<TNodeData, TEdgeData>::TNodeI::IsOutNId(const int& NId) const {
  const TNode& Node = NodeHI.GetDat();
  for (int edge = 0; edge < Node.GetOutDeg(); edge++) {
    if (NId == Net->GetEdge(Node.GetOutEId(edge)).GetDstNId())
      return true;
  }
  return false;
}
template <class TNodeData, class TEdgeData>
int TNodeEdgeNet<TNodeData, TEdgeData>::AddNode(int NId) {
  if (NId == -1) {
    NId = MxNId;  MxNId++;
  } else {
    IAssertR(!IsNode(NId), TStr::Fmt("NodeId %d already exists", NId));
    MxNId = TMath::Mx(NId+1, MxNId());
  }
  NodeH.AddDat(NId, TNode(NId));
  return NId;
}
template <class TNodeData, class TEdgeData>
int TNodeEdgeNet<TNodeData, TEdgeData>::AddNodeUnchecked(int NId) {
  if (NId == -1) {
    NId = MxNId;  MxNId++;
  } else {
    if (IsNode(NId)) { return -1;}
    MxNId = TMath::Mx(NId+1, MxNId());
  }
  NodeH.AddDat(NId, TNode(NId));
  return NId;
}
template <class TNodeData, class TEdgeData>
int TNodeEdgeNet<TNodeData, TEdgeData>::AddNode(int NId, const TNodeData& NodeDat) {
  if (NId == -1) {
    NId = MxNId;  MxNId++;
  } else {
    IAssertR(!IsNode(NId), TStr::Fmt("NodeId %d already exists", NId));
    MxNId = TMath::Mx(NId+1, MxNId());
  }
  NodeH.AddDat(NId, TNode(NId, NodeDat));
  return NId;
}
template <class TNodeData, class TEdgeData>
void TNodeEdgeNet<TNodeData, TEdgeData>::DelNode(const int& NId) {
  const TNode& Node = GetNode(NId);
  for (int out = 0; out < Node.GetOutDeg(); out++) {
    const int EId = Node.GetOutEId(out);
    const TEdge& Edge = GetEdge(EId);
    IAssert(Edge.GetSrcNId() == NId);
    GetNode(Edge.GetDstNId()).InEIdV.DelIfIn(EId);
    EdgeH.DelKey(EId);
  }
  for (int in = 0; in < Node.GetInDeg(); in++) {
    const int EId = Node.GetInEId(in);
    const TEdge& Edge = GetEdge(EId);
    IAssert(Edge.GetDstNId() == NId);
    GetNode(Edge.GetSrcNId()).OutEIdV.DelIfIn(EId);
    EdgeH.DelKey(EId);
  }
  NodeH.DelKey(NId);
}
template <class TNodeData, class TEdgeData>
void TNodeEdgeNet<TNodeData, TEdgeData>::SetNDat(const int& NId, const TNodeData& NodeDat) {
  IAssertR(IsNode(NId), TStr::Fmt("NodeId %d does not exist.", NId).CStr());
  NodeH.GetDat(NId).NodeDat = NodeDat;
}
template <class TNodeData, class TEdgeData>
int TNodeEdgeNet<TNodeData, TEdgeData>::GetUniqEdges(const bool& IsDir) const {
  TIntPrSet UniqESet(GetEdges());
  for (TEdgeI EI = BegEI(); EI < EndEI(); EI++) {
    const int Src = EI.GetSrcNId();
    const int Dst = EI.GetDstNId();
    if (IsDir) { UniqESet.AddKey(TIntPr(Src, Dst)); }
    else { UniqESet.AddKey(TIntPr(TMath::Mn(Src, Dst), TMath::Mx(Src, Dst))); }
  }
  return UniqESet.Len();
}
template <class TNodeData, class TEdgeData>
int TNodeEdgeNet<TNodeData, TEdgeData>::AddEdge(const int& SrcNId, const int& DstNId, int EId) {
  if (EId == -1) { EId = MxEId;  MxEId++; }
  else { MxEId = TMath::Mx(EId+1, MxEId()); }
  IAssertR(!IsEdge(EId), TStr::Fmt("EdgeId %d already exists", EId));
  IAssertR(IsNode(SrcNId) && IsNode(DstNId), TStr::Fmt("%d or %d not a node.", SrcNId, DstNId).CStr());
  EdgeH.AddDat(EId, TEdge(EId, SrcNId, DstNId));
  GetNode(SrcNId).OutEIdV.AddSorted(EId);
  GetNode(DstNId).InEIdV.AddSorted(EId);
  return EId;
}
template <class TNodeData, class TEdgeData>
int TNodeEdgeNet<TNodeData, TEdgeData>::AddEdge(const int& SrcNId, const int& DstNId, int EId, const TEdgeData& EdgeDat) {
  if (EId == -1) { EId = MxEId;  MxEId++; }
  else { MxEId = TMath::Mx(EId+1, MxEId()); }
  IAssertR(!IsEdge(EId), TStr::Fmt("EdgeId %d already exists", EId));
  IAssertR(IsNode(SrcNId) && IsNode(DstNId), TStr::Fmt("%d or %d not a node.", SrcNId, DstNId).CStr());
  EdgeH.AddDat(EId, TEdge(EId, SrcNId, DstNId, EdgeDat));
  GetNode(SrcNId).OutEIdV.AddSorted(EId);
  GetNode(DstNId).InEIdV.AddSorted(EId);
  return EId;
}
template <class TNodeData, class TEdgeData>
void TNodeEdgeNet<TNodeData, TEdgeData>::DelEdge(const int& EId) {
  IAssert(IsEdge(EId));
  const int SrcNId = GetEdge(EId).GetSrcNId();
  const int DstNId = GetEdge(EId).GetDstNId();
  GetNode(SrcNId).OutEIdV.DelIfIn(EId);
  GetNode(DstNId).InEIdV.DelIfIn(EId);
  EdgeH.DelKey(EId);
}
template <class TNodeData, class TEdgeData>
void TNodeEdgeNet<TNodeData, TEdgeData>::DelEdge(const int& SrcNId, const int& DstNId, const bool& IsDir) {
  int EId;
  IAssert(IsEdge(SrcNId, DstNId, EId, IsDir));
  GetNode(SrcNId).OutEIdV.DelIfIn(EId);
  GetNode(DstNId).InEIdV.DelIfIn(EId);
  EdgeH.DelKey(EId);
}
template <class TNodeData, class TEdgeData>
bool TNodeEdgeNet<TNodeData, TEdgeData>::IsEdge(const int& SrcNId, const int& DstNId, int& EId, const bool& IsDir) const {
  if (! IsNode(SrcNId)) { return false; }
  if (! IsNode(DstNId)) { return false; }
  const TNode& SrcNode = GetNode(SrcNId);
  for (int edge = 0; edge < SrcNode.GetOutDeg(); edge++) {
    const TEdge& Edge = GetEdge(SrcNode.GetOutEId(edge));
    if (DstNId == Edge.GetDstNId()) {
      EId = Edge.GetId();  return true; }
  }
  if (! IsDir) {
    for (int edge = 0; edge < SrcNode.GetInDeg(); edge++) {
    const TEdge& Edge = GetEdge(SrcNode.GetInEId(edge));
    if (DstNId == Edge.GetSrcNId()) {
      EId = Edge.GetId();  return true; }
    }
  }
  return false;
}
template <class TNodeData, class TEdgeData>
void TNodeEdgeNet<TNodeData, TEdgeData>::SetEDat(const int& EId, const TEdgeData& EdgeDat) {
  IAssertR(IsEdge(EId), TStr::Fmt("EdgeId %d does not exist.", EId).CStr());
  GetEI(EId).GetDat() = EdgeDat;
}
template <class TNodeData, class TEdgeData>
void TNodeEdgeNet<TNodeData, TEdgeData>::SetAllEDat(const TEdgeData& EdgeDat) {
  for (TEdgeI EI = BegEI(); EI < EndEI(); EI++) {
    EI() = EdgeDat;
  }
}
template <class TNodeData, class TEdgeData>
void TNodeEdgeNet<TNodeData, TEdgeData>::GetNIdV(TIntV& NIdV) const {
  NIdV.Gen(GetNodes(), 0);
  for (int N=NodeH.FFirstKeyId(); NodeH.FNextKeyId(N);) {
    NIdV.Add(NodeH.GetKey(N));
  }
}
template <class TNodeData, class TEdgeData>
void TNodeEdgeNet<TNodeData, TEdgeData>::GetEIdV(TIntV& EIdV) const {
  EIdV.Gen(GetEdges(), 0);
  for (int E=EdgeH.FFirstKeyId(); EdgeH.FNextKeyId(E);) {
    EIdV.Add(EdgeH.GetKey(E));
  }
}
template <class TNodeData, class TEdgeData>
void TNodeEdgeNet<TNodeData, TEdgeData>::Defrag(const bool& OnlyNodeLinks) {
  for (int kid = NodeH.FFirstKeyId(); NodeH.FNextKeyId(kid);) {
    TNode& Node = NodeH[kid];
    Node.InEIdV.Pack();  Node.OutEIdV.Pack();
  }
  if (! OnlyNodeLinks && ! NodeH.IsKeyIdEqKeyN()) { NodeH.Defrag(); }
  if (! OnlyNodeLinks && ! EdgeH.IsKeyIdEqKeyN()) { EdgeH.Defrag(); }
}
template <class TNodeData, class TEdgeData>
bool TNodeEdgeNet<TNodeData, TEdgeData>::IsOk(const bool& ThrowExcept) const {
  bool RetVal = true;
  for (int N = NodeH.FFirstKeyId(); NodeH.FNextKeyId(N); ) {
    const TNode& Node = NodeH[N];
    if (! Node.OutEIdV.IsSorted()) {
      const TStr Msg = TStr::Fmt("Out-edge list of node %d is not sorted.", Node.GetId());
      if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
    }
    if (! Node.InEIdV.IsSorted()) {
      const TStr Msg = TStr::Fmt("In-edge list of node %d is not sorted.", Node.GetId());
      if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
    }

    int prevEId = -1;
    for (int e = 0; e < Node.GetOutDeg(); e++) {
      if (! IsEdge(Node.GetOutEId(e))) {
        const TStr Msg = TStr::Fmt("Out-edge id %d of node %d does not exist.",  Node.GetOutEId(e), Node.GetId());
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      if (e > 0 && prevEId == Node.GetOutEId(e)) {
        const TStr Msg = TStr::Fmt("Node %d has duplidate out-edge id %d.", Node.GetId(), Node.GetOutEId(e));
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      prevEId = Node.GetOutEId(e);
    }

    prevEId = -1;
    for (int e = 0; e < Node.GetInDeg(); e++) {
      if (! IsEdge(Node.GetInEId(e))) {
        const TStr Msg = TStr::Fmt("Out-edge id %d of node %d does not exist.",  Node.GetInEId(e), Node.GetId());
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      if (e > 0 && prevEId == Node.GetInEId(e)) {
        const TStr Msg = TStr::Fmt("Node %d has duplidate out-edge id %d.", Node.GetId(), Node.GetInEId(e));
        if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
      }
      prevEId = Node.GetInEId(e);
    }
  }
  for (int E = EdgeH.FFirstKeyId(); EdgeH.FNextKeyId(E); ) {
    const TEdge& Edge = EdgeH[E];
    if (! IsNode(Edge.GetSrcNId())) {
      const TStr Msg = TStr::Fmt("Edge %d source node %d does not exist.", Edge.GetId(), Edge.GetSrcNId());
      if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
    }
    if (! IsNode(Edge.GetDstNId())) {
      const TStr Msg = TStr::Fmt("Edge %d destination node %d does not exist.", Edge.GetId(), Edge.GetDstNId());
      if (ThrowExcept) { EAssertR(false, Msg); } else { ErrNotify(Msg.CStr()); } RetVal=false;
    }
  }
  return RetVal;
}
typedef TNodeEdgeNet<TInt, TInt> TIntNENet;
typedef TPt<TIntNENet> PIntNENet;
typedef TNodeEdgeNet<TFlt, TFlt> TFltNENet;
typedef TPt<TFltNENet> PFltNENet;
class TNEANet;
typedef TPt<TNEANet> PNEANet;
class TNEANet {
public:
  typedef TNEANet TNet;
  typedef TPt<TNEANet> PNet;
public:
  class TNode {
  private:
    TInt Id;
    TIntV InEIdV, OutEIdV;
  public:
    TNode() : Id(-1), InEIdV(), OutEIdV() { }
    TNode(const int& NId) : Id(NId), InEIdV(), OutEIdV() { }
    TNode(const TNode& Node) : Id(Node.Id), InEIdV(Node.InEIdV), OutEIdV(Node.OutEIdV) { }
    TNode(TSIn& SIn) : Id(SIn), InEIdV(SIn), OutEIdV(SIn) { }
    void Save(TSOut& SOut) const { Id.Save(SOut); InEIdV.Save(SOut); OutEIdV.Save(SOut); }
    int GetId() const { return Id; }
    int GetDeg() const { return GetInDeg() + GetOutDeg(); }
    int GetInDeg() const { return InEIdV.Len(); }
    int GetOutDeg() const { return OutEIdV.Len(); }
    int GetInEId(const int& EdgeN) const { return InEIdV[EdgeN]; }
    int GetOutEId(const int& EdgeN) const { return OutEIdV[EdgeN]; }
    int GetNbrEId(const int& EdgeN) const { return EdgeN<GetOutDeg()?GetOutEId(EdgeN):GetInEId(EdgeN-GetOutDeg()); }
    bool IsInEId(const int& EId) const { return InEIdV.SearchBin(EId) != -1; }
    bool IsOutEId(const int& EId) const { return OutEIdV.SearchBin(EId) != -1; }
    void LoadShM(TShMIn& MStream) {
      Id = TInt(MStream);
      InEIdV.LoadShM(MStream);
      OutEIdV.LoadShM(MStream);
    }
    friend class TNEANet;
  };
  class TEdge {
  private:
    TInt Id, SrcNId, DstNId;
  public:
    TEdge() : Id(-1), SrcNId(-1), DstNId(-1) { }
    TEdge(const int& EId, const int& SourceNId, const int& DestNId) : Id(EId), SrcNId(SourceNId), DstNId(DestNId) { }
    TEdge(const TEdge& Edge) : Id(Edge.Id), SrcNId(Edge.SrcNId), DstNId(Edge.DstNId) { }
    TEdge(TSIn& SIn) : Id(SIn), SrcNId(SIn), DstNId(SIn) { }
    void Save(TSOut& SOut) const { Id.Save(SOut); SrcNId.Save(SOut); DstNId.Save(SOut); }
    int GetId() const { return Id; }
    int GetSrcNId() const { return SrcNId; }
    int GetDstNId() const { return DstNId; }
    void Load(TSIn& InStream) {
      Id = TInt(InStream);
      SrcNId = TInt(InStream);
      DstNId = TInt(InStream);
    }
    friend class TNEANet;
  };

  class TNodeI {
  protected:
    typedef THash<TInt, TNode>::TIter THashIter;
    THashIter NodeHI;
    const TNEANet *Graph;
  public:
    TNodeI() : NodeHI(), Graph(NULL) { }
    TNodeI(const THashIter& NodeHIter, const TNEANet* GraphPt) : NodeHI(NodeHIter), Graph(GraphPt) { }
    TNodeI(const TNodeI& NodeI) : NodeHI(NodeI.NodeHI), Graph(NodeI.Graph) { }
    TNodeI& operator = (const TNodeI& NodeI) { NodeHI = NodeI.NodeHI; Graph=NodeI.Graph; return *this; }

    TNodeI& operator++ (int) { NodeHI++; return *this; }
    bool operator < (const TNodeI& NodeI) const { return NodeHI < NodeI.NodeHI; }
    bool operator == (const TNodeI& NodeI) const { return NodeHI == NodeI.NodeHI; }

    int GetId() const { return NodeHI.GetDat().GetId(); }

    int GetDeg() const { return NodeHI.GetDat().GetDeg(); }

    int GetInDeg() const { return NodeHI.GetDat().GetInDeg(); }

    int GetOutDeg() const { return NodeHI.GetDat().GetOutDeg(); }

    int GetInNId(const int& EdgeN) const { return Graph->GetEdge(NodeHI.GetDat().GetInEId(EdgeN)).GetSrcNId(); }

    int GetOutNId(const int& EdgeN) const { return Graph->GetEdge(NodeHI.GetDat().GetOutEId(EdgeN)).GetDstNId(); }

    int GetNbrNId(const int& EdgeN) const { const TEdge& E = Graph->GetEdge(NodeHI.GetDat().GetNbrEId(EdgeN)); return GetId()==E.GetSrcNId() ? E.GetDstNId():E.GetSrcNId(); }

    bool IsInNId(const int& NId) const;

    bool IsOutNId(const int& NId) const;

    bool IsNbrNId(const int& NId) const { return IsOutNId(NId) || IsInNId(NId); }

    int GetInEId(const int& EdgeN) const { return NodeHI.GetDat().GetInEId(EdgeN); }

    int GetOutEId(const int& EdgeN) const { return NodeHI.GetDat().GetOutEId(EdgeN); }

    int GetNbrEId(const int& EdgeN) const { return NodeHI.GetDat().GetNbrEId(EdgeN); }

    bool IsInEId(const int& EId) const { return NodeHI.GetDat().IsInEId(EId); }

    bool IsOutEId(const int& EId) const { return NodeHI.GetDat().IsOutEId(EId); }

    bool IsNbrEId(const int& EId) const { return IsInEId(EId) || IsOutEId(EId); }

    void GetAttrNames(TStrV& Names) const { Graph->AttrNameNI(GetId(), Names); }

    void GetAttrVal(TStrV& Val) const { Graph->AttrValueNI(GetId(), Val); }

    void GetIntAttrNames(TStrV& Names) const { Graph->IntAttrNameNI(GetId(), Names); }

    void GetIntAttrVal(TIntV& Val) const { Graph->IntAttrValueNI(GetId(), Val); }

    void GetIntVAttrNames(TStrV& Names) const { Graph->IntVAttrNameNI(GetId(), Names); }

    void GetIntVAttrVal(TVec<TIntV>& Val) const { Graph->IntVAttrValueNI(GetId(), Val); }

    void GetStrAttrNames(TStrV& Names) const { Graph->StrAttrNameNI(GetId(), Names); }

    void GetStrAttrVal(TStrV& Val) const { Graph->StrAttrValueNI(GetId(), Val); }

    void GetFltAttrNames(TStrV& Names) const { Graph->FltAttrNameNI(GetId(), Names); }

    void GetFltAttrVal(TFltV& Val) const { Graph->FltAttrValueNI(GetId(), Val); }
    friend class TNEANet;
  };

  class TEdgeI {
  private:
    typedef THash<TInt, TEdge>::TIter THashIter;
    THashIter EdgeHI;
    const TNEANet *Graph;
  public:
    TEdgeI() : EdgeHI(), Graph(NULL) { }
    TEdgeI(const THashIter& EdgeHIter, const TNEANet *GraphPt) : EdgeHI(EdgeHIter), Graph(GraphPt) { }
    TEdgeI(const TEdgeI& EdgeI) : EdgeHI(EdgeI.EdgeHI), Graph(EdgeI.Graph) { }
    TEdgeI& operator = (const TEdgeI& EdgeI) { if (this!=&EdgeI) { EdgeHI=EdgeI.EdgeHI; Graph=EdgeI.Graph; }  return *this; }

    TEdgeI& operator++ (int) { EdgeHI++; return *this; }
    bool operator < (const TEdgeI& EdgeI) const { return EdgeHI < EdgeI.EdgeHI; }
    bool operator == (const TEdgeI& EdgeI) const { return EdgeHI == EdgeI.EdgeHI; }

    int GetId() const { return EdgeHI.GetDat().GetId(); }

    int GetSrcNId() const { return EdgeHI.GetDat().GetSrcNId(); }

    int GetDstNId() const { return EdgeHI.GetDat().GetDstNId(); }

    void GetAttrNames(TStrV& Names) const { Graph->AttrNameEI(GetId(), Names); }

    void GetAttrVal(TStrV& Val) const { Graph->AttrValueEI(GetId(), Val); }

    void GetIntAttrNames(TStrV& Names) const { Graph->IntAttrNameEI(GetId(), Names); }

    void GetIntAttrVal(TIntV& Val) const { Graph->IntAttrValueEI(GetId(), Val); }

    void GetIntVAttrNames(TStrV& Names) const { Graph->IntVAttrNameEI(GetId(), Names); }

    void GetIntVAttrVal(TVec<TIntV>& Val) const { Graph->IntVAttrValueEI(GetId(), Val); }

    void GetStrAttrNames(TStrV& Names) const { Graph->StrAttrNameEI(GetId(), Names); }

    void GetStrAttrVal(TStrV& Val) const { Graph->StrAttrValueEI(GetId(), Val); }

    void GetFltAttrNames(TStrV& Names) const { Graph->FltAttrNameEI(GetId(), Names); }

    void GetFltAttrVal(TFltV& Val) const { Graph->FltAttrValueEI(GetId(), Val); }
    friend class TNEANet;
  };

  class TAIntI {
  private:
    typedef TIntV::TIter TIntVecIter;
    TIntVecIter HI;
    bool isNode;
    TStr attr;
    const TNEANet *Graph;
  public:
    TAIntI() : HI(), attr(), Graph(NULL) { }
    TAIntI(const TIntVecIter& HIter, TStr attribute, bool isEdgeIter, const TNEANet* GraphPt) : HI(HIter), attr(), Graph(GraphPt) { isNode = !isEdgeIter; attr = attribute; }
    TAIntI(const TAIntI& I) : HI(I.HI), attr(I.attr), Graph(I.Graph) { isNode = I.isNode; }
    TAIntI& operator = (const TAIntI& I) { HI = I.HI; Graph=I.Graph; isNode = I.isNode; attr = I.attr; return *this; }
    bool operator < (const TAIntI& I) const { return HI < I.HI; }
    bool operator == (const TAIntI& I) const { return HI == I.HI; }

    TInt GetDat() const { return HI[0]; }

    bool IsDeleted() const { return isNode ? GetDat() == Graph->GetIntAttrDefaultN(attr) : GetDat() == Graph->GetIntAttrDefaultE(attr); };
    TAIntI& operator++(int) { HI++; return *this; }
    friend class TNEANet;
  };
  class TAIntVI {
  private:
    typedef TVec<TIntV>::TIter TIntVVecIter;
    TIntVVecIter HI;
    bool IsDense;
    typedef THash<TInt, TIntV>::TIter TIntHVecIter;
    TIntHVecIter HHI;
    bool isNode;
    TStr attr;
    const TNEANet *Graph;
  public:
    TAIntVI() : HI(), IsDense(), HHI(), attr(), Graph(NULL) { }
    TAIntVI(const TIntVVecIter& HIter, const TIntHVecIter& HHIter, TStr attribute, bool isEdgeIter, const TNEANet* GraphPt, bool is_dense) : HI(HIter), IsDense(is_dense), HHI(HHIter), attr(), Graph(GraphPt) {
      isNode = !isEdgeIter; attr = attribute;
    }
    TAIntVI(const TAIntVI& I) : HI(I.HI), IsDense(I.IsDense), HHI(I.HHI), attr(I.attr), Graph(I.Graph) { isNode = I.isNode; }
    TAIntVI& operator = (const TAIntVI& I) { HI = I.HI; HHI = I.HHI, Graph=I.Graph; isNode = I.isNode; attr = I.attr; return *this; }
    bool operator < (const TAIntVI& I) const { return HI == I.HI ? HHI < I.HHI : HI < I.HI; }
    bool operator == (const TAIntVI& I) const { return HI == I.HI && HHI == I.HHI; }

    TIntV GetDat() const { return IsDense? HI[0] : HHI.GetDat(); }
    TAIntVI& operator++(int) { if (IsDense) {HI++;} else {HHI++;} return *this; }
    friend class TNEANet;
  };

  class TAStrI {
  private:
    typedef TStrV::TIter TStrVecIter;
    TStrVecIter HI;
    bool isNode;
    TStr attr;
    const TNEANet *Graph;
  public:
    TAStrI() : HI(), attr(), Graph(NULL) { }
    TAStrI(const TStrVecIter& HIter, TStr attribute, bool isEdgeIter, const TNEANet* GraphPt) : HI(HIter), attr(), Graph(GraphPt) { isNode = !isEdgeIter; attr = attribute; }
    TAStrI(const TAStrI& I) : HI(I.HI), attr(I.attr), Graph(I.Graph) { isNode = I.isNode; }
    TAStrI& operator = (const TAStrI& I) { HI = I.HI; Graph=I.Graph; isNode = I.isNode; attr = I.attr; return *this; }
    bool operator < (const TAStrI& I) const { return HI < I.HI; }
    bool operator == (const TAStrI& I) const { return HI == I.HI; }

    TStr GetDat() const { return HI[0]; }

    bool IsDeleted() const { return isNode ? GetDat() == Graph->GetStrAttrDefaultN(attr) : GetDat() == Graph->GetStrAttrDefaultE(attr); };
    TAStrI& operator++(int) { HI++; return *this; }
    friend class TNEANet;
  };

  class TAFltI {
  private:
    typedef TFltV::TIter TFltVecIter;
    TFltVecIter HI;
    bool isNode;
    TStr attr;
    const TNEANet *Graph;
  public:
    TAFltI() : HI(), attr(), Graph(NULL) { }
    TAFltI(const TFltVecIter& HIter, TStr attribute, bool isEdgeIter, const TNEANet* GraphPt) : HI(HIter), attr(), Graph(GraphPt) { isNode = !isEdgeIter; attr = attribute; }
    TAFltI(const TAFltI& I) : HI(I.HI), attr(I.attr), Graph(I.Graph) { isNode = I.isNode; }
    TAFltI& operator = (const TAFltI& I) { HI = I.HI; Graph=I.Graph; isNode = I.isNode; attr = I.attr; return *this; }
    bool operator < (const TAFltI& I) const { return HI < I.HI; }
    bool operator == (const TAFltI& I) const { return HI == I.HI; }

    TFlt GetDat() const { return HI[0]; }

    bool IsDeleted() const { return isNode ? GetDat() == Graph->GetFltAttrDefaultN(attr) : GetDat() == Graph->GetFltAttrDefaultE(attr); };
    TAFltI& operator++(int) { HI++; return *this; }
    friend class TNEANet;
  };
protected:
  TNode& GetNode(const int& NId) { return NodeH.GetDat(NId); }
  const TNode& GetNode(const int& NId) const { return NodeH.GetDat(NId); }
  TEdge& GetEdge(const int& EId) { return EdgeH.GetDat(EId); }
  const TEdge& GetEdge(const int& EId) const { return EdgeH.GetDat(EId); }
  int AddAttributes(const int NId);
protected:

  TInt GetIntAttrDefaultN(const TStr& attribute) const { return IntDefaultsN.IsKey(attribute) ? IntDefaultsN.GetDat(attribute) : (TInt) TInt::Mn; }

  TStr GetStrAttrDefaultN(const TStr& attribute) const { return StrDefaultsN.IsKey(attribute) ? StrDefaultsN.GetDat(attribute) : (TStr) TStr::GetNullStr(); }

  TFlt GetFltAttrDefaultN(const TStr& attribute) const { return FltDefaultsN.IsKey(attribute) ? FltDefaultsN.GetDat(attribute) : (TFlt) TFlt::Mn; }

  TInt GetIntAttrDefaultE(const TStr& attribute) const { return IntDefaultsE.IsKey(attribute) ? IntDefaultsE.GetDat(attribute) : (TInt) TInt::Mn; }

  TStr GetStrAttrDefaultE(const TStr& attribute) const { return StrDefaultsE.IsKey(attribute) ? StrDefaultsE.GetDat(attribute) : (TStr) TStr::GetNullStr(); }

  TFlt GetFltAttrDefaultE(const TStr& attribute) const { return FltDefaultsE.IsKey(attribute) ? FltDefaultsE.GetDat(attribute) : (TFlt) TFlt::Mn; }
public:
  TCRef CRef;
protected:
  TInt MxNId, MxEId;
  THash<TInt, TNode> NodeH;
  THash<TInt, TEdge> EdgeH;

  TStrIntPrH KeyToIndexTypeN, KeyToIndexTypeE;

  THash<TStr, TBool> KeyToDenseN, KeyToDenseE;
  THash<TStr, TInt> IntDefaultsN, IntDefaultsE;
  THash<TStr, TStr> StrDefaultsN, StrDefaultsE;
  THash<TStr, TFlt> FltDefaultsN, FltDefaultsE;
  TVec<TIntV> VecOfIntVecsN, VecOfIntVecsE;
  TVec<TStrV> VecOfStrVecsN, VecOfStrVecsE;
  TVec<TFltV> VecOfFltVecsN, VecOfFltVecsE;
  TVec<TVec<TIntV> > VecOfIntVecVecsN, VecOfIntVecVecsE;
  TVec<THash<TInt, TIntV> > VecOfIntHashVecsN, VecOfIntHashVecsE;
  enum { IntType, StrType, FltType, IntVType };
  TAttr SAttrN;
  TAttr SAttrE;
private:
  class LoadTNodeFunctor {
  public:
    LoadTNodeFunctor() {}
    void operator() (TNode* n, TShMIn& ShMIn) { n->LoadShM(ShMIn);}
  };
  class LoadVecFunctor {
  public:
    LoadVecFunctor() {}
    template<typename TElem>
    void operator() (TVec<TElem>* n, TShMIn& ShMIn) {
      n->LoadShM(ShMIn);
    }
  };
  class LoadVecOfVecFunctor {
  public:
    LoadVecOfVecFunctor() {}
    template<typename TElem>
    void operator() (TVec<TVec<TElem> >* n, TShMIn& ShMIn) {
      LoadVecFunctor f;
      n->LoadShM(ShMIn, f);
    }
  };
  class LoadHashOfVecFunctor {
  public:
    LoadHashOfVecFunctor() {}
    template<typename TElem>
    void operator() (THash<TInt, TVec<TElem> >* n, TShMIn& ShMIn) {
      LoadVecFunctor f;
      n->LoadShM(ShMIn, f);
    }
  };
protected:

  TInt CheckDenseOrSparseN(const TStr& attr) const {
    if (!KeyToDenseN.IsKey(attr)) return -1;
    if (KeyToDenseN.GetDat(attr)) return 1;
    return 0;
  }
  TInt CheckDenseOrSparseE(const TStr& attr) const {
    if (!KeyToDenseE.IsKey(attr)) return -1;
    if (KeyToDenseE.GetDat(attr)) return 1;
    return 0;
  }
  
public:
  TNEANet() : CRef(), MxNId(0), MxEId(0), NodeH(), EdgeH(),
    KeyToIndexTypeN(), KeyToIndexTypeE(), KeyToDenseN(), KeyToDenseE(), IntDefaultsN(), IntDefaultsE(),
    StrDefaultsN(), StrDefaultsE(), FltDefaultsN(), FltDefaultsE(),
    VecOfIntVecsN(), VecOfIntVecsE(), VecOfStrVecsN(), VecOfStrVecsE(),
    VecOfFltVecsN(), VecOfFltVecsE(),  VecOfIntVecVecsN(), VecOfIntVecVecsE(),
    VecOfIntHashVecsN(), VecOfIntHashVecsE(), SAttrN(), SAttrE(){ }

  explicit TNEANet(const int& Nodes, const int& Edges) : CRef(),
    MxNId(0), MxEId(0), NodeH(), EdgeH(), KeyToIndexTypeN(), KeyToIndexTypeE(), KeyToDenseN(), KeyToDenseE(),
    IntDefaultsN(), IntDefaultsE(), StrDefaultsN(), StrDefaultsE(),
    FltDefaultsN(), FltDefaultsE(), VecOfIntVecsN(), VecOfIntVecsE(),
    VecOfStrVecsN(), VecOfStrVecsE(), VecOfFltVecsN(), VecOfFltVecsE(), VecOfIntVecVecsN(), VecOfIntVecVecsE(),
    VecOfIntHashVecsN(), VecOfIntHashVecsE(), SAttrN(), SAttrE()
    { Reserve(Nodes, Edges); }
  TNEANet(const TNEANet& Graph) : MxNId(Graph.MxNId), MxEId(Graph.MxEId),
    NodeH(Graph.NodeH), EdgeH(Graph.EdgeH), KeyToIndexTypeN(), KeyToIndexTypeE(), KeyToDenseN(), KeyToDenseE(),
    IntDefaultsN(), IntDefaultsE(), StrDefaultsN(), StrDefaultsE(),
    FltDefaultsN(), FltDefaultsE(), VecOfIntVecsN(), VecOfIntVecsE(),
    VecOfStrVecsN(), VecOfStrVecsE(), VecOfFltVecsN(), VecOfFltVecsE(), VecOfIntVecVecsN(), VecOfIntVecVecsE(),
    VecOfIntHashVecsN(), VecOfIntHashVecsE(), SAttrN(), SAttrE() { }

  TNEANet(TSIn& SIn) : MxNId(SIn), MxEId(SIn), NodeH(SIn), EdgeH(SIn),
    KeyToIndexTypeN(SIn), KeyToIndexTypeE(SIn), KeyToDenseN(SIn), KeyToDenseE(SIn), IntDefaultsN(SIn), IntDefaultsE(SIn),
    StrDefaultsN(SIn), StrDefaultsE(SIn), FltDefaultsN(SIn), FltDefaultsE(SIn),
    VecOfIntVecsN(SIn), VecOfIntVecsE(SIn), VecOfStrVecsN(SIn),VecOfStrVecsE(SIn),
    VecOfFltVecsN(SIn), VecOfFltVecsE(SIn), VecOfIntVecVecsN(SIn), VecOfIntVecVecsE(SIn), VecOfIntHashVecsN(SIn), VecOfIntHashVecsE(SIn),
    SAttrN(SIn), SAttrE(SIn) { }
protected:
  TNEANet(const TNEANet& Graph, bool modeSubGraph) : MxNId(Graph.MxNId), MxEId(Graph.MxEId),
    NodeH(Graph.NodeH), EdgeH(Graph.EdgeH), KeyToIndexTypeN(), KeyToIndexTypeE(Graph.KeyToIndexTypeE), KeyToDenseN(), KeyToDenseE(Graph.KeyToDenseE),
    IntDefaultsN(Graph.IntDefaultsN), IntDefaultsE(Graph.IntDefaultsE), StrDefaultsN(Graph.StrDefaultsN), StrDefaultsE(Graph.StrDefaultsE),
    FltDefaultsN(Graph.FltDefaultsN), FltDefaultsE(Graph.FltDefaultsE), VecOfIntVecsN(Graph.VecOfIntVecsN), VecOfIntVecsE(Graph.VecOfIntVecsE),
    VecOfStrVecsN(Graph.VecOfStrVecsN), VecOfStrVecsE(Graph.VecOfStrVecsE), VecOfFltVecsN(Graph.VecOfFltVecsN), VecOfFltVecsE(Graph.VecOfFltVecsE),
    VecOfIntVecVecsN(), VecOfIntVecVecsE(Graph.VecOfIntVecVecsE), VecOfIntHashVecsN(), VecOfIntHashVecsE(Graph.VecOfIntHashVecsE) { }
  TNEANet(bool copyAll, const TNEANet& Graph) : MxNId(Graph.MxNId), MxEId(Graph.MxEId),
    NodeH(Graph.NodeH), EdgeH(Graph.EdgeH), KeyToIndexTypeN(Graph.KeyToIndexTypeN), KeyToIndexTypeE(Graph.KeyToIndexTypeE), KeyToDenseN(Graph.KeyToDenseN), KeyToDenseE(Graph.KeyToDenseE),
    IntDefaultsN(Graph.IntDefaultsN), IntDefaultsE(Graph.IntDefaultsE), StrDefaultsN(Graph.StrDefaultsN), StrDefaultsE(Graph.StrDefaultsE),
    FltDefaultsN(Graph.FltDefaultsN), FltDefaultsE(Graph.FltDefaultsE), VecOfIntVecsN(Graph.VecOfIntVecsN), VecOfIntVecsE(Graph.VecOfIntVecsE),
    VecOfStrVecsN(Graph.VecOfStrVecsN), VecOfStrVecsE(Graph.VecOfStrVecsE), VecOfFltVecsN(Graph.VecOfFltVecsN), VecOfFltVecsE(Graph.VecOfFltVecsE),
    VecOfIntVecVecsN(Graph.VecOfIntVecVecsN), VecOfIntVecVecsE(Graph.VecOfIntVecVecsE), VecOfIntHashVecsN(Graph.VecOfIntHashVecsN), VecOfIntHashVecsE(Graph.VecOfIntHashVecsE), SAttrN(Graph.SAttrN), SAttrE(Graph.SAttrE) { }

public:

  void Save(TSOut& SOut) const {
    MxNId.Save(SOut); MxEId.Save(SOut); NodeH.Save(SOut); EdgeH.Save(SOut);
    KeyToIndexTypeN.Save(SOut); KeyToIndexTypeE.Save(SOut);
    KeyToDenseN.Save(SOut); KeyToDenseE.Save(SOut);
    IntDefaultsN.Save(SOut); IntDefaultsE.Save(SOut);
    StrDefaultsN.Save(SOut); StrDefaultsE.Save(SOut);
    FltDefaultsN.Save(SOut); FltDefaultsE.Save(SOut);
    VecOfIntVecsN.Save(SOut); VecOfIntVecsE.Save(SOut);
    VecOfStrVecsN.Save(SOut); VecOfStrVecsE.Save(SOut);
    VecOfFltVecsN.Save(SOut); VecOfFltVecsE.Save(SOut);
    VecOfIntVecVecsN.Save(SOut); VecOfIntVecVecsE.Save(SOut);
    VecOfIntHashVecsN.Save(SOut); VecOfIntHashVecsE.Save(SOut); 
    SAttrN.Save(SOut); SAttrE.Save(SOut); }

  void Save_V1(TSOut& SOut) const {
    MxNId.Save(SOut); MxEId.Save(SOut); NodeH.Save(SOut); EdgeH.Save(SOut);
    KeyToIndexTypeN.Save(SOut); KeyToIndexTypeE.Save(SOut);
    IntDefaultsN.Save(SOut); IntDefaultsE.Save(SOut);
    StrDefaultsN.Save(SOut); StrDefaultsE.Save(SOut);
    FltDefaultsN.Save(SOut); FltDefaultsE.Save(SOut);
    VecOfIntVecsN.Save(SOut); VecOfIntVecsE.Save(SOut);
    VecOfStrVecsN.Save(SOut); VecOfStrVecsE.Save(SOut);
    VecOfFltVecsN.Save(SOut); VecOfFltVecsE.Save(SOut); }

  void Save_V2(TSOut& SOut) const {
    MxNId.Save(SOut); MxEId.Save(SOut); NodeH.Save(SOut); EdgeH.Save(SOut);
    KeyToIndexTypeN.Save(SOut); KeyToIndexTypeE.Save(SOut);
    IntDefaultsN.Save(SOut); IntDefaultsE.Save(SOut);
    StrDefaultsN.Save(SOut); StrDefaultsE.Save(SOut);
    FltDefaultsN.Save(SOut); FltDefaultsE.Save(SOut);
    VecOfIntVecsN.Save(SOut); VecOfIntVecsE.Save(SOut);
    VecOfStrVecsN.Save(SOut); VecOfStrVecsE.Save(SOut);
    VecOfFltVecsN.Save(SOut); VecOfFltVecsE.Save(SOut);
    VecOfIntVecVecsN.Save(SOut); VecOfIntVecVecsE.Save(SOut); 
    SAttrN.Save(SOut); SAttrE.Save(SOut); }

  static PNEANet New() { return PNEANet(new TNEANet()); }

  static PNEANet New(const int& Nodes, const int& Edges) { return PNEANet(new TNEANet(Nodes, Edges)); }

  static PNEANet Load(TSIn& SIn) { return PNEANet(new TNEANet(SIn)); }

  static PNEANet Load_V1(TSIn& SIn) {
    PNEANet Graph = PNEANet(new TNEANet());
    Graph->MxNId.Load(SIn); Graph->MxEId.Load(SIn);
    Graph->NodeH.Load(SIn); Graph->EdgeH.Load(SIn);
    Graph->KeyToIndexTypeN.Load(SIn); Graph->KeyToIndexTypeE.Load(SIn);
    Graph->IntDefaultsN.Load(SIn); Graph->IntDefaultsE.Load(SIn);
    Graph->StrDefaultsN.Load(SIn); Graph->StrDefaultsE.Load(SIn);
    Graph->FltDefaultsN.Load(SIn); Graph->FltDefaultsE.Load(SIn);
    Graph->VecOfIntVecsN.Load(SIn); Graph->VecOfIntVecsE.Load(SIn);
    Graph->VecOfStrVecsN.Load(SIn); Graph->VecOfStrVecsE.Load(SIn);
    Graph->VecOfFltVecsN.Load(SIn); Graph->VecOfFltVecsE.Load(SIn);
    return Graph;
  }

  static PNEANet Load_V2(TSIn& SIn) {
    PNEANet Graph = PNEANet(new TNEANet());
    Graph->MxNId.Load(SIn); Graph->MxEId.Load(SIn);
    Graph->NodeH.Load(SIn); Graph->EdgeH.Load(SIn);
    Graph->KeyToIndexTypeN.Load(SIn); Graph->KeyToIndexTypeE.Load(SIn);
    Graph->IntDefaultsN.Load(SIn); Graph->IntDefaultsE.Load(SIn);
    Graph->StrDefaultsN.Load(SIn); Graph->StrDefaultsE.Load(SIn);
    Graph->FltDefaultsN.Load(SIn); Graph->FltDefaultsE.Load(SIn);
    Graph->VecOfIntVecsN.Load(SIn); Graph->VecOfIntVecsE.Load(SIn);
    Graph->VecOfStrVecsN.Load(SIn); Graph->VecOfStrVecsE.Load(SIn);
    Graph->VecOfFltVecsN.Load(SIn); Graph->VecOfFltVecsE.Load(SIn);
    Graph->VecOfIntVecVecsN.Load(SIn); Graph->VecOfIntVecVecsE.Load(SIn);
    Graph->SAttrN.Load(SIn); Graph->SAttrE.Load(SIn);
    return Graph;
  }

  void LoadNetworkShM(TShMIn& ShMIn);

  static PNEANet LoadShM(TShMIn& ShMIn) {
    TNEANet* Network = new TNEANet();
    Network->LoadNetworkShM(ShMIn);
    return PNEANet(Network);
  }
  void ConvertToSparse() {
    TInt VecLength = VecOfIntVecVecsN.Len();
    THash<TStr, TIntPr>::TIter iter;
    if (VecLength != 0) {
      VecOfIntHashVecsN = TVec<THash<TInt, TIntV> >(VecLength);
      for (iter = KeyToIndexTypeN.BegI(); !iter.IsEnd(); iter=iter.Next()) {
        if (iter.GetDat().Val1 == IntVType) {
          TStr attribute = iter.GetKey();
          TInt index = iter.GetDat().Val2();
          for (int i=0; i<VecOfIntVecVecsN[index].Len(); i++) {
            if(VecOfIntVecVecsN[index][i].Len() > 0) {
              VecOfIntHashVecsN[index].AddDat(TInt(i), VecOfIntVecVecsN[index][i]);
            }
          }
          KeyToDenseN.AddDat(attribute, TBool(false));
        }
      }
    }
    VecOfIntVecVecsN.Clr();
    VecLength = VecOfIntVecVecsE.Len();
    if (VecLength != 0) {
      VecOfIntHashVecsE = TVec<THash<TInt, TIntV> >(VecLength);
      for (iter = KeyToIndexTypeE.BegI(); !iter.IsEnd(); iter=iter.Next()) {
        if (iter.GetDat().Val1 == IntVType) {
          TStr attribute = iter.GetKey();
          TInt index = iter.GetDat().Val2();
          for (int i=0; i<VecOfIntVecVecsE[index].Len(); i++) {
            if(VecOfIntVecVecsE[index][i].Len() > 0) {
              VecOfIntHashVecsE[index].AddDat(TInt(i), VecOfIntVecVecsE[index][i]);
            }
          }
          KeyToDenseE.AddDat(attribute, TBool(false));
        }
      }
    }
    VecOfIntVecVecsE.Clr();
  }

  bool HasFlag(const TGraphFlag& Flag) const;
  
  TNEANet& operator = (const TNEANet& Graph) { if (this!=&Graph) {
    MxNId=Graph.MxNId; MxEId=Graph.MxEId; NodeH=Graph.NodeH; EdgeH=Graph.EdgeH; }
    return *this; }

  int GetNodes() const { return NodeH.Len(); }

  int AddNode(int NId = -1);

  int AddNodeUnchecked(int NId = -1);

  int AddNode(const TNodeI& NodeI) { return AddNode(NodeI.GetId()); }

  virtual void DelNode(const int& NId);

  void DelNode(const TNode& NodeI) { DelNode(NodeI.GetId()); }

  bool IsNode(const int& NId) const { return NodeH.IsKey(NId); }

  TNodeI BegNI() const { return TNodeI(NodeH.BegI(), this); }

  TNodeI EndNI() const { return TNodeI(NodeH.EndI(), this); }

  TNodeI GetNI(const int& NId) const { return TNodeI(NodeH.GetI(NId), this); }

  TAIntI BegNAIntI(const TStr& attr) const {
    return TAIntI(VecOfIntVecsN[KeyToIndexTypeN.GetDat(attr).Val2].BegI(), attr, false, this); }

  TAIntI EndNAIntI(const TStr& attr) const {
    return TAIntI(VecOfIntVecsN[KeyToIndexTypeN.GetDat(attr).Val2].EndI(), attr, false, this); }

  TAIntI GetNAIntI(const TStr& attr, const int& NId) const {
    return TAIntI(VecOfIntVecsN[KeyToIndexTypeN.GetDat(attr).Val2].GetI(NodeH.GetKeyId(NId)), attr, false, this); }

  TAIntVI BegNAIntVI(const TStr& attr) const {
    TVec<TIntV>::TIter HI = NULL;
    THash<TInt, TIntV>::TIter HHI;
    TInt location = CheckDenseOrSparseN(attr);
    TBool IsDense = true;
    if (location != -1) {
      TInt index = KeyToIndexTypeN.GetDat(attr).Val2;
      if (location == 1) {
        HI = VecOfIntVecVecsN[index].BegI();
      } else {
        IsDense = false;
        HHI = VecOfIntHashVecsN[index].BegI();
      }
    }
    return TAIntVI(HI, HHI, attr, false, this, IsDense);
  }

  TAIntVI EndNAIntVI(const TStr& attr) const {
    TVec<TIntV>::TIter HI = NULL;
    THash<TInt, TIntV>::TIter HHI;
    TInt location = CheckDenseOrSparseN(attr);
    TBool IsDense = true;
    if (location != -1) {
      TInt index = KeyToIndexTypeN.GetDat(attr).Val2;
      if (location == 1) {
        HI = VecOfIntVecVecsN[index].EndI();
      } else {
        IsDense = false;
        HHI = VecOfIntHashVecsN[index].EndI();
      }
    }
    return TAIntVI(HI, HHI, attr, false, this, IsDense);
  }

  TAIntVI GetNAIntVI(const TStr& attr, const int& NId) const {
    TVec<TIntV>::TIter HI = NULL;
    THash<TInt, TIntV>::TIter HHI;
    TInt location = CheckDenseOrSparseN(attr);
    TBool IsDense = true;
    if (location != -1) {
      TInt index = KeyToIndexTypeN.GetDat(attr).Val2;
      if (location == 1) {
        HI = VecOfIntVecVecsN[index].GetI(NodeH.GetKeyId(NId));
      } else {
        IsDense = false;
        HHI = VecOfIntHashVecsN[index].GetI(NodeH.GetKeyId(NId));
      }
    }
    return TAIntVI(HI, HHI, attr, false, this, IsDense);
  }

  TAStrI BegNAStrI(const TStr& attr) const {
    return TAStrI(VecOfStrVecsN[KeyToIndexTypeN.GetDat(attr).Val2].BegI(), attr, false, this); }

  TAStrI EndNAStrI(const TStr& attr) const {
    return TAStrI(VecOfStrVecsN[KeyToIndexTypeN.GetDat(attr).Val2].EndI(), attr, false, this); }

  TAStrI GetNAStrI(const TStr& attr, const int& NId) const {
    return TAStrI(VecOfStrVecsN[KeyToIndexTypeN.GetDat(attr).Val2].GetI(NodeH.GetKeyId(NId)), attr, false, this); }

  TAFltI BegNAFltI(const TStr& attr) const {
    return TAFltI(VecOfFltVecsN[KeyToIndexTypeN.GetDat(attr).Val2].BegI(), attr, false, this); }

  TAFltI EndNAFltI(const TStr& attr) const {
    return TAFltI(VecOfFltVecsN[KeyToIndexTypeN.GetDat(attr).Val2].EndI(), attr, false, this); }

  TAFltI GetNAFltI(const TStr& attr, const int& NId) const {
    return TAFltI(VecOfFltVecsN[KeyToIndexTypeN.GetDat(attr).Val2].GetI(NodeH.GetKeyId(NId)), attr, false, this); }

  void AttrNameNI(const TInt& NId, TStrV& Names) const {
    AttrNameNI(NId, KeyToIndexTypeN.BegI(), Names);}
  void AttrNameNI(const TInt& NId, TStrIntPrH::TIter NodeHI, TStrV& Names) const;

  void AttrValueNI(const TInt& NId, TStrV& Values) const {
    AttrValueNI(NId, KeyToIndexTypeN.BegI(), Values);}
  void AttrValueNI(const TInt& NId, TStrIntPrH::TIter NodeHI, TStrV& Values) const;

  void IntAttrNameNI(const TInt& NId, TStrV& Names) const {
    IntAttrNameNI(NId, KeyToIndexTypeN.BegI(), Names);}
  void IntAttrNameNI(const TInt& NId, TStrIntPrH::TIter NodeHI, TStrV& Names) const;

  void IntAttrValueNI(const TInt& NId, TIntV& Values) const {
    IntAttrValueNI(NId, KeyToIndexTypeN.BegI(), Values);}
  void IntAttrValueNI(const TInt& NId, TStrIntPrH::TIter NodeHI, TIntV& Values) const;

  void IntVAttrNameNI(const TInt& NId, TStrV& Names) const {
    IntVAttrNameNI(NId, KeyToIndexTypeN.BegI(), Names);}
  void IntVAttrNameNI(const TInt& NId, TStrIntPrH::TIter NodeHI, TStrV& Names) const;

  void IntVAttrValueNI(const TInt& NId, TVec<TIntV>& Values) const {
    IntVAttrValueNI(NId, KeyToIndexTypeN.BegI(), Values);}
  void IntVAttrValueNI(const TInt& NId, TStrIntPrH::TIter NodeHI, TVec<TIntV>& Values) const;

  void StrAttrNameNI(const TInt& NId, TStrV& Names) const {
    StrAttrNameNI(NId, KeyToIndexTypeN.BegI(), Names);}
  void StrAttrNameNI(const TInt& NId, TStrIntPrH::TIter NodeHI, TStrV& Names) const;

  void StrAttrValueNI(const TInt& NId, TStrV& Values) const {
    StrAttrValueNI(NId, KeyToIndexTypeN.BegI(), Values);}
  void StrAttrValueNI(const TInt& NId, TStrIntPrH::TIter NodeHI, TStrV& Values) const;

  void FltAttrNameNI(const TInt& NId, TStrV& Names) const {
    FltAttrNameNI(NId, KeyToIndexTypeN.BegI(), Names);}
  void FltAttrNameNI(const TInt& NId, TStrIntPrH::TIter NodeHI, TStrV& Names) const;

  void FltAttrValueNI(const TInt& NId, TFltV& Values) const {
    FltAttrValueNI(NId, KeyToIndexTypeN.BegI(), Values);}
  void FltAttrValueNI(const TInt& NId, TStrIntPrH::TIter NodeHI, TFltV& Values) const;

  void AttrNameEI(const TInt& EId, TStrV& Names) const {
    AttrNameEI(EId, KeyToIndexTypeE.BegI(), Names);}
  void AttrNameEI(const TInt& EId, TStrIntPrH::TIter EdgeHI, TStrV& Names) const;

  void AttrValueEI(const TInt& EId, TStrV& Values) const {
    AttrValueEI(EId, KeyToIndexTypeE.BegI(), Values);}
  void AttrValueEI(const TInt& EId, TStrIntPrH::TIter EdgeHI, TStrV& Values) const;

  void IntAttrNameEI(const TInt& EId, TStrV& Names) const {
    IntAttrNameEI(EId, KeyToIndexTypeE.BegI(), Names);}
  void IntAttrNameEI(const TInt& EId, TStrIntPrH::TIter EdgeHI, TStrV& Names) const;

  void IntAttrValueEI(const TInt& EId, TIntV& Values) const {
    IntAttrValueEI(EId, KeyToIndexTypeE.BegI(), Values);}
  void IntAttrValueEI(const TInt& EId, TStrIntPrH::TIter EdgeHI, TIntV& Values) const;

  void IntVAttrNameEI(const TInt& EId, TStrV& Names) const {
    IntVAttrNameEI(EId, KeyToIndexTypeE.BegI(), Names);}
  void IntVAttrNameEI(const TInt& EId, TStrIntPrH::TIter EdgeHI, TStrV& Names) const;

  void IntVAttrValueEI(const TInt& EId, TVec<TIntV>& Values) const {
    IntVAttrValueEI(EId, KeyToIndexTypeE.BegI(), Values);}
  void IntVAttrValueEI(const TInt& EId, TStrIntPrH::TIter EdgeHI, TVec<TIntV>& Values) const;

  void StrAttrNameEI(const TInt& EId, TStrV& Names) const {
    StrAttrNameEI(EId, KeyToIndexTypeE.BegI(), Names);}
  void StrAttrNameEI(const TInt& EId, TStrIntPrH::TIter EdgeHI, TStrV& Names) const;

  void StrAttrValueEI(const TInt& EId, TStrV& Values) const {
    StrAttrValueEI(EId, KeyToIndexTypeE.BegI(), Values);}
  void StrAttrValueEI(const TInt& EId, TStrIntPrH::TIter EdgeHI, TStrV& Values) const;

  void FltAttrNameEI(const TInt& EId, TStrV& Names) const {
    FltAttrNameEI(EId, KeyToIndexTypeE.BegI(), Names);}
  void FltAttrNameEI(const TInt& EId, TStrIntPrH::TIter EdgeHI, TStrV& Names) const;

  void FltAttrValueEI(const TInt& EId, TFltV& Values) const {
    FltAttrValueEI(EId, KeyToIndexTypeE.BegI(), Values);}
  void FltAttrValueEI(const TInt& EId, TStrIntPrH::TIter EdgeHI, TFltV& Values) const;

  TAIntI BegEAIntI(const TStr& attr) const {
    return TAIntI(VecOfIntVecsE[KeyToIndexTypeE.GetDat(attr).Val2].BegI(), attr, true, this);
  }

  TAIntI EndEAIntI(const TStr& attr) const {
    return TAIntI(VecOfIntVecsE[KeyToIndexTypeE.GetDat(attr).Val2].EndI(), attr, true, this);
  }

  TAIntI GetEAIntI(const TStr& attr, const int& EId) const {
    return TAIntI(VecOfIntVecsE[KeyToIndexTypeE.GetDat(attr).Val2].GetI(EdgeH.GetKeyId(EId)), attr, true, this);
  }

  TAIntVI BegEAIntVI(const TStr& attr) const {
    TVec<TIntV>::TIter HI = NULL;
    THash<TInt, TIntV>::TIter HHI;
    TInt location = CheckDenseOrSparseE(attr);
    TBool IsDense = true;
    if (location != -1) {
      TInt index = KeyToIndexTypeE.GetDat(attr).Val2;
      if (location == 1) {
        HI = VecOfIntVecVecsE[index].BegI();
      } else {
        IsDense = false;
        HHI = VecOfIntHashVecsE[index].BegI();
      }
    }
    return TAIntVI(HI, HHI, attr, true, this, IsDense);
  }

  TAIntVI EndEAIntVI(const TStr& attr) const {
    TVec<TIntV>::TIter HI = NULL;
    THash<TInt, TIntV>::TIter HHI;
    TInt location = CheckDenseOrSparseE(attr);
    TBool IsDense = true;
    if (location != -1) {
      TInt index = KeyToIndexTypeE.GetDat(attr).Val2;
      if (location == 1) {
        HI = VecOfIntVecVecsE[index].EndI();
      } else {
        IsDense = false;
        HHI = VecOfIntHashVecsE[index].EndI();
      }
    }
    return TAIntVI(HI, HHI, attr, true, this, IsDense);
  }

  TAIntVI GetEAIntVI(const TStr& attr, const int& EId) const {
    TVec<TIntV>::TIter HI = NULL;
    THash<TInt, TIntV>::TIter HHI;
    TInt location = CheckDenseOrSparseE(attr);
    TBool IsDense = true;
    if (location != -1) {
      TInt index = KeyToIndexTypeE.GetDat(attr).Val2;
      if (location == 1) {
        HI = VecOfIntVecVecsE[index].GetI(EdgeH.GetKeyId(EId));
      } else {
        IsDense = false;
        HHI = VecOfIntHashVecsE[index].GetI(EdgeH.GetKeyId(EId));
      }
    }
    return TAIntVI(HI, HHI, attr, true, this, IsDense);
  }

  TAStrI BegEAStrI(const TStr& attr) const {
    return TAStrI(VecOfStrVecsE[KeyToIndexTypeE.GetDat(attr).Val2].BegI(), attr, true, this);   }

  TAStrI EndEAStrI(const TStr& attr) const {
    return TAStrI(VecOfStrVecsE[KeyToIndexTypeE.GetDat(attr).Val2].EndI(), attr, true, this);
  }

  TAStrI GetEAStrI(const TStr& attr, const int& EId) const {
    return TAStrI(VecOfStrVecsE[KeyToIndexTypeE.GetDat(attr).Val2].GetI(EdgeH.GetKeyId(EId)), attr, true, this);
  }

  TAFltI BegEAFltI(const TStr& attr) const {
    return TAFltI(VecOfFltVecsE[KeyToIndexTypeE.GetDat(attr).Val2].BegI(), attr, true, this);
  }

  TAFltI EndEAFltI(const TStr& attr) const {
    return TAFltI(VecOfFltVecsE[KeyToIndexTypeE.GetDat(attr).Val2].EndI(), attr, true, this);
  }

  TAFltI GetEAFltI(const TStr& attr, const int& EId) const {
    return TAFltI(VecOfFltVecsE[KeyToIndexTypeE.GetDat(attr).Val2].GetI(EdgeH.GetKeyId(EId)), attr, true, this);
  }

  int GetMxNId() const { return MxNId; }

  int GetMxEId() const { return MxEId; }

  int GetEdges() const { return EdgeH.Len(); }

  int AddEdge(const int& SrcNId, const int& DstNId, int EId  = -1);

  int AddEdge(const TEdgeI& EdgeI) { return AddEdge(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), EdgeI.GetId()); }

  void DelEdge(const int& EId);

  void DelEdge(const int& SrcNId, const int& DstNId, const bool& IsDir = true);

  bool IsEdge(const int& EId) const { return EdgeH.IsKey(EId); }

  bool IsEdge(const int& SrcNId, const int& DstNId, const bool& IsDir = true) const { int EId; return IsEdge(SrcNId, DstNId, EId, IsDir); }

  bool IsEdge(const int& SrcNId, const int& DstNId, int& EId, const bool& IsDir = true) const;

  int GetEId(const int& SrcNId, const int& DstNId) const { int EId; return IsEdge(SrcNId, DstNId, EId)?EId:-1; }

  TEdgeI BegEI() const { return TEdgeI(EdgeH.BegI(), this); }

  TEdgeI EndEI() const { return TEdgeI(EdgeH.EndI(), this); }

  TEdgeI GetEI(const int& EId) const { return TEdgeI(EdgeH.GetI(EId), this); }

  TEdgeI GetEI(const int& SrcNId, const int& DstNId) const { return GetEI(GetEId(SrcNId, DstNId)); }

  int GetRndNId(TRnd& Rnd=TInt::Rnd) { return NodeH.GetKey(NodeH.GetRndKeyId(Rnd, 0.8)); }

  TNodeI GetRndNI(TRnd& Rnd=TInt::Rnd) { return GetNI(GetRndNId(Rnd)); }

  int GetRndEId(TRnd& Rnd=TInt::Rnd) { return EdgeH.GetKey(EdgeH.GetRndKeyId(Rnd, 0.8)); }

  TEdgeI GetRndEI(TRnd& Rnd=TInt::Rnd) { return GetEI(GetRndEId(Rnd)); }

  void GetNIdV(TIntV& NIdV) const;

  void GetEIdV(TIntV& EIdV) const;

  bool Empty() const { return GetNodes()==0; }

  void Clr() { MxNId=0; MxEId=0; NodeH.Clr(); EdgeH.Clr();
    KeyToIndexTypeN.Clr(); KeyToIndexTypeE.Clr(); IntDefaultsN.Clr(); IntDefaultsE.Clr();
    StrDefaultsN.Clr(); StrDefaultsE.Clr(); FltDefaultsN.Clr(); FltDefaultsE.Clr();
    VecOfIntVecsN.Clr(); VecOfIntVecsE.Clr(); VecOfStrVecsN.Clr(); VecOfStrVecsE.Clr();
    VecOfFltVecsN.Clr(); VecOfFltVecsE.Clr(); VecOfIntVecVecsN.Clr(); VecOfIntVecVecsE.Clr(); 
    SAttrN.Clr(); SAttrE.Clr();}

  void Reserve(const int& Nodes, const int& Edges) {
    if (Nodes>0) { NodeH.Gen(Nodes/2); } if (Edges>0) { EdgeH.Gen(Edges/2); } }

  void Defrag(const bool& OnlyNodeLinks=false);

  bool IsOk(const bool& ThrowExcept=true) const;

  void Dump(FILE *OutF=stdout) const;

  int AddIntAttrDatN(const TNodeI& NodeI, const TInt& value, const TStr& attr) { return AddIntAttrDatN(NodeI.GetId(), value, attr); }
  int AddIntAttrDatN(const int& NId, const TInt& value, const TStr& attr);

  int AddStrAttrDatN(const TNodeI& NodeI, const TStr& value, const TStr& attr) { return AddStrAttrDatN(NodeI.GetId(), value, attr); }
  int AddStrAttrDatN(const int& NId, const TStr& value, const TStr& attr);

  int AddFltAttrDatN(const TNodeI& NodeI, const TFlt& value, const TStr& attr) { return AddFltAttrDatN(NodeI.GetId(), value, attr); }
  int AddFltAttrDatN(const int& NId, const TFlt& value, const TStr& attr);

  int AddIntVAttrDatN(const TNodeI& NodeI, const TIntV& value, const TStr& attr) { return AddIntVAttrDatN(NodeI.GetId(), value, attr); }
  int AddIntVAttrDatN(const int& NId, const TIntV& value, const TStr& attr, TBool UseDense=true);

  int AppendIntVAttrDatN(const TNodeI& NodeI, const TInt& value, const TStr& attr) { return AppendIntVAttrDatN(NodeI.GetId(), value, attr); }
  int AppendIntVAttrDatN(const int& NId, const TInt& value, const TStr& attr, TBool UseDense=true);

  int DelFromIntVAttrDatN(const TNodeI& NodeI, const TInt& value, const TStr& attr) { return DelFromIntVAttrDatN(NodeI.GetId(), value, attr); }
  int DelFromIntVAttrDatN(const int& NId, const TInt& value, const TStr& attr);

  int AddIntAttrDatE(const TEdgeI& EdgeI, const TInt& value, const TStr& attr) { return AddIntAttrDatE(EdgeI.GetId(), value, attr); }
  int AddIntAttrDatE(const int& EId, const TInt& value, const TStr& attr);

  int AddStrAttrDatE(const TEdgeI& EdgeI, const TStr& value, const TStr& attr) { return AddStrAttrDatE(EdgeI.GetId(), value, attr); }
  int AddStrAttrDatE(const int& EId, const TStr& value, const TStr& attr);

  int AddFltAttrDatE(const TEdgeI& EdgeI, const TFlt& value, const TStr& attr) { return AddFltAttrDatE(EdgeI.GetId(), value, attr); }
  int AddFltAttrDatE(const int& EId, const TFlt& value, const TStr& attr);

  int AddIntVAttrDatE(const TEdgeI& EdgeI, const TIntV& value, const TStr& attr) { return AddIntVAttrDatE(EdgeI.GetId(), value, attr); }
  int AddIntVAttrDatE(const int& EId, const TIntV& value, const TStr& attr, TBool UseDense=true);

  int AppendIntVAttrDatE(const TEdgeI& EdgeI, const TInt& value, const TStr& attr) { return AppendIntVAttrDatE(EdgeI.GetId(), value, attr); }
  int AppendIntVAttrDatE(const int& EId, const TInt& value, const TStr& attr, TBool UseDense=true);

  TInt GetIntAttrDatN(const TNodeI& NodeI, const TStr& attr) { return GetIntAttrDatN(NodeI.GetId(), attr); }
  TInt GetIntAttrDatN(const int& NId, const TStr& attr);

  TStr GetStrAttrDatN(const TNodeI& NodeI, const TStr& attr) { return GetStrAttrDatN(NodeI.GetId(), attr); }
  TStr GetStrAttrDatN(const int& NId, const TStr& attr);

  TFlt GetFltAttrDatN(const TNodeI& NodeI, const TStr& attr) { return GetFltAttrDatN(NodeI.GetId(), attr); }
  TFlt GetFltAttrDatN(const int& NId, const TStr& attr);

  TIntV GetIntVAttrDatN(const TNodeI& NodeI, const TStr& attr) const { return GetIntVAttrDatN(NodeI.GetId(), attr); }
  TIntV GetIntVAttrDatN(const int& NId, const TStr& attr) const;

  int GetIntAttrIndN(const TStr& attr);

  int GetAttrIndN(const TStr& attr);

  TInt GetIntAttrIndDatN(const TNodeI& NodeI, const int& index) { return GetIntAttrIndDatN(NodeI.GetId(), index); }

  TInt GetIntAttrIndDatN(const int& NId, const int& index);

  TStr GetStrAttrIndDatN(const TNodeI& NodeI, const int& index) { return GetStrAttrIndDatN(NodeI.GetId(), index); }

  TStr GetStrAttrIndDatN(const int& NId, const int& index);

  TFlt GetFltAttrIndDatN(const TNodeI& NodeI, const int& index) { return GetFltAttrIndDatN(NodeI.GetId(), index); }

  TFlt GetFltAttrIndDatN(const int& NId, const int& index);

  TInt GetIntAttrDatE(const TEdgeI& EdgeI, const TStr& attr) { return GetIntAttrDatE(EdgeI.GetId(), attr); }
  TInt GetIntAttrDatE(const int& EId, const TStr& attr);

  TStr GetStrAttrDatE(const TEdgeI& EdgeI, const TStr& attr) { return GetStrAttrDatE(EdgeI.GetId(), attr); }
  TStr GetStrAttrDatE(const int& EId, const TStr& attr);

  TFlt GetFltAttrDatE(const TEdgeI& EdgeI, const TStr& attr) { return GetFltAttrDatE(EdgeI.GetId(), attr); }
  TFlt GetFltAttrDatE(const int& EId, const TStr& attr);

  TIntV GetIntVAttrDatE(const TEdgeI& EdgeI, const TStr& attr) { return GetIntVAttrDatE(EdgeI.GetId(), attr); }
  TIntV GetIntVAttrDatE(const int& EId, const TStr& attr);

  int GetIntAttrIndE(const TStr& attr);

  int GetAttrIndE(const TStr& attr);

  TInt GetIntAttrIndDatE(const TEdgeI& EdgeI, const int& index) { return GetIntAttrIndDatE(EdgeI.GetId(), index); }

  TInt GetIntAttrIndDatE(const int& EId, const int& index);
 

  TFlt GetFltAttrIndDatE(const TEdgeI& EdgeI, const int& index) { return GetFltAttrIndDatE(EdgeI.GetId(), index); }

  TFlt GetFltAttrIndDatE(const int& EId, const int& index);
 

  TStr GetStrAttrIndDatE(const TEdgeI& EdgeI, const int& index) { return GetStrAttrIndDatE(EdgeI.GetId(), index); }

  TStr GetStrAttrIndDatE(const int& EId, const int& index);
 

  int DelAttrDatN(const TNodeI& NodeI, const TStr& attr) { return DelAttrDatN(NodeI.GetId(), attr); } 
  int DelAttrDatN(const int& NId, const TStr& attr); 

  int DelAttrDatE(const TEdgeI& EdgeI, const TStr& attr) { return DelAttrDatE(EdgeI.GetId(), attr); } 
  int DelAttrDatE(const int& EId, const TStr& attr); 

  int AddIntAttrN(const TStr& attr, TInt defaultValue=TInt::Mn);

  int AddStrAttrN(const TStr& attr, TStr defaultValue=TStr::GetNullStr());

  int AddFltAttrN(const TStr& attr, TFlt defaultValue=TFlt::Mn);

  int AddIntVAttrN(const TStr& attr, TBool UseDense=true);

  int AddIntAttrE(const TStr& attr, TInt defaultValue=TInt::Mn);

  int AddStrAttrE(const TStr& attr, TStr defaultValue=TStr::GetNullStr());

  int AddFltAttrE(const TStr& attr, TFlt defaultValue=TFlt::Mn);

  int AddIntVAttrE(const TStr& attr, TBool UseDense=true);

  int DelAttrN(const TStr& attr);

  int DelAttrE(const TStr& attr);

  bool IsAttrDeletedN(const int& NId, const TStr& attr) const;

  bool IsIntAttrDeletedN(const int& NId, const TStr& attr) const;

  bool IsIntVAttrDeletedN(const int& NId, const TStr& attr) const;

  bool IsStrAttrDeletedN(const int& NId, const TStr& attr) const;

  bool IsFltAttrDeletedN(const int& NId, const TStr& attr) const;

  bool NodeAttrIsDeleted(const int& NId, const TStrIntPrH::TIter& NodeHI) const;

  bool NodeAttrIsIntDeleted(const int& NId, const TStrIntPrH::TIter& NodeHI) const;

  bool NodeAttrIsIntVDeleted(const int& NId, const TStrIntPrH::TIter& NodeHI) const;

  bool NodeAttrIsStrDeleted(const int& NId, const TStrIntPrH::TIter& NodeHI) const;

  bool NodeAttrIsFltDeleted(const int& NId, const TStrIntPrH::TIter& NodeHI) const;

  bool IsAttrDeletedE(const int& EId, const TStr& attr) const;

  bool IsIntAttrDeletedE(const int& EId, const TStr& attr) const;

  bool IsIntVAttrDeletedE(const int& EId, const TStr& attr) const;

  bool IsStrAttrDeletedE(const int& EId, const TStr& attr) const;

  bool IsFltAttrDeletedE(const int& EId, const TStr& attr) const;

  bool EdgeAttrIsDeleted(const int& EId, const TStrIntPrH::TIter& EdgeHI) const;

  bool EdgeAttrIsIntDeleted(const int& EId, const TStrIntPrH::TIter& EdgeHI) const;

  bool EdgeAttrIsIntVDeleted(const int& EId, const TStrIntPrH::TIter& EdgeHI) const;

  bool EdgeAttrIsStrDeleted(const int& EId, const TStrIntPrH::TIter& EdgeHI) const;

  bool EdgeAttrIsFltDeleted(const int& EId, const TStrIntPrH::TIter& EdgeHI) const;

  TStr GetNodeAttrValue(const int& NId, const TStrIntPrH::TIter& NodeHI) const;

  TStr GetEdgeAttrValue(const int& EId, const TStrIntPrH::TIter& EdgeHI) const;

  TFlt GetWeightOutEdges(const TNodeI& NI, const TStr& attr);

  bool IsFltAttrE(const TStr& attr);

  bool IsIntAttrE(const TStr& attr);

  bool IsStrAttrE(const TStr& attr);

  TVec<TFlt>& GetFltAttrVecE(const TStr& attr);

  int GetFltKeyIdE(const int& EId);

  void GetWeightOutEdgesV(TFltV& OutWeights, const TFltV& AttrVal) ;

  void GetAttrNNames(TStrV& IntAttrNames, TStrV& FltAttrNames, TStrV& StrAttrNames) const;

  void GetAttrENames(TStrV& IntAttrNames, TStrV& FltAttrNames, TStrV& StrAttrNames) const;

  int AddSAttrDatN(const TInt& NId, const TStr& AttrName, const TInt& Val);

  int AddSAttrDatN(const TInt& NId, const TInt& AttrId, const TInt& Val);

  int AddSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, const TInt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, const TInt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrId, Val);
  }

  int AddSAttrDatN(const TInt& NId, const TStr& AttrName, const TFlt& Val);

  int AddSAttrDatN(const TInt& NId, const TInt& AttrId, const TFlt& Val);

  int AddSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, const TFlt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, const TFlt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrId, Val);
  }

  int AddSAttrDatN(const TInt& NId, const TStr& AttrName, const TStr& Val);

  int AddSAttrDatN(const TInt& NId, const TInt& AttrId, const TStr& Val);

  int AddSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, const TStr& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, const TStr& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrId, Val);
  }

  int GetSAttrDatN(const TInt& NId, const TStr& AttrName, TInt& ValX) const;

  int GetSAttrDatN(const TInt& NId, const TInt& AttrId, TInt& ValX) const;

  int GetSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, TInt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrName, ValX);
  }

  int GetSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, TInt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrId, ValX);
  }

  int GetSAttrDatN(const TInt& NId, const TStr& AttrName, TFlt& ValX) const;

  int GetSAttrDatN(const TInt& NId, const TInt& AttrId, TFlt& ValX) const;

  int GetSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, TFlt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrName, ValX);
  } 

  int GetSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, TFlt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrId, ValX);
  }

  int GetSAttrDatN(const TInt& NId, const TStr& AttrName, TStr& ValX) const;

  int GetSAttrDatN(const TInt& NId, const TInt& AttrId, TStr& ValX) const;

  int GetSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, TStr& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrName, ValX);
  }

  int GetSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, TStr& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrId, ValX);
  }

  int DelSAttrDatN(const TInt& NId, const TStr& AttrName);

  int DelSAttrDatN(const TInt& NId, const TInt& AttrId);

  int DelSAttrDatN(const TNodeI& NodeI, const TStr& AttrName) {
    return DelSAttrDatN(NodeI.GetId(), AttrName);
  }

  int DelSAttrDatN(const TNodeI& NodeI, const TInt& AttrId) {
    return DelSAttrDatN(NodeI.GetId(), AttrId);
  }

  int GetSAttrVN(const TInt& NId, const TAttrType AttrType, TAttrPrV& AttrV) const;

  int GetSAttrVN(const TNodeI& NodeI, const TAttrType AttrType, TAttrPrV& AttrV) const {
    return GetSAttrVN(NodeI.GetId(), AttrType, AttrV);
  }

  int GetIdVSAttrN(const TStr& AttrName, TIntV& IdV) const;

  int GetIdVSAttrN(const TInt& AttrId, TIntV& IdV) const;

  int AddSAttrN(const TStr& Name, const TAttrType& AttrType, TInt& AttrId);

  int GetSAttrIdN(const TStr& Name, TInt& AttrIdX, TAttrType& AttrTypeX) const;

  int GetSAttrNameN(const TInt& AttrId, TStr& NameX, TAttrType& AttrTypeX) const;

  int AddSAttrDatE(const TInt& EId, const TStr& AttrName, const TInt& Val);

  int AddSAttrDatE(const TInt& EId, const TInt& AttrId, const TInt& Val);

  int AddSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, const TInt& Val) {
    return AddSAttrDatE(EdgeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, const TInt& Val) {
    return AddSAttrDatE(EdgeI.GetId(), AttrId, Val);
  }

  int AddSAttrDatE(const TInt& EId, const TStr& AttrName, const TFlt& Val);

  int AddSAttrDatE(const TInt& EId, const TInt& AttrId, const TFlt& Val);

  int AddSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, const TFlt& Val) {
    return AddSAttrDatE(EdgeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, const TFlt& Val){
    return AddSAttrDatE(EdgeI.GetId(), AttrId, Val);
  }

  int AddSAttrDatE(const TInt& EId, const TStr& AttrName, const TStr& Val);

  int AddSAttrDatE(const TInt& EId, const TInt& AttrId, const TStr& Val);

  int AddSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, const TStr& Val) {
    return AddSAttrDatE(EdgeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, const TStr& Val) {
    return AddSAttrDatE(EdgeI.GetId(), AttrId, Val);
  }

  int GetSAttrDatE(const TInt& EId, const TStr& AttrName, TInt& ValX) const;

  int GetSAttrDatE(const TInt& EId, const TInt& AttrId, TInt& ValX) const;

  int GetSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, TInt& ValX) const {
    return GetSAttrDatE(EdgeI.GetId(), AttrName, ValX);
  }

  int GetSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, TInt& ValX) const {
    return GetSAttrDatE(EdgeI.GetId(), AttrId, ValX);
  } 

  int GetSAttrDatE(const TInt& EId, const TStr& AttrName, TFlt& ValX) const; 

  int GetSAttrDatE(const TInt& EId, const TInt& AttrId, TFlt& ValX) const;

  int GetSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, TFlt& ValX) const {
    return GetSAttrDatE(EdgeI.GetId(), AttrName, ValX);
  }

  int GetSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, TFlt& ValX) const {
    return GetSAttrDatE(EdgeI.GetId(), AttrId, ValX);
  } 

  int GetSAttrDatE(const TInt& EId, const TStr& AttrName, TStr& ValX) const;

  int GetSAttrDatE(const TInt& EId, const TInt& AttrId, TStr& ValX) const;

  int GetSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, TStr& ValX) const {
    return GetSAttrDatE(EdgeI.GetId(), AttrName, ValX);
  }

  int GetSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, TStr& ValX) const {
    return GetSAttrDatE(EdgeI.GetId(), AttrId, ValX);
  }

  int DelSAttrDatE(const TInt& EId, const TStr& AttrName);

  int DelSAttrDatE(const TInt& EId, const TInt& AttrId);

  int DelSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName) {
    return DelSAttrDatE(EdgeI.GetId(), AttrName);
  }

  int DelSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId) {
    return DelSAttrDatE(EdgeI.GetId(), AttrId);
  } 

  int GetSAttrVE(const TInt& EId, const TAttrType AttrType, TAttrPrV& AttrV) const;

  int GetSAttrVE(const TEdgeI& EdgeI, const TAttrType AttrType, TAttrPrV& AttrV) const {
    return GetSAttrVE(EdgeI.GetId(), AttrType, AttrV);
  }

  int GetIdVSAttrE(const TStr& AttrName, TIntV& IdV) const;

  int GetIdVSAttrE(const TInt& AttrId, TIntV& IdV) const;

  int AddSAttrE(const TStr& Name, const TAttrType& AttrType, TInt& AttrId);

  int GetSAttrIdE(const TStr& Name, TInt& AttrIdX, TAttrType& AttrTypeX) const;

  int GetSAttrNameE(const TInt& AttrId, TStr& NameX, TAttrType& AttrTypeX) const;

  static PNEANet GetSmallGraph();
  friend class TPt<TNEANet>;
};
namespace TSnap {
template <> struct IsMultiGraph<TNEANet> { enum { Val = 1 }; };
template <> struct IsDirected<TNEANet> { enum { Val = 1 }; };
}

class TUndirNet;
typedef TPt<TUndirNet> PUndirNet;
class TDirNet;
typedef TPt<TDirNet> PDirNet;
class TUndirNet {
public:
  typedef TUndirNet TNet;
  typedef TPt<TUndirNet> PNet;
public:
  class TNode {
  private:
    TInt Id;
    TIntV NIdV;
  public:
    TNode() : Id(-1), NIdV() { }
    TNode(const int& NId) : Id(NId), NIdV() { }
    TNode(const TNode& Node) : Id(Node.Id), NIdV(Node.NIdV) { }
    TNode(TSIn& SIn) : Id(SIn), NIdV(SIn) { }
    void Save(TSOut& SOut) const { Id.Save(SOut); NIdV.Save(SOut); }
    int GetId() const { return Id; }
    int GetDeg() const { return NIdV.Len(); }
    int GetInDeg() const { return GetDeg(); }
    int GetOutDeg() const { return GetDeg(); }
    int GetInNId(const int& NodeN) const { return GetNbrNId(NodeN); }
    int GetOutNId(const int& NodeN) const { return GetNbrNId(NodeN); }
    int GetNbrNId(const int& NodeN) const { return NIdV[NodeN]; }
    bool IsNbrNId(const int& NId) const { return NIdV.SearchBin(NId)!=-1; }
    bool IsInNId(const int& NId) const { return IsNbrNId(NId); }
    bool IsOutNId(const int& NId) const { return IsNbrNId(NId); }
    void PackOutNIdV() { NIdV.Pack(); }
    void PackNIdV() { NIdV.Pack(); }
    void SortNIdV() { NIdV.Sort();}
    void LoadShM(TShMIn& MStream) {
      Id = TInt(MStream);
      NIdV.LoadShM(MStream);
    }
    friend class TUndirNet;
    friend class TUndirNetMtx;
  };

  class TNodeI {
  private:
    typedef THash<TInt, TNode>::TIter THashIter;
    THashIter NodeHI;
  public:
    TNodeI() : NodeHI() { }
    TNodeI(const THashIter& NodeHIter) : NodeHI(NodeHIter) { }
    TNodeI(const TNodeI& NodeI) : NodeHI(NodeI.NodeHI) { }
    TNodeI& operator = (const TNodeI& NodeI) { NodeHI = NodeI.NodeHI; return *this; }

    TNodeI& operator++ (int) { NodeHI++; return *this; }

    TNodeI& operator-- (int) { NodeHI--; return *this; }
    bool operator < (const TNodeI& NodeI) const { return NodeHI < NodeI.NodeHI; }
    bool operator == (const TNodeI& NodeI) const { return NodeHI == NodeI.NodeHI; }

    int GetId() const { return NodeHI.GetDat().GetId(); }

    int GetDeg() const { return NodeHI.GetDat().GetDeg(); }

    int GetInDeg() const { return NodeHI.GetDat().GetInDeg(); }

    int GetOutDeg() const { return NodeHI.GetDat().GetOutDeg(); }

    void SortNIdV() { NodeHI.GetDat().SortNIdV(); }

    int GetInNId(const int& NodeN) const { return NodeHI.GetDat().GetInNId(NodeN); }

    int GetOutNId(const int& NodeN) const { return NodeHI.GetDat().GetOutNId(NodeN); }

    int GetNbrNId(const int& NodeN) const { return NodeHI.GetDat().GetNbrNId(NodeN); }

    bool IsInNId(const int& NId) const { return NodeHI.GetDat().IsInNId(NId); }

    bool IsOutNId(const int& NId) const { return NodeHI.GetDat().IsOutNId(NId); }

    bool IsNbrNId(const int& NId) const { return NodeHI.GetDat().IsNbrNId(NId); }
    friend class TUndirNet;
  };

  class TEdgeI {
  private:
    TNodeI CurNode, EndNode;
    int CurEdge;
  public:
    TEdgeI() : CurNode(), EndNode(), CurEdge(0) { }
    TEdgeI(const TNodeI& NodeI, const TNodeI& EndNodeI, const int& EdgeN=0) : CurNode(NodeI), EndNode(EndNodeI), CurEdge(EdgeN) { }
    TEdgeI(const TEdgeI& EdgeI) : CurNode(EdgeI.CurNode), EndNode(EdgeI.EndNode), CurEdge(EdgeI.CurEdge) { }
    TEdgeI& operator = (const TEdgeI& EdgeI) { if (this!=&EdgeI) { CurNode=EdgeI.CurNode; EndNode=EdgeI.EndNode; CurEdge=EdgeI.CurEdge; } return *this; }

    TEdgeI& operator++ (int) { do { CurEdge++; if (CurEdge >= CurNode.GetOutDeg()) { CurEdge=0; CurNode++; while (CurNode < EndNode && CurNode.GetOutDeg()==0) { CurNode++; } } } while (CurNode < EndNode && GetSrcNId()>GetDstNId()); return *this; }
    bool operator < (const TEdgeI& EdgeI) const { return CurNode<EdgeI.CurNode || (CurNode==EdgeI.CurNode && CurEdge<EdgeI.CurEdge); }
    bool operator == (const TEdgeI& EdgeI) const { return CurNode == EdgeI.CurNode && CurEdge == EdgeI.CurEdge; }

    int GetId() const { return -1; }

    int GetSrcNId() const { return CurNode.GetId(); }

    int GetDstNId() const { return CurNode.GetOutNId(CurEdge); }
    friend class TUndirNet;
  };
private:
  TCRef CRef;
  TInt MxNId, NEdges;
  THash<TInt, TNode> NodeH;
  TAttr SAttrN;
  TAttrPair SAttrE;
private:
  TNode& GetNode(const int& NId) { return NodeH.GetDat(NId); }
  const TNode& GetNode(const int& NId) const { return NodeH.GetDat(NId); }
  TIntPr OrderEdgeNodes(const int& SrcNId, const int& DstNId) const;
private:
  class LoadTNodeFunctor {
  public:
    LoadTNodeFunctor() {}
    void operator() (TNode* n, TShMIn& ShMIn) {n->LoadShM(ShMIn);}
  };
private:
  void LoadNetworkShM(TShMIn& ShMIn) {
    MxNId = TInt(ShMIn);
    NEdges = TInt(ShMIn);
    LoadTNodeFunctor NodeFn;
    NodeH.LoadShM(ShMIn, NodeFn);
    SAttrN.Load(ShMIn);
    SAttrE = TAttrPair(ShMIn);
  }
public:
  TUndirNet() : CRef(), MxNId(0), NEdges(0), NodeH(), SAttrN(), SAttrE() { }

  explicit TUndirNet(const int& Nodes, const int& Edges) : MxNId(0), NEdges(0), SAttrN(), SAttrE() { Reserve(Nodes, Edges); }
  TUndirNet(const TUndirNet& Graph) : MxNId(Graph.MxNId), NEdges(Graph.NEdges), NodeH(Graph.NodeH),
    SAttrN(), SAttrE() { }

  TUndirNet(TSIn& SIn) : MxNId(SIn), NEdges(SIn), NodeH(SIn), SAttrN(SIn), SAttrE(SIn) { }

  void Save(TSOut& SOut) const { MxNId.Save(SOut); NEdges.Save(SOut); NodeH.Save(SOut);
    SAttrN.Save(SOut); SAttrE.Save(SOut); }

  void Save_V1(TSOut& SOut) const { MxNId.Save(SOut); NEdges.Save(SOut); NodeH.Save(SOut); }

  static PUndirNet New() { return new TUndirNet(); }

  static PUndirNet New(const int& Nodes, const int& Edges) { return new TUndirNet(Nodes, Edges); }

  static PUndirNet Load(TSIn& SIn) { return PUndirNet(new TUndirNet(SIn)); }

  static PUndirNet Load_V1(TSIn& SIn) { PUndirNet Graph = PUndirNet(new TUndirNet());
    Graph->MxNId.Load(SIn); Graph->NEdges.Load(SIn); Graph->NodeH.Load(SIn); return Graph;
  }

  static PUndirNet LoadShM(TShMIn& ShMIn) {
    TUndirNet* Network = new TUndirNet();
    Network->LoadNetworkShM(ShMIn);
    return PUndirNet(Network);
  }

  bool HasFlag(const TGraphFlag& Flag) const;
  TUndirNet& operator = (const TUndirNet& Graph) {
    if (this!=&Graph) { MxNId=Graph.MxNId; NEdges=Graph.NEdges; NodeH=Graph.NodeH; } return *this; }
  

  int GetNodes() const { return NodeH.Len(); }

  int AddNode(int NId = -1);

  int AddNodeUnchecked(int NId = -1);

  int AddNode(const TNodeI& NodeI) { return AddNode(NodeI.GetId()); }

  int AddNode(const int& NId, const TIntV& NbrNIdV);

  int AddNode(const int& NId, const TVecPool<TInt>& Pool, const int& NIdVId);

  void DelNode(const int& NId);

  void DelNode(const TNode& NodeI) { DelNode(NodeI.GetId()); }

  bool IsNode(const int& NId) const { return NodeH.IsKey(NId); }

  TNodeI BegNI() const { return TNodeI(NodeH.BegI()); }

  TNodeI EndNI() const { return TNodeI(NodeH.EndI()); }

  TNodeI GetNI(const int& NId) const { return TNodeI(NodeH.GetI(NId)); }

  int GetMxNId() const { return MxNId; }

  int GetEdges() const;

  int AddEdge(const int& SrcNId, const int& DstNId);

  int AddEdgeUnchecked(const int& SrcNId, const int& DstNId);

  int AddEdge(const TEdgeI& EdgeI) { return AddEdge(EdgeI.GetSrcNId(), EdgeI.GetDstNId()); }

  void DelEdge(const int& SrcNId, const int& DstNId);

  bool IsEdge(const int& SrcNId, const int& DstNId) const;

  TEdgeI BegEI() const { TNodeI NI = BegNI(); TEdgeI EI(NI, EndNI(), 0); if (GetNodes() != 0 && (NI.GetOutDeg()==0 || NI.GetId()>NI.GetOutNId(0))) { EI++; } return EI; }

  TEdgeI EndEI() const { return TEdgeI(EndNI(), EndNI()); }

  TEdgeI GetEI(const int& EId) const;

  TEdgeI GetEI(const int& SrcNId, const int& DstNId) const;

  int GetRndNId(TRnd& Rnd=TInt::Rnd) { return NodeH.GetKey(NodeH.GetRndKeyId(Rnd, 0.8)); }

  TNodeI GetRndNI(TRnd& Rnd=TInt::Rnd) { return GetNI(GetRndNId(Rnd)); }

  void GetNIdV(TIntV& NIdV) const;

  bool Empty() const { return GetNodes()==0; }

  void Clr() { MxNId=0; NEdges=0; NodeH.Clr(); SAttrN.Clr(); SAttrE.Clr(); }

  void Reserve(const int& Nodes, const int& Edges) { if (Nodes>0) NodeH.Gen(Nodes/2); }

  void ReserveNIdDeg(const int& NId, const int& Deg) { GetNode(NId).NIdV.Reserve(Deg); }

  void SortNodeAdjV() { for (TNodeI NI = BegNI(); NI < EndNI(); NI++) { NI.SortNIdV();} }

  void Defrag(const bool& OnlyNodeLinks=false);

  bool IsOk(const bool& ThrowExcept=true) const;

  void Dump(FILE *OutF=stdout) const;

  static PUndirNet GetSmallGraph();

  int AddSAttrDatN(const TInt& NId, const TStr& AttrName, const TInt& Val);

  int AddSAttrDatN(const TInt& NId, const TInt& AttrId, const TInt& Val);

  int AddSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, const TInt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, const TInt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrId, Val);
  }

  int AddSAttrDatN(const TInt& NId, const TStr& AttrName, const TFlt& Val);

  int AddSAttrDatN(const TInt& NId, const TInt& AttrId, const TFlt& Val);

  int AddSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, const TFlt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, const TFlt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrId, Val);
  }

  int AddSAttrDatN(const TInt& NId, const TStr& AttrName, const TStr& Val);

  int AddSAttrDatN(const TInt& NId, const TInt& AttrId, const TStr& Val);

  int AddSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, const TStr& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, const TStr& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrId, Val);
  }

  int GetSAttrDatN(const TInt& NId, const TStr& AttrName, TInt& ValX) const;

  int GetSAttrDatN(const TInt& NId, const TInt& AttrId, TInt& ValX) const;

  int GetSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, TInt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrName, ValX);
  }

  int GetSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, TInt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrId, ValX);
  }

  int GetSAttrDatN(const TInt& NId, const TStr& AttrName, TFlt& ValX) const;

  int GetSAttrDatN(const TInt& NId, const TInt& AttrId, TFlt& ValX) const;

  int GetSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, TFlt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrName, ValX);
  } 

  int GetSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, TFlt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrId, ValX);
  }

  int GetSAttrDatN(const TInt& NId, const TStr& AttrName, TStr& ValX) const;

  int GetSAttrDatN(const TInt& NId, const TInt& AttrId, TStr& ValX) const;

  int GetSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, TStr& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrName, ValX);
  }

  int GetSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, TStr& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrId, ValX);
  }

  int DelSAttrDatN(const TInt& NId, const TStr& AttrName);

  int DelSAttrDatN(const TInt& NId, const TInt& AttrId);

  int DelSAttrDatN(const TNodeI& NodeI, const TStr& AttrName) {
    return DelSAttrDatN(NodeI.GetId(), AttrName);
  }

  int DelSAttrDatN(const TNodeI& NodeI, const TInt& AttrId) {
    return DelSAttrDatN(NodeI.GetId(), AttrId);
  }

  int GetSAttrVN(const TInt& NId, const TAttrType AttrType, TAttrPrV& AttrV) const;

  int GetSAttrVN(const TNodeI& NodeI, const TAttrType AttrType, TAttrPrV& AttrV) const {
    return GetSAttrVN(NodeI.GetId(), AttrType, AttrV);
  }

  int GetIdVSAttrN(const TStr& AttrName, TIntV& IdV) const;

  int GetIdVSAttrN(const TInt& AttrId, TIntV& IdV) const;

  int AddSAttrN(const TStr& Name, const TAttrType& AttrType, TInt& AttrId);

  int GetSAttrIdN(const TStr& Name, TInt& AttrIdX, TAttrType& AttrTypeX) const;

  int GetSAttrNameN(const TInt& AttrId, TStr& NameX, TAttrType& AttrTypeX) const;

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, const TInt& Val);

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, const TInt& Val);

  int AddSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, const TInt& Val) {
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, Val);
  }

  int AddSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, const TInt& Val) {
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, Val);
  }

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, const TFlt& Val);

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, const TFlt& Val);

  int AddSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, const TFlt& Val) {
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, Val);
  }

  int AddSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, const TFlt& Val){
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, Val);
  }

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, const TStr& Val);

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, const TStr& Val);

  int AddSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, const TStr& Val) {
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, Val);
  }

  int AddSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, const TStr& Val) {
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, Val);
  }

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, TInt& ValX) const;

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, TInt& ValX) const;

  int GetSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, TInt& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, ValX);
  }

  int GetSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, TInt& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, ValX);
  } 

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, TFlt& ValX) const; 

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, TFlt& ValX) const;

  int GetSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, TFlt& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, ValX);
  }

  int GetSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, TFlt& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, ValX);
  } 

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, TStr& ValX) const;

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, TStr& ValX) const;

  int GetSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, TStr& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, ValX);
  }

  int GetSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, TStr& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, ValX);
  }

  int DelSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName);

  int DelSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId);

  int DelSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName) {
    return DelSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName);
  }

  int DelSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId) {
    return DelSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId);
  } 

  int GetSAttrVE(const int& SrcNId, const int& DstNId, const TAttrType AttrType, TAttrPrV& AttrV) const;

  int GetSAttrVE(const TEdgeI& EdgeI, const TAttrType AttrType, TAttrPrV& AttrV) const {
    return GetSAttrVE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrType, AttrV);
  }

  int GetIdVSAttrE(const TStr& AttrName, TIntPrV& IdV) const;

  int GetIdVSAttrE(const TInt& AttrId, TIntPrV& IdV) const;

  int AddSAttrE(const TStr& Name, const TAttrType& AttrType, TInt& AttrId);

  int GetSAttrIdE(const TStr& Name, TInt& AttrIdX, TAttrType& AttrTypeX) const;

  int GetSAttrNameE(const TInt& AttrId, TStr& NameX, TAttrType& AttrTypeX) const;
  friend class TUndirNetMtx;
  friend class TPt<TUndirNet>;
};
class TDirNet {
public:
  typedef TDirNet TNet;
  typedef TPt<TDirNet> PNet;
public:
  class TNode {
  private:
    TInt Id;
    TIntV InNIdV, OutNIdV;
  public:
    TNode() : Id(-1), InNIdV(), OutNIdV() { }
    TNode(const int& NId) : Id(NId), InNIdV(), OutNIdV() { }
    TNode(const TNode& Node) : Id(Node.Id), InNIdV(Node.InNIdV), OutNIdV(Node.OutNIdV) { }
    TNode(TSIn& SIn) : Id(SIn), InNIdV(SIn), OutNIdV(SIn) { }
    void Save(TSOut& SOut) const { Id.Save(SOut); InNIdV.Save(SOut); OutNIdV.Save(SOut); }
    int GetId() const { return Id; }
    int GetDeg() const { return GetInDeg() + GetOutDeg(); }
    int GetInDeg() const { return InNIdV.Len(); }
    int GetOutDeg() const { return OutNIdV.Len(); }
    int GetInNId(const int& NodeN) const { return InNIdV[NodeN]; }
    int GetOutNId(const int& NodeN) const { return OutNIdV[NodeN]; }
    int GetNbrNId(const int& NodeN) const { return NodeN<GetOutDeg()?GetOutNId(NodeN):GetInNId(NodeN-GetOutDeg()); }
    bool IsInNId(const int& NId) const { return InNIdV.SearchBin(NId) != -1; }
    bool IsOutNId(const int& NId) const { return OutNIdV.SearchBin(NId) != -1; }
    bool IsNbrNId(const int& NId) const { return IsOutNId(NId) || IsInNId(NId); }
    void PackOutNIdV() { OutNIdV.Pack(); }
    void PackNIdV() { InNIdV.Pack(); }
    void SortNIdV() { InNIdV.Sort(); OutNIdV.Sort();}
    void LoadShM(TShMIn& MStream) {
      Id = TInt(MStream);
      InNIdV.LoadShM(MStream);
      OutNIdV.LoadShM(MStream);
    }
    friend class TDirNet;
    friend class TDirNetMtx;
  };

  class TNodeI {
  private:
    typedef THash<TInt, TNode>::TIter THashIter;
    THashIter NodeHI;
  public:
    TNodeI() : NodeHI() { }
    TNodeI(const THashIter& NodeHIter) : NodeHI(NodeHIter) { }
    TNodeI(const TNodeI& NodeI) : NodeHI(NodeI.NodeHI) { }
    TNodeI& operator = (const TNodeI& NodeI) { NodeHI = NodeI.NodeHI; return *this; }

    TNodeI& operator++ (int) { NodeHI++; return *this; }

    TNodeI& operator-- (int) { NodeHI--; return *this; }
    bool operator < (const TNodeI& NodeI) const { return NodeHI < NodeI.NodeHI; }
    bool operator == (const TNodeI& NodeI) const { return NodeHI == NodeI.NodeHI; }

    int GetId() const { return NodeHI.GetDat().GetId(); }

    int GetDeg() const { return NodeHI.GetDat().GetDeg(); }

    int GetInDeg() const { return NodeHI.GetDat().GetInDeg(); }

    int GetOutDeg() const { return NodeHI.GetDat().GetOutDeg(); }

    void SortNIdV() { NodeHI.GetDat().SortNIdV(); }

    int GetInNId(const int& NodeN) const { return NodeHI.GetDat().GetInNId(NodeN); }

    int GetOutNId(const int& NodeN) const { return NodeHI.GetDat().GetOutNId(NodeN); }

    int GetNbrNId(const int& NodeN) const { return NodeHI.GetDat().GetNbrNId(NodeN); }

    bool IsInNId(const int& NId) const { return NodeHI.GetDat().IsInNId(NId); }

    bool IsOutNId(const int& NId) const { return NodeHI.GetDat().IsOutNId(NId); }

    bool IsNbrNId(const int& NId) const { return IsOutNId(NId) || IsInNId(NId); }
    friend class TDirNet;
  };

  class TEdgeI {
  private:
    TNodeI CurNode, EndNode;
    int CurEdge;
  public:
    TEdgeI() : CurNode(), EndNode(), CurEdge(0) { }
    TEdgeI(const TNodeI& NodeI, const TNodeI& EndNodeI, const int& EdgeN=0) : CurNode(NodeI), EndNode(EndNodeI), CurEdge(EdgeN) { }
    TEdgeI(const TEdgeI& EdgeI) : CurNode(EdgeI.CurNode), EndNode(EdgeI.EndNode), CurEdge(EdgeI.CurEdge) { }
    TEdgeI& operator = (const TEdgeI& EdgeI) { if (this!=&EdgeI) { CurNode=EdgeI.CurNode; EndNode=EdgeI.EndNode; CurEdge=EdgeI.CurEdge; }  return *this; }

    TEdgeI& operator++ (int) { CurEdge++; if (CurEdge >= CurNode.GetOutDeg()) { CurEdge=0; CurNode++;
      while (CurNode < EndNode && CurNode.GetOutDeg()==0) { CurNode++; } }  return *this; }
    bool operator < (const TEdgeI& EdgeI) const { return CurNode<EdgeI.CurNode || (CurNode==EdgeI.CurNode && CurEdge<EdgeI.CurEdge); }
    bool operator == (const TEdgeI& EdgeI) const { return CurNode == EdgeI.CurNode && CurEdge == EdgeI.CurEdge; }

    int GetId() const { return -1; }

    int GetSrcNId() const { return CurNode.GetId(); }

    int GetDstNId() const { return CurNode.GetOutNId(CurEdge); }
    friend class TDirNet;
  };
private:
  TCRef CRef;
  TInt MxNId;
  THash<TInt, TNode> NodeH;
  TAttr SAttrN;
  TAttrPair SAttrE;
private:
  TNode& GetNode(const int& NId) { return NodeH.GetDat(NId); }
  const TNode& GetNode(const int& NId) const { return NodeH.GetDat(NId); }
private:
  class TNodeFunctor {
  public:
    TNodeFunctor() {}
    void operator() (TNode* n, TShMIn& ShMIn) { n->LoadShM(ShMIn);}
  };
private:
  void LoadNetworkShM(TShMIn& ShMIn) {
    MxNId = TInt(ShMIn);
    TNodeFunctor f;
    NodeH.LoadShM(ShMIn, f);
    SAttrN.Load(ShMIn);
    SAttrE = TAttrPair(ShMIn);
  }
public:
  TDirNet() : CRef(), MxNId(0), NodeH(), SAttrN(), SAttrE() { }

  explicit TDirNet(const int& Nodes, const int& Edges) : MxNId(0), SAttrN(), SAttrE() { Reserve(Nodes, Edges); }
  TDirNet(const TDirNet& Graph) : MxNId(Graph.MxNId), NodeH(Graph.NodeH), SAttrN(), SAttrE() { }

  TDirNet(TSIn& SIn) : MxNId(SIn), NodeH(SIn), SAttrN(SIn), SAttrE(SIn) { }

  void Save(TSOut& SOut) const { MxNId.Save(SOut); NodeH.Save(SOut); SAttrN.Save(SOut); SAttrE.Save(SOut); }

  void Save_V1(TSOut& SOut) const { MxNId.Save(SOut); NodeH.Save(SOut); }

  static PDirNet New() { return new TDirNet(); }

  static PDirNet New(const int& Nodes, const int& Edges) { return new TDirNet(Nodes, Edges); }

  static PDirNet Load(TSIn& SIn) { return PDirNet(new TDirNet(SIn)); }

  static PDirNet Load_V1(TSIn& SIn) { PDirNet Graph = PDirNet(new TDirNet());
    Graph->MxNId.Load(SIn); Graph->NodeH.Load(SIn); return Graph;
  }

  static PDirNet LoadShM(TShMIn& ShMIn) {
    TDirNet* Network = new TDirNet();
    Network->LoadNetworkShM(ShMIn);
    return PDirNet(Network);
  }

  bool HasFlag(const TGraphFlag& Flag) const;
  TDirNet& operator = (const TDirNet& Graph) {
    if (this!=&Graph) { MxNId=Graph.MxNId; NodeH=Graph.NodeH; }  return *this; }
  

  int GetNodes() const { return NodeH.Len(); }

  int AddNode(int NId = -1);

  int AddNodeUnchecked(int NId = -1);

  int AddNode(const TNodeI& NodeId) { return AddNode(NodeId.GetId()); }

  int AddNode(const int& NId, const TIntV& InNIdV, const TIntV& OutNIdV);

  int AddNode(const int& NId, const TVecPool<TInt>& Pool, const int& SrcVId, const int& DstVId);

  void DelNode(const int& NId);

  void DelNode(const TNode& NodeI) { DelNode(NodeI.GetId()); }

  bool IsNode(const int& NId) const { return NodeH.IsKey(NId); }

  TNodeI BegNI() const { return TNodeI(NodeH.BegI()); }

  TNodeI EndNI() const { return TNodeI(NodeH.EndI()); }

  TNodeI GetNI(const int& NId) const { return TNodeI(NodeH.GetI(NId)); }



  int GetMxNId() const { return MxNId; }

  int GetEdges() const;

  int AddEdge(const int& SrcNId, const int& DstNId);

  int AddEdgeUnchecked(const int& SrcNId, const int& DstNId);

  int AddEdge(const TEdgeI& EdgeI) { return AddEdge(EdgeI.GetSrcNId(), EdgeI.GetDstNId()); }

  void DelEdge(const int& SrcNId, const int& DstNId, const bool& IsDir = true);

  bool IsEdge(const int& SrcNId, const int& DstNId, const bool& IsDir = true) const;

  TEdgeI BegEI() const { TNodeI NI=BegNI(); while(NI<EndNI() && NI.GetOutDeg()==0){NI++;} return TEdgeI(NI, EndNI()); }

  TEdgeI EndEI() const { return TEdgeI(EndNI(), EndNI()); }

  TEdgeI GetEI(const int& EId) const;

  TEdgeI GetEI(const int& SrcNId, const int& DstNId) const;

  int GetRndNId(TRnd& Rnd=TInt::Rnd) { return NodeH.GetKey(NodeH.GetRndKeyId(Rnd, 0.8)); }

  TNodeI GetRndNI(TRnd& Rnd=TInt::Rnd) { return GetNI(GetRndNId(Rnd)); }

  void GetNIdV(TIntV& NIdV) const;

  bool Empty() const { return GetNodes()==0; }

  void Clr() { MxNId=0; NodeH.Clr(); SAttrN.Clr(); SAttrE.Clr(); }

  void Reserve(const int& Nodes, const int& Edges) { if (Nodes>0) { NodeH.Gen(Nodes/2); } }

  void ReserveNIdInDeg(const int& NId, const int& InDeg) { GetNode(NId).InNIdV.Reserve(InDeg); }

  void ReserveNIdOutDeg(const int& NId, const int& OutDeg) { GetNode(NId).OutNIdV.Reserve(OutDeg); }

  void SortNodeAdjV() { for (TNodeI NI = BegNI(); NI < EndNI(); NI++) { NI.SortNIdV();} }

  void Defrag(const bool& OnlyNodeLinks=false);

  bool IsOk(const bool& ThrowExcept=true) const;

  void Dump(FILE *OutF=stdout) const;

  static PDirNet GetSmallGraph();

  int AddSAttrDatN(const TInt& NId, const TStr& AttrName, const TInt& Val);

  int AddSAttrDatN(const TInt& NId, const TInt& AttrId, const TInt& Val);

  int AddSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, const TInt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, const TInt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrId, Val);
  }

  int AddSAttrDatN(const TInt& NId, const TStr& AttrName, const TFlt& Val);

  int AddSAttrDatN(const TInt& NId, const TInt& AttrId, const TFlt& Val);

  int AddSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, const TFlt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, const TFlt& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrId, Val);
  }

  int AddSAttrDatN(const TInt& NId, const TStr& AttrName, const TStr& Val);

  int AddSAttrDatN(const TInt& NId, const TInt& AttrId, const TStr& Val);

  int AddSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, const TStr& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrName, Val);
  }

  int AddSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, const TStr& Val) {
    return AddSAttrDatN(NodeI.GetId(), AttrId, Val);
  }

  int GetSAttrDatN(const TInt& NId, const TStr& AttrName, TInt& ValX) const;

  int GetSAttrDatN(const TInt& NId, const TInt& AttrId, TInt& ValX) const;

  int GetSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, TInt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrName, ValX);
  }

  int GetSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, TInt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrId, ValX);
  }

  int GetSAttrDatN(const TInt& NId, const TStr& AttrName, TFlt& ValX) const;

  int GetSAttrDatN(const TInt& NId, const TInt& AttrId, TFlt& ValX) const;

  int GetSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, TFlt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrName, ValX);
  } 

  int GetSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, TFlt& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrId, ValX);
  }

  int GetSAttrDatN(const TInt& NId, const TStr& AttrName, TStr& ValX) const;

  int GetSAttrDatN(const TInt& NId, const TInt& AttrId, TStr& ValX) const;

  int GetSAttrDatN(const TNodeI& NodeI, const TStr& AttrName, TStr& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrName, ValX);
  }

  int GetSAttrDatN(const TNodeI& NodeI, const TInt& AttrId, TStr& ValX) const {
    return GetSAttrDatN(NodeI.GetId(), AttrId, ValX);
  }

  int DelSAttrDatN(const TInt& NId, const TStr& AttrName);

  int DelSAttrDatN(const TInt& NId, const TInt& AttrId);

  int DelSAttrDatN(const TNodeI& NodeI, const TStr& AttrName) {
    return DelSAttrDatN(NodeI.GetId(), AttrName);
  }

  int DelSAttrDatN(const TNodeI& NodeI, const TInt& AttrId) {
    return DelSAttrDatN(NodeI.GetId(), AttrId);
  }

  int GetSAttrVN(const TInt& NId, const TAttrType AttrType, TAttrPrV& AttrV) const;

  int GetSAttrVN(const TNodeI& NodeI, const TAttrType AttrType, TAttrPrV& AttrV) const {
    return GetSAttrVN(NodeI.GetId(), AttrType, AttrV);
  }

  int GetIdVSAttrN(const TStr& AttrName, TIntV& IdV) const;

  int GetIdVSAttrN(const TInt& AttrId, TIntV& IdV) const;

  int AddSAttrN(const TStr& Name, const TAttrType& AttrType, TInt& AttrId);

  int GetSAttrIdN(const TStr& Name, TInt& AttrIdX, TAttrType& AttrTypeX) const;

  int GetSAttrNameN(const TInt& AttrId, TStr& NameX, TAttrType& AttrTypeX) const;

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, const TInt& Val);

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, const TInt& Val);

  int AddSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, const TInt& Val) {
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, Val);
  }

  int AddSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, const TInt& Val) {
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, Val);
  }

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, const TFlt& Val);

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, const TFlt& Val);

  int AddSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, const TFlt& Val) {
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, Val);
  }

  int AddSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, const TFlt& Val){
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, Val);
  }

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, const TStr& Val);

  int AddSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, const TStr& Val);

  int AddSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, const TStr& Val) {
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, Val);
  }

  int AddSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, const TStr& Val) {
    return AddSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, Val);
  }

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, TInt& ValX) const;

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, TInt& ValX) const;

  int GetSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, TInt& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, ValX);
  }

  int GetSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, TInt& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, ValX);
  } 

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, TFlt& ValX) const; 

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, TFlt& ValX) const;

  int GetSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, TFlt& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, ValX);
  }

  int GetSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, TFlt& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, ValX);
  } 

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName, TStr& ValX) const;

  int GetSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId, TStr& ValX) const;

  int GetSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName, TStr& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName, ValX);
  }

  int GetSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId, TStr& ValX) const {
    return GetSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId, ValX);
  }

  int DelSAttrDatE(const int& SrcNId, const int& DstNId, const TStr& AttrName);

  int DelSAttrDatE(const int& SrcNId, const int& DstNId, const TInt& AttrId);

  int DelSAttrDatE(const TEdgeI& EdgeI, const TStr& AttrName) {
    return DelSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrName);
  }

  int DelSAttrDatE(const TEdgeI& EdgeI, const TInt& AttrId) {
    return DelSAttrDatE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrId);
  } 

  int GetSAttrVE(const int& SrcNId, const int& DstNId, const TAttrType AttrType, TAttrPrV& AttrV) const;

  int GetSAttrVE(const TEdgeI& EdgeI, const TAttrType AttrType, TAttrPrV& AttrV) const {
    return GetSAttrVE(EdgeI.GetSrcNId(), EdgeI.GetDstNId(), AttrType, AttrV);
  }

  int GetIdVSAttrE(const TStr& AttrName, TIntPrV& IdV) const;

  int GetIdVSAttrE(const TInt& AttrId, TIntPrV& IdV) const;

  int AddSAttrE(const TStr& Name, const TAttrType& AttrType, TInt& AttrId);

  int GetSAttrIdE(const TStr& Name, TInt& AttrIdX, TAttrType& AttrTypeX) const;

  int GetSAttrNameE(const TInt& AttrId, TStr& NameX, TAttrType& AttrTypeX) const;
  friend class TPt<TDirNet>;
  friend class TDirNetMtx;
};
namespace TSnap {
template <> struct IsDirected<TDirNet> { enum { Val = 1 }; };
}
#endif
