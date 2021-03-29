

namespace TSnap {


const TStr EDGES_START = ("#EDGES");
const TStr NODES_START = ("#NODES");
const TStr END_SENTINEL = ("#END");
const TStr SRC_ID_NAME = ("SrcNId");
const TStr DST_ID_NAME = ("DstNId");
const TStr NID_NAME = ("NId");
const TStr INT_TYPE_PREFIX = ("Int");
const TStr FLT_TYPE_PREFIX = ("Flt");
const TStr STR_TYPE_PREFIX = ("Str");
const TStr NULL_VAL = ("__null__");


template <class PGraph> PGraph LoadEdgeList(const TStr& InFNm, const int& SrcColId=0, const int& DstColId=1);

template <class PGraph> PGraph LoadEdgeList(const TStr& InFNm, const int& SrcColId, const int& DstColId, const char& Separator);

PNEANet LoadEdgeListNet(const TStr& InFNm, const char& Separator);

template <class PGraph> PGraph LoadEdgeListStr(const TStr& InFNm, const int& SrcColId=0, const int& DstColId=1);

template <class PGraph> PGraph LoadEdgeListStr(const TStr& InFNm, const int& SrcColId, const int& DstColId, TStrHash<TInt>& StrToNIdH);

template <class PGraph> PGraph LoadConnList(const TStr& InFNm);

template <class PGraph> PGraph LoadConnListStr(const TStr& InFNm, TStrHash<TInt>& StrToNIdH);


template <class PGraph> PGraph LoadPajek(const TStr& InFNm);

PNGraph LoadDyNet(const TStr& FNm);

TVec<PNGraph> LoadDyNetGraphV(const TStr& FNm);








template <class PGraph> void SaveEdgeList(const PGraph& Graph, const TStr& OutFNm, const TStr& Desc=TStr());

void SaveEdgeListNet(const PNEANet& Graph, const TStr& OutFNm, const TStr& Desc);

template <class PGraph> void SavePajek(const PGraph& Graph, const TStr& OutFNm);

template <class PGraph> void SavePajek(const PGraph& Graph, const TStr& OutFNm, const TIntStrH& NIdColorH);

template <class PGraph> void SavePajek(const PGraph& Graph, const TStr& OutFNm, const TIntStrH& NIdColorH, const TIntStrH& NIdLabelH);

template <class PGraph> void SavePajek(const PGraph& Graph, const TStr& OutFNm, const TIntStrH& NIdColorH, const TIntStrH& NIdLabelH, const TIntStrH& EIdColorH);

template <class PGraph> void SaveMatlabSparseMtx(const PGraph& Graph, const TStr& OutFNm);

template<class PGraph> void SaveGViz(const PGraph& Graph, const TStr& OutFNm, const TStr& Desc=TStr(), const bool& NodeLabels=false, const TIntStrH& NIdColorH=TIntStrH());

template<class PGraph> void SaveGViz(const PGraph& Graph, const TStr& OutFNm, const TStr& Desc, const TIntStrH& NIdLabelH);








template <class PGraph>
PGraph LoadEdgeList(const TStr& InFNm, const int& SrcColId, const int& DstColId) {
  TSsParser Ss(InFNm, ssfWhiteSep, true, true, true);
  PGraph Graph = PGraph::TObj::New();
  int SrcNId, DstNId;

  while (Ss.Next()) {
    if (! Ss.GetInt(SrcColId, SrcNId) || ! Ss.GetInt(DstColId, DstNId)) { continue; }
    if (! Graph->IsNode(SrcNId)) { Graph->AddNode(SrcNId); }
    if (! Graph->IsNode(DstNId)) { Graph->AddNode(DstNId); }
    Graph->AddEdge(SrcNId, DstNId);
  }
  Graph->Defrag();
  return Graph;
}


template <class PGraph>
PGraph LoadEdgeList(const TStr& InFNm, const int& SrcColId, const int& DstColId, const char& Separator) {
  TSsParser Ss(InFNm, Separator);
  PGraph Graph = PGraph::TObj::New();
  int SrcNId, DstNId;
  while (Ss.Next()) {
    if (! Ss.GetInt(SrcColId, SrcNId) || ! Ss.GetInt(DstColId, DstNId)) { continue; }
    if (! Graph->IsNode(SrcNId)) { Graph->AddNode(SrcNId); }
    if (! Graph->IsNode(DstNId)) { Graph->AddNode(DstNId); }
    Graph->AddEdge(SrcNId, DstNId);
  }
  Graph->Defrag();
  return Graph;
}


template <class PGraph>
PGraph LoadEdgeListStr(const TStr& InFNm, const int& SrcColId, const int& DstColId) {
  TSsParser Ss(InFNm, ssfWhiteSep);
  PGraph Graph = PGraph::TObj::New();
  TStrHash<TInt> StrToNIdH(Mega(1), true);
  while (Ss.Next()) {
    const int SrcNId = StrToNIdH.AddKey(Ss[SrcColId]);
    const int DstNId = StrToNIdH.AddKey(Ss[DstColId]);
    if (! Graph->IsNode(SrcNId)) { Graph->AddNode(SrcNId); }
    if (! Graph->IsNode(DstNId)) { Graph->AddNode(DstNId); }
    Graph->AddEdge(SrcNId, DstNId);
  }
  Graph->Defrag();
  return Graph;
}


template <class PGraph>
PGraph LoadEdgeListStr(const TStr& InFNm, const int& SrcColId, const int& DstColId, TStrHash<TInt>& StrToNIdH) {
  TSsParser Ss(InFNm, ssfWhiteSep);
  PGraph Graph = PGraph::TObj::New();
  while (Ss.Next()) {
    const int SrcNId = StrToNIdH.AddKey(Ss[SrcColId]);
    const int DstNId = StrToNIdH.AddKey(Ss[DstColId]);
    if (! Graph->IsNode(SrcNId)) { Graph->AddNode(SrcNId); }
    if (! Graph->IsNode(DstNId)) { Graph->AddNode(DstNId); }
    Graph->AddEdge(SrcNId, DstNId);
  }
  Graph->Defrag();
  return Graph;
}


template <class PGraph>
PGraph LoadConnList(const TStr& InFNm) {
  TSsParser Ss(InFNm, ssfWhiteSep, true, true, true);
  PGraph Graph = PGraph::TObj::New();
  while (Ss.Next()) {
    if (! Ss.IsInt(0)) { continue; }
    const int SrcNId = Ss.GetInt(0);
    if (! Graph->IsNode(SrcNId)) { Graph->AddNode(SrcNId); }
    for (int dst = 1; dst < Ss.Len(); dst++) {
      const int DstNId = Ss.GetInt(dst);
      if (! Graph->IsNode(DstNId)) { Graph->AddNode(DstNId); }
      Graph->AddEdge(SrcNId, DstNId);
    }
  }
  Graph->Defrag();
  return Graph;
}


template <class PGraph> 
PGraph LoadConnListStr(const TStr& InFNm, TStrHash<TInt>& StrToNIdH) {
  TSsParser Ss(InFNm, ssfWhiteSep, true, true, true);
  PGraph Graph = PGraph::TObj::New();
  while (Ss.Next()) {
    const int SrcNId = StrToNIdH.AddDatId(Ss[0]);
    if (! Graph->IsNode(SrcNId)) { Graph->AddNode(SrcNId); }
    for (int dst = 1; dst < Ss.Len(); dst++) {
      const int DstNId = StrToNIdH.AddDatId(Ss[dst]);
      if (! Graph->IsNode(DstNId)) { Graph->AddNode(DstNId); }
      Graph->AddEdge(SrcNId, DstNId);
    }
  }
  Graph->Defrag();
  return Graph;
}

template <class PGraph>
PGraph LoadPajek(const TStr& InFNm) {
  PGraph Graph = PGraph::TObj::New();
  TSsParser Ss(InFNm, ssfSpaceSep, true, true, true);
  while ((Ss.Len()==0 || strstr(Ss[0], "*vertices") == NULL) && ! Ss.Eof()) {
    Ss.Next();  Ss.ToLc(); }

  bool EdgeList = true;
  EAssert(strstr(Ss[0], "*vertices") != NULL);
  while (Ss.Next()) {
    Ss.ToLc();
    if (Ss.Len()>0 && Ss[0][0] == '%') { continue; }
    if (strstr(Ss[0], "*arcslist")!=NULL || strstr(Ss[0],"*edgeslist")!=NULL) { EdgeList=false; break; } 
    if (strstr(Ss[0], "*arcs")!=NULL || strstr(Ss[0],"*edges")!=NULL) { break; }
    Graph->AddNode(Ss.GetInt(0));
  }

  while (Ss.Next()) {
    if (Ss.Len()>0 && Ss[0][0] == '%') { continue; }
    if (Ss.Len()>0 && Ss[0][0] == '*') { break; }
    if (EdgeList) {

      if (Ss.Len() >= 2 && Ss.IsInt(0) && Ss.IsInt(1)) {
        Graph->AddEdge(Ss.GetInt(0), Ss.GetInt(1));
      }
    } else {

      const int SrcNId = Ss.GetInt(0);
      for (int i = 1; i < Ss.Len(); i++) {
        Graph->AddEdge(SrcNId, Ss.GetInt(i)); }
    }
  }
  return Graph;
}

template <class PGraph>
void SaveEdgeList(const PGraph& Graph, const TStr& OutFNm, const TStr& Desc) {
  FILE *F = fopen(OutFNm.CStr(), "wt");
  if (HasGraphFlag(typename PGraph::TObj, gfDirected)) { fprintf(F, "# Directed graph: %s \n", OutFNm.CStr()); } 
  else { fprintf(F, "# Undirected graph (each unordered pair of nodes is saved once): %s\n", OutFNm.CStr()); }
  if (! Desc.Empty()) { fprintf(F, "# %s\n", Desc.CStr()); }
  fprintf(F, "# Nodes: %d Edges: %d\n", Graph->GetNodes(), Graph->GetEdges());
  if (HasGraphFlag(typename PGraph::TObj, gfDirected)) { fprintf(F, "# FromNodeId\tToNodeId\n"); }
  else { fprintf(F, "# NodeId\tNodeId\n"); }
  for (typename PGraph::TObj::TEdgeI ei = Graph->BegEI(); ei < Graph->EndEI(); ei++) {
    fprintf(F, "%d\t%d\n", ei.GetSrcNId(), ei.GetDstNId());
  }
  fclose(F);
}

template <class PGraph>
void SavePajek(const PGraph& Graph, const TStr& OutFNm) {
  TIntH NIdToIdH(Graph->GetNodes(), true);
  FILE *F = fopen(OutFNm.CStr(), "wt");
  fprintf(F, "*Vertices %d\n", Graph->GetNodes());
  int i = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++, i++) {
    fprintf(F, "%d  \"%d\" ic Red fos 10\n", i+1, NI.GetId());
    NIdToIdH.AddDat(NI.GetId(), i+1);
  }
  if (HasGraphFlag(typename PGraph::TObj, gfDirected)) {
    fprintf(F, "*Arcs %d\n", Graph->GetEdges()); }
  else {
    fprintf(F, "*Edges %d\n", Graph->GetEdges());
  }
  for (typename PGraph::TObj::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
    const int SrcNId = NIdToIdH.GetDat(EI.GetSrcNId());
    const int DstNId = NIdToIdH.GetDat(EI.GetDstNId());
    fprintf(F, "%d %d %d c Black\n", SrcNId, DstNId, 1);
  }
  fclose(F);
}



template <class PGraph>
void SavePajek(const PGraph& Graph, const TStr& OutFNm, const TIntStrH& NIdColorH) {
  TIntH NIdToIdH(Graph->GetNodes(), true);
  FILE *F = fopen(OutFNm.CStr(), "wt");
  fprintf(F, "*Vertices %d\n", Graph->GetNodes());
  int i = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++, i++) {
    fprintf(F, "%d  \"%d\" ic %s fos 10\n", i+1, NI.GetId(),
      NIdColorH.IsKey(NI.GetId()) ? NIdColorH.GetDat(NI.GetId()).CStr() : "Red");
    NIdToIdH.AddDat(NI.GetId(), i+1);
  }
  if (HasGraphFlag(typename PGraph::TObj, gfDirected)) {
    fprintf(F, "*Arcs %d\n", Graph->GetEdges()); }
  else {
    fprintf(F, "*Edges %d\n", Graph->GetEdges());
  }
  for (typename PGraph::TObj::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
    const int SrcNId = NIdToIdH.GetDat(EI.GetSrcNId());
    const int DstNId = NIdToIdH.GetDat(EI.GetDstNId());
    fprintf(F, "%d %d %d c Black\n", SrcNId, DstNId, 1);
  }
  fclose(F);
}




template <class PGraph>
void SavePajek(const PGraph& Graph, const TStr& OutFNm, const TIntStrH& NIdColorH, const TIntStrH& NIdLabelH) {
  TIntH NIdToIdH(Graph->GetNodes(), true);
  FILE *F = fopen(OutFNm.CStr(), "wt");
  fprintf(F, "*Vertices %d\n", Graph->GetNodes());
  int i = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++, i++) {
    fprintf(F, "%d  \"%s\" ic %s fos 10\n", i+1,
      NIdLabelH.IsKey(NI.GetId()) ? NIdLabelH.GetDat(NI.GetId()).CStr() : TStr::Fmt("%d", NI.GetId()).CStr(),
      NIdColorH.IsKey(NI.GetId()) ? NIdColorH.GetDat(NI.GetId()).CStr() : "Red");
    NIdToIdH.AddDat(NI.GetId(), i+1);
  }
  if (HasGraphFlag(typename PGraph::TObj, gfDirected)) {
    fprintf(F, "*Arcs %d\n", Graph->GetEdges()); }
  else {
    fprintf(F, "*Edges %d\n", Graph->GetEdges());
  }
  for (typename PGraph::TObj::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
    const int SrcNId = NIdToIdH.GetDat(EI.GetSrcNId());
    const int DstNId = NIdToIdH.GetDat(EI.GetDstNId());
    fprintf(F, "%d %d %d c Black\n", SrcNId, DstNId, 1);
  }
  fclose(F);
}





template <class PGraph>
void SavePajek(const PGraph& Graph, const TStr& OutFNm, const TIntStrH& NIdColorH, const TIntStrH& NIdLabelH, const TIntStrH& EIdColorH) {
  CAssert(HasGraphFlag(typename PGraph::TObj, gfMultiGraph));
  TIntH NIdToIdH(Graph->GetNodes(), true);
  FILE *F = fopen(OutFNm.CStr(), "wt");
  fprintf(F, "*Vertices %d\n", Graph->GetNodes());
  int i = 0;
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++, i++) {
    fprintf(F, "%d  \"%s\" ic %s fos 10\n", i+1,
      NIdLabelH.IsKey(NI.GetId()) ? NIdLabelH.GetDat(NI.GetId()).CStr() : TStr::Fmt("%d", NI.GetId()).CStr(),
      NIdColorH.IsKey(NI.GetId()) ? NIdColorH.GetDat(NI.GetId()).CStr() : "Red");
    NIdToIdH.AddDat(NI.GetId(), i+1);
  }
  if (HasGraphFlag(typename PGraph::TObj, gfDirected)) {
    fprintf(F, "*Arcs %d\n", Graph->GetEdges()); }
  else {
    fprintf(F, "*Edges %d\n", Graph->GetEdges());
  }
  for (typename PGraph::TObj::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
    const int SrcNId = NIdToIdH.GetDat(EI.GetSrcNId());
    const int DstNId = NIdToIdH.GetDat(EI.GetDstNId());
    fprintf(F, "%d %d 1 c %s\n", SrcNId, DstNId,
      EIdColorH.IsKey(EI.GetId()) ? EIdColorH.GetDat(EI.GetId()).CStr() : "Black");
  }
  fclose(F);
}


template <class PGraph>
void SaveMatlabSparseMtx(const PGraph& Graph, const TStr& OutFNm) {
  FILE *F = fopen(OutFNm.CStr(), "wt");
  TIntSet NIdSet(Graph->GetNodes());
  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    NIdSet.AddKey(NI.GetId());
  }
  for (typename PGraph::TObj::TEdgeI EI = Graph->BegEI(); EI < Graph->EndEI(); EI++) {
    const int Src = NIdSet.GetKeyId(EI.GetSrcNId())+1;
    const int Dst = NIdSet.GetKeyId(EI.GetDstNId())+1;
    fprintf(F, "%d\t%d\t1\n", Src, Dst);
    if (! HasGraphFlag(typename PGraph::TObj, gfDirected) && Src!=Dst) {
      fprintf(F, "%d\t%d\t1\n", Dst, Src);
    }
  }
  fclose(F);
}

template<class PGraph>
void SaveGViz(const PGraph& Graph, const TStr& OutFNm, const TStr& Desc, const bool& NodeLabels, const TIntStrH& NIdColorH) {
  const bool IsDir = HasGraphFlag(typename PGraph::TObj, gfDirected);
  FILE *F = fopen(OutFNm.CStr(), "wt");
  if (! Desc.Empty()) fprintf(F, "/*****\n%s\n*****/\n\n", Desc.CStr());
  if (IsDir) { fprintf(F, "digraph G {\n"); } else { fprintf(F, "graph G {\n"); }
  fprintf(F, "  graph [splines=false overlap=false]\n");


  fprintf(F, "  node  [shape=ellipse, width=0.3, height=0.3%s]\n", NodeLabels?"":", label=\"\"");


  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (NIdColorH.IsKey(NI.GetId())) {
      fprintf(F, "  %d [style=filled, fillcolor=\"%s\"];\n", NI.GetId(), NIdColorH.GetDat(NI.GetId()).CStr()); }
    else {
      fprintf(F, "  %d ;\n", NI.GetId());
    }
  }

  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (NI.GetOutDeg()==0 && NI.GetInDeg()==0 && !NIdColorH.IsKey(NI.GetId())) {
      fprintf(F, "%d;\n", NI.GetId()); }
    else {
      for (int e = 0; e < NI.GetOutDeg(); e++) {
        if (! IsDir && NI.GetId() > NI.GetOutNId(e)) { continue; }
        fprintf(F, "  %d %s %d;\n", NI.GetId(), IsDir?"->":"--", NI.GetOutNId(e));
      }
    }
  }
  if (! Desc.Empty()) {
    fprintf(F, "  label = \"\\n%s\\n\";", Desc.CStr());
    fprintf(F, "  fontsize=24;\n");
  }
  fprintf(F, "}\n");
  fclose(F);
}

template<class PGraph>
void SaveGViz(const PGraph& Graph, const TStr& OutFNm, const TStr& Desc, const TIntStrH& NIdLabelH) {
  const bool IsDir = Graph->HasFlag(gfDirected);
  FILE *F = fopen(OutFNm.CStr(), "wt");
  if (! Desc.Empty()) fprintf(F, "/*****\n%s\n*****/\n\n", Desc.CStr());
  if (IsDir) { fprintf(F, "digraph G {\n"); } else { fprintf(F, "graph G {\n"); }
  fprintf(F, "  graph [splines=true overlap=false]\n");
  fprintf(F, "  node  [shape=ellipse, width=0.3, height=0.3]\n");


  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    fprintf(F, "  %d [label=\"%s\"];\n", NI.GetId(), NIdLabelH.GetDat(NI.GetId()).CStr());
}

  for (typename PGraph::TObj::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    if (NI.GetOutDeg()==0 && NI.GetInDeg()==0 && ! NIdLabelH.IsKey(NI.GetId())) {
      fprintf(F, "%d;\n", NI.GetId()); }
    else {
      for (int e = 0; e < NI.GetOutDeg(); e++) {
        if (! IsDir && NI.GetId() > NI.GetOutNId(e)) { continue; }
        fprintf(F, "  %d %s %d;\n", NI.GetId(), IsDir?"->":"--", NI.GetOutNId(e));
      }
    }
  }
  if (! Desc.Empty()) {
    fprintf(F, "  label = \"\\n%s\\n\";", Desc.CStr());
    fprintf(F, "  fontsize=24;\n");
  }
  fprintf(F, "}\n");
  fclose(F);
}

}
