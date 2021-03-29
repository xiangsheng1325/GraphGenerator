void TForestFire::InfectAll() {
  InfectNIdV.Gen(Graph->GetNodes());
  for (TNGraph::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    InfectNIdV.Add(NI.GetId()); }
}
void TForestFire::InfectRnd(const int& NInfect) {
  IAssert(NInfect < Graph->GetNodes());
  TIntV NIdV(Graph->GetNodes(), 0);
  for (TNGraph::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    NIdV.Add(NI.GetId()); }
  NIdV.Shuffle(Rnd);
  InfectNIdV.Gen(NInfect, 0);
  for (int i = 0; i < NInfect; i++) {
    InfectNIdV.Add(NIdV[i]); }
}
void TForestFire::BurnExpFire() {
  const double OldFwdBurnProb = FwdBurnProb;
  const double OldBckBurnProb = BckBurnProb;
  const int NInfect = InfectNIdV.Len();
  const TNGraph& G = *Graph;
  TIntH BurnedNIdH;
  TIntV BurningNIdV = InfectNIdV;
  TIntV NewBurnedNIdV;
  bool HasAliveNbrs;
  int NBurned = NInfect, NDiedFire=0;
  for (int i = 0; i < InfectNIdV.Len(); i++) {
    BurnedNIdH.AddDat(InfectNIdV[i]); }
  NBurnedTmV.Clr(false);  NBurningTmV.Clr(false);  NewBurnedTmV.Clr(false);
  for (int time = 0; ; time++) {
    NewBurnedNIdV.Clr(false);

    for (int node = 0; node < BurningNIdV.Len(); node++) {
      const int& BurningNId = BurningNIdV[node];
      const TNGraph::TNodeI Node = G.GetNI(BurningNId);
      HasAliveNbrs = false;
      NDiedFire = 0;

      for (int e = 0; e < Node.GetOutDeg(); e++) {
        const int OutNId = Node.GetOutNId(e);
        if (! BurnedNIdH.IsKey(OutNId)) {
          HasAliveNbrs = true;
          if (Rnd.GetUniDev() < FwdBurnProb) {
            BurnedNIdH.AddDat(OutNId);  NewBurnedNIdV.Add(OutNId);  NBurned++; }
        }
      }

      if (BckBurnProb > 0.0) {
        for (int e = 0; e < Node.GetInDeg(); e++) {
          const int InNId = Node.GetInNId(e);
          if (! BurnedNIdH.IsKey(InNId)) {
            HasAliveNbrs = true;
            if (Rnd.GetUniDev() < BckBurnProb) {
              BurnedNIdH.AddDat(InNId);  NewBurnedNIdV.Add(InNId);  NBurned++; }
          }
        }
      }
      if (! HasAliveNbrs) { NDiedFire++; }
    }
    NBurnedTmV.Add(NBurned);
    NBurningTmV.Add(BurningNIdV.Len() - NDiedFire);
    NewBurnedTmV.Add(NewBurnedNIdV.Len());

    BurningNIdV.Swap(NewBurnedNIdV);
    if (BurningNIdV.Empty()) break;
    FwdBurnProb = FwdBurnProb * ProbDecay;
    BckBurnProb = BckBurnProb * ProbDecay;
  }
  BurnedNIdV.Gen(BurnedNIdH.Len(), 0);
  for (int i = 0; i < BurnedNIdH.Len(); i++) {
    BurnedNIdV.Add(BurnedNIdH.GetKey(i)); }
  FwdBurnProb = OldFwdBurnProb;
  BckBurnProb = OldBckBurnProb;
}
void TForestFire::BurnGeoFire() {
  const double OldFwdBurnProb=FwdBurnProb;
  const double OldBckBurnProb=BckBurnProb;
  const int& NInfect = InfectNIdV.Len();
  const TNGraph& G = *Graph;
  TIntH BurnedNIdH;
  TIntV BurningNIdV = InfectNIdV;
  TIntV NewBurnedNIdV;
  bool HasAliveInNbrs, HasAliveOutNbrs;
  TIntV AliveNIdV;
  int NBurned = NInfect, time;
  for (int i = 0; i < InfectNIdV.Len(); i++) {
    BurnedNIdH.AddDat(InfectNIdV[i]); }
  NBurnedTmV.Clr(false);  NBurningTmV.Clr(false);  NewBurnedTmV.Clr(false);
  for (time = 0; ; time++) {
    NewBurnedNIdV.Clr(false);
    for (int node = 0; node < BurningNIdV.Len(); node++) {
      const int& BurningNId = BurningNIdV[node];
      const TNGraph::TNodeI Node = G.GetNI(BurningNId);

      HasAliveOutNbrs = false;
      AliveNIdV.Clr(false);
      for (int e = 0; e < Node.GetOutDeg(); e++) {
        const int OutNId = Node.GetOutNId(e);
        if (! BurnedNIdH.IsKey(OutNId)) {
          HasAliveOutNbrs = true;  AliveNIdV.Add(OutNId); }
      }

      const int BurnNFwdLinks = Rnd.GetGeoDev(1.0-FwdBurnProb) - 1;
      if (HasAliveOutNbrs && BurnNFwdLinks > 0) {
        AliveNIdV.Shuffle(Rnd);
        for (int i = 0; i < TMath::Mn(BurnNFwdLinks, AliveNIdV.Len()); i++) {
          BurnedNIdH.AddDat(AliveNIdV[i]);
          NewBurnedNIdV.Add(AliveNIdV[i]);  NBurned++; }
      }

      if (BckBurnProb > 0.0) {

        HasAliveInNbrs = false;
        AliveNIdV.Clr(false);
        for (int e = 0; e < Node.GetInDeg(); e++) {
          const int InNId = Node.GetInNId(e);
          if (! BurnedNIdH.IsKey(InNId)) {
            HasAliveInNbrs = true;  AliveNIdV.Add(InNId); }
        }

        const int BurnNBckLinks = Rnd.GetGeoDev(1.0-BckBurnProb) - 1;
        if (HasAliveInNbrs && BurnNBckLinks > 0) {
          AliveNIdV.Shuffle(Rnd);
          for (int i = 0; i < TMath::Mn(BurnNBckLinks, AliveNIdV.Len()); i++) {
            BurnedNIdH.AddDat(AliveNIdV[i]);
            NewBurnedNIdV.Add(AliveNIdV[i]);  NBurned++; }
        }
      }
    }
    NBurnedTmV.Add(NBurned);  NBurningTmV.Add(BurningNIdV.Len());  NewBurnedTmV.Add(NewBurnedNIdV.Len());

    BurningNIdV.Swap(NewBurnedNIdV);
    if (BurningNIdV.Empty()) break;
    FwdBurnProb = FwdBurnProb * ProbDecay;
    BckBurnProb = BckBurnProb * ProbDecay;
  }
  BurnedNIdV.Gen(BurnedNIdH.Len(), 0);
  for (int i = 0; i < BurnedNIdH.Len(); i++) {
    BurnedNIdV.Add(BurnedNIdH.GetKey(i)); }
  FwdBurnProb = OldFwdBurnProb;
  BckBurnProb = OldBckBurnProb;
}
void TForestFire::PlotFire(const TStr& FNmPref, const TStr& Desc, const bool& PlotAllBurned) {
  TGnuPlot GnuPlot(FNmPref, TStr::Fmt("%s. ForestFire. G(%d, %d). Fwd:%g  Bck:%g  NInfect:%d",
    Desc.CStr(), Graph->GetNodes(), Graph->GetEdges(), FwdBurnProb(), BckBurnProb(), InfectNIdV.Len()));
  GnuPlot.SetXYLabel("TIME EPOCH", "Number of NODES");
  if (PlotAllBurned) GnuPlot.AddPlot(NBurnedTmV, gpwLinesPoints, "All burned nodes till time");
  GnuPlot.AddPlot(NBurningTmV, gpwLinesPoints, "Burning nodes at time");
  GnuPlot.AddPlot(NewBurnedTmV, gpwLinesPoints, "Newly burned nodes at time");
  GnuPlot.SavePng(TFile::GetUniqueFNm(TStr::Fmt("fireSz.%s_#.png", FNmPref.CStr())));
}
PNGraph TForestFire::GenGraph(const int& Nodes, const double& FwdProb, const double& BckProb) {
  TFfGGen Ff(false, 1, FwdProb, BckProb, 1.0, 0.0, 0.0);
  Ff.GenGraph(Nodes);
  return Ff.GetGraph();
}
int TFfGGen::TimeLimitSec = 30*60;
TFfGGen::TFfGGen(const bool& BurnExpFireP, const int& StartNNodes, const double& ForwBurnProb,
                 const double& BackBurnProb, const double& DecayProb, const double& Take2AmbasPrb, const double& OrphanPrb) :
 Graph(), BurnExpFire(BurnExpFireP), StartNodes(StartNNodes), FwdBurnProb(ForwBurnProb),
 BckBurnProb(BackBurnProb), ProbDecay(DecayProb), Take2AmbProb(Take2AmbasPrb), OrphanProb(OrphanPrb) {

}
TStr TFfGGen::GetParamStr() const {
  return TStr::Fmt("%s  FWD:%g  BCK:%g, StartNds:%d, Take2:%g, Orphan:%g, ProbDecay:%g",
    BurnExpFire?"EXP":"GEO", FwdBurnProb(), BckBurnProb(), StartNodes(), Take2AmbProb(), OrphanProb(), ProbDecay());
}
TFfGGen::TStopReason TFfGGen::AddNodes(const int& GraphNodes, const bool& FloodStop) {
  printf("\n***ForestFire:  %s  Nodes:%d  StartNodes:%d  Take2AmbProb:%g\n", BurnExpFire?"ExpFire":"GeoFire", GraphNodes, StartNodes(), Take2AmbProb());
  printf("                FwdBurnP:%g  BckBurnP:%g  ProbDecay:%g  Orphan:%g\n", FwdBurnProb(), BckBurnProb(), ProbDecay(), OrphanProb());
  TExeTm ExeTm;
  int Burned1=0, Burned2=0, Burned3=0;

  if (Graph.Empty()) { Graph = PNGraph::New(); }
  if (Graph->GetNodes() == 0) {
    for (int n = 0; n < StartNodes; n++) { Graph->AddNode(); }
  }
  int NEdges = Graph->GetEdges();

  TRnd Rnd(0);
  TForestFire ForestFire(Graph, FwdBurnProb, BckBurnProb, ProbDecay, 0);

  for (int NNodes = Graph->GetNodes()+1; NNodes <= GraphNodes; NNodes++) {
    const int NewNId = Graph->AddNode(-1);
    IAssert(NewNId == Graph->GetNodes()-1);

    if (OrphanProb == 0.0 || Rnd.GetUniDev() > OrphanProb) {

      if (Take2AmbProb == 0.0 || Rnd.GetUniDev() > Take2AmbProb || NewNId < 2) {
        ForestFire.Infect(Rnd.GetUniDevInt(NewNId));
      } else {
        const int AmbassadorNId1 = Rnd.GetUniDevInt(NewNId);
        int AmbassadorNId2 = Rnd.GetUniDevInt(NewNId);
        while (AmbassadorNId1 == AmbassadorNId2) {
          AmbassadorNId2 = Rnd.GetUniDevInt(NewNId); }
        ForestFire.Infect(TIntV::GetV(AmbassadorNId1, AmbassadorNId2));
      }

      if (BurnExpFire) { ForestFire.BurnExpFire(); }
      else { ForestFire.BurnGeoFire(); }

      for (int e = 0; e < ForestFire.GetBurned(); e++) {
        Graph->AddEdge(NewNId, ForestFire.GetBurnedNId(e));
        NEdges++;
      }
      Burned1=Burned2;  Burned2=Burned3;  Burned3=ForestFire.GetBurned();
    } else {

      Burned1=Burned2;  Burned2=Burned3;  Burned3=0;
    }
    if (NNodes % Kilo(1) == 0) {
      printf("(%d, %d)  burned: [%d,%d,%d]  [%s]\n", NNodes, NEdges, Burned1, Burned2, Burned3, ExeTm.GetStr()); }
    if (FloodStop && NEdges>GraphNodes && (NEdges/double(NNodes)>1000.0)) {
      printf(". FLOOD. G(%6d, %6d)\n", NNodes, NEdges);  return srFlood; }
    if (NNodes % 1000 == 0 && TimeLimitSec > 0 && ExeTm.GetSecs() > TimeLimitSec) {
      printf(". TIME LIMIT. G(%d, %d)\n", Graph->GetNodes(), Graph->GetEdges());
      return srTimeLimit; }
  }
  IAssert(Graph->GetEdges() == NEdges);
  return srOk;
}
TFfGGen::TStopReason TFfGGen::GenGraph(const int& GraphNodes, const bool& FloodStop) {
  Graph = PNGraph::New();
  return AddNodes(GraphNodes, FloodStop);
}
TFfGGen::TStopReason TFfGGen::GenGraph(const int& GraphNodes, PGStatVec& EvolStat, const bool& FloodStop) {
  int GrowthStatNodes = 100;
  Graph = PNGraph::New();
  AddNodes(StartNodes);
  TStopReason SR = srUndef;
  while (Graph->GetNodes() < GraphNodes) {
    SR = AddNodes(GrowthStatNodes, FloodStop);
    if (SR != srOk) { return SR; }
    EvolStat->Add(Graph, TSecTm(Graph->GetNodes()));
    GrowthStatNodes = int(1.5*GrowthStatNodes);
  }
  return SR;
}
void TFfGGen::PlotFireSize(const TStr& FNmPref, const TStr& DescStr) {
  TGnuPlot GnuPlot("fs."+FNmPref, TStr::Fmt("%s. Fire size. G(%d, %d)",
    DescStr.CStr(), Graph->GetNodes(), Graph->GetEdges()));
  GnuPlot.SetXYLabel("Vertex id (iterations)", "Fire size (node out-degree)");
  TFltPrV IdToOutDegV;
  for (TNGraph::TNodeI NI = Graph->BegNI(); NI < Graph->EndNI(); NI++) {
    IdToOutDegV.Add(TFltPr(NI.GetId(), NI.GetOutDeg())); }
  IdToOutDegV.Sort();
  GnuPlot.AddPlot(IdToOutDegV, gpwImpulses, "Node out-degree");
  GnuPlot.SavePng();
}
void TFfGGen::GenFFGraphs(const double& FProb, const double& BProb, const TStr& FNm) {
  const int NRuns = 10;
  const int NNodes = 10000;
  TGStat::NDiamRuns = 10;






  TVec<PGStatVec> GAtTmV;
  TFfGGen FF(false, 1, FProb, BProb, 1.0, 0, 0);
  for (int r = 0; r < NRuns; r++) {
    PGStatVec GV = TGStatVec::New(tmuNodes, TGStat::AllStat());
    FF.GenGraph(NNodes, GV, true);
    for (int i = 0; i < GV->Len(); i++) {
      if (i == GAtTmV.Len()) {
        GAtTmV.Add(TGStatVec::New(tmuNodes, TGStat::AllStat()));
      }
      GAtTmV[i]->Add(GV->At(i));
    }
    IAssert(GAtTmV.Len() == GV->Len());
  }
  PGStatVec AvgStat = TGStatVec::New(tmuNodes, TGStat::AllStat());
  for (int i = 0; i < GAtTmV.Len(); i++) {
    AvgStat->Add(GAtTmV[i]->GetAvgGStat(false));
  }
  AvgStat->PlotAllVsX(gsvNodes, FNm, TStr::Fmt("Forest Fire: F:%g  B:%g (%d runs)", FProb, BProb, NRuns));
  AvgStat->Last()->PlotAll(FNm, TStr::Fmt("Forest Fire: F:%g  B:%g (%d runs)", FProb, BProb, NRuns));
}
int TUndirFFire::BurnGeoFire(const int& StartNId) {
  BurnedSet.Clr(false);
  BurningNIdV.Clr(false);  
  NewBurnedNIdV.Clr(false);
  AliveNIdV.Clr(false);
  const TUNGraph& G = *Graph;
  int NBurned = 1;
  BurnedSet.AddKey(StartNId);
  BurningNIdV.Add(StartNId);
  while (! BurningNIdV.Empty()) {
    for (int node = 0; node < BurningNIdV.Len(); node++) {
      const int& BurningNId = BurningNIdV[node];
      const TUNGraph::TNodeI& Node = G.GetNI(BurningNId);

      AliveNIdV.Clr(false);
      for (int e = 0; e < Node.GetOutDeg(); e++) {
        const int OutNId = Node.GetOutNId(e);
        if (! BurnedSet.IsKey(OutNId)) {
          AliveNIdV.Add(OutNId); }
      }

      const int BurnNLinks = Rnd.GetGeoDev(1.0-BurnProb) - 1;
      if (! AliveNIdV.Empty() && BurnNLinks > 0) {
        AliveNIdV.Shuffle(Rnd);
        for (int i = 0; i < TMath::Mn(BurnNLinks, AliveNIdV.Len()); i++) {
          BurnedSet.AddKey(AliveNIdV[i]);
          NewBurnedNIdV.Add(AliveNIdV[i]);
          NBurned++;
        }
      }
    }
    BurningNIdV.Swap(NewBurnedNIdV);

    NewBurnedNIdV.Clr(false);
  }
  IAssert(BurnedSet.Len() == NBurned);
  return NBurned;
}
TFfGGen::TStopReason TUndirFFire::AddNodes(const int& GraphNodes, const bool& FloodStop) {
  printf("\n***Undirected GEO ForestFire: graph(%d,%d) add %d nodes, burn prob %.3f\n", 
    Graph->GetNodes(), Graph->GetEdges(), GraphNodes, BurnProb);
  TExeTm ExeTm;
  int Burned1=0, Burned2=0, Burned3=0;
  TIntPrV NodesEdgesV;

  if (Graph.Empty()) { Graph = PUNGraph::New(); }
  if (Graph->GetNodes() == 0) { Graph->AddNode(); }
  int NEdges = Graph->GetEdges();

  for (int NNodes = Graph->GetNodes()+1; NNodes <= GraphNodes; NNodes++) {
    const int NewNId = Graph->AddNode(-1);
    IAssert(NewNId == Graph->GetNodes()-1);
    const int StartNId = Rnd.GetUniDevInt(NewNId);
    const int NBurned = BurnGeoFire(StartNId);

    for (int e = 0; e < NBurned; e++) {
      Graph->AddEdge(NewNId, GetBurnedNId(e)); }
    NEdges += NBurned;
    Burned1=Burned2;  Burned2=Burned3;  Burned3=NBurned;
    if (NNodes % Kilo(1) == 0) {
      printf("(%d, %d)    burned: [%d,%d,%d]  [%s]\n", NNodes, NEdges, Burned1, Burned2, Burned3, ExeTm.GetStr()); 
      NodesEdgesV.Add(TIntPr(NNodes, NEdges));
    }
    if (FloodStop && NEdges>1000 && NEdges/double(NNodes)>100.0) {
      printf("!!! FLOOD. G(%6d, %6d)\n", NNodes, NEdges);  return TFfGGen::srFlood; }
  }
  printf("\n");
  IAssert(Graph->GetEdges() == NEdges);
  return TFfGGen::srOk;
}
