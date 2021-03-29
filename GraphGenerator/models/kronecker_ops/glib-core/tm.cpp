bool TTmInfo::InitP=false;
TStrV TTmInfo::UsMonthNmV;
TStrV TTmInfo::SiMonthNmV;
TStrV TTmInfo::UsDayOfWeekNmV;
TStrV TTmInfo::SiDayOfWeekNmV;
void TTmInfo::InitMonthNmV(){

  UsMonthNmV.Add("jan"); UsMonthNmV.Add("feb"); UsMonthNmV.Add("mar");
  UsMonthNmV.Add("apr"); UsMonthNmV.Add("may"); UsMonthNmV.Add("jun");
  UsMonthNmV.Add("jul"); UsMonthNmV.Add("aug"); UsMonthNmV.Add("sep");
  UsMonthNmV.Add("oct"); UsMonthNmV.Add("nov"); UsMonthNmV.Add("dec");
  IAssert(UsMonthNmV.Len()==12);

  SiMonthNmV.Add("jan"); SiMonthNmV.Add("feb"); SiMonthNmV.Add("mar");
  SiMonthNmV.Add("apr"); SiMonthNmV.Add("maj"); SiMonthNmV.Add("jun");
  SiMonthNmV.Add("jul"); SiMonthNmV.Add("avg"); SiMonthNmV.Add("sep");
  SiMonthNmV.Add("okt"); SiMonthNmV.Add("nov"); SiMonthNmV.Add("dec");
  IAssert(SiMonthNmV.Len()==12);
}
void TTmInfo::InitDayOfWeekNmV(){

  UsDayOfWeekNmV.Add("sun"); UsDayOfWeekNmV.Add("mon");
  UsDayOfWeekNmV.Add("tue"); UsDayOfWeekNmV.Add("wed");
  UsDayOfWeekNmV.Add("thu"); UsDayOfWeekNmV.Add("fri");
  UsDayOfWeekNmV.Add("sat");
  IAssert(UsDayOfWeekNmV.Len()==7);

  SiDayOfWeekNmV.Add("ned"); SiDayOfWeekNmV.Add("pon");
  SiDayOfWeekNmV.Add("tor"); SiDayOfWeekNmV.Add("sre");
  SiDayOfWeekNmV.Add("cet"); SiDayOfWeekNmV.Add("pet");
  SiDayOfWeekNmV.Add("sob");
  IAssert(SiDayOfWeekNmV.Len()==7);
}
int TTmInfo::GetMonthN(const TStr& MonthNm, const TLoc& Loc){
  EnsureInit();
  int MonthN=-1;
  switch (Loc){
    case lUs: MonthN=UsMonthNmV.SearchForw(MonthNm.GetLc()); break;
    case lSi: MonthN=SiMonthNmV.SearchForw(MonthNm.GetLc()); break;
    default: Fail;
  }
  if (MonthN==-1){return -1;} else {return MonthN+1;}
}
TStr TTmInfo::GetMonthNm(const int& MonthN, const TLoc& Loc){
  EnsureInit();
  IAssert((1<=MonthN)&&(MonthN<=12));
  switch (Loc){
    case lUs: return UsMonthNmV[MonthN-1];
    case lSi: return SiMonthNmV[MonthN-1];
    default: Fail; return TStr();
  }
}
int TTmInfo::GetDayOfWeekN(const TStr& DayOfWeekNm, const TLoc& Loc){
  EnsureInit();
  int DayOfWeekN=-1;
  switch (Loc){
    case lUs: DayOfWeekN=UsDayOfWeekNmV.SearchForw(DayOfWeekNm.GetLc()); break;
    case lSi: DayOfWeekN=SiDayOfWeekNmV.SearchForw(DayOfWeekNm.GetLc()); break;
    default: Fail;
  }
  if (DayOfWeekN==-1){return -1;} else {return DayOfWeekN+1;}
}
TStr TTmInfo::GetDayOfWeekNm(const int& DayOfWeekN, const TLoc& Loc){
  EnsureInit();
  IAssert((1<=DayOfWeekN)&&(DayOfWeekN<=7));
  switch (Loc){
    case lUs: return UsDayOfWeekNmV[DayOfWeekN-1];
    case lSi: return SiDayOfWeekNmV[DayOfWeekN-1];
    default: Fail; return TStr();
  }
}
TStr TTmInfo::GetHmFromMins(const int& Mins){
  return TInt::GetStr(Mins/60, "%02d")+":"+TInt::GetStr(Mins%60, "%02d");
}
int TTmInfo::GetTmUnitSecs(const TTmUnit& TmUnit) {
  switch(TmUnit) {
    case tmuYear : return 365*24*3600;
    case tmuMonth : return 31*24*3600;
    case tmuWeek : return 7*24*3600;
    case tmuDay : return 24*3600;
    case tmu12Hour : return 12*3600;
    case tmu6Hour : return 6*3600;
    case tmu4Hour : return 4*3600;
    case tmu2Hour : return 2*3600;
    case tmu1Hour : return 1*3600;
    case tmu30Min : return 30*60;
    case tmu15Min : return 15*60;
    case tmu10Min : return 10*60;
    case tmu1Min : return 60;
    case tmu1Sec : return 1;
    case tmuNodes : Fail;
    case tmuEdges : Fail;
    default: Fail;
  }
  return -1;
}
TStr TTmInfo::GetTmUnitStr(const TTmUnit& TmUnit) {
  switch(TmUnit) {
    case tmuYear : return "Year";
    case tmuMonth : return "Month";
    case tmuWeek : return "Week";
    case tmuDay : return "Day";
    case tmu12Hour : return "12 Hours";
    case tmu6Hour : return "6 Hours";
    case tmu4Hour : return "4 Hours";
    case tmu2Hour : return "2 Hours";
    case tmu1Hour : return "1 Hour";
    case tmu30Min : return "30 Minutes";
    case tmu15Min : return "15 Minutes";
    case tmu10Min : return "10 Minutes";
    case tmu1Min : return "Minute";
    case tmu1Sec : return "Second";
    case tmuNodes : return "Nodes";
    case tmuEdges : return "Edges";
    default: Fail;
  }
  return TStr::GetNullStr();
}
TStr TTmInfo::GetTmZoneDiffStr(const TStr& TmZoneStr){
  if (TmZoneStr=="A"){ return "+1000";}
  if (TmZoneStr=="ACDT"){ return "+1030";}
  if (TmZoneStr=="ACST"){ return "+0930";}
  if (TmZoneStr=="ADT"){ return "-0300";}
  if (TmZoneStr=="AEDT"){ return "+1100";}
  if (TmZoneStr=="AEST"){ return "+1000";}
  if (TmZoneStr=="AKDT"){ return "-0800";}
  if (TmZoneStr=="AKST"){ return "-0900";}
  if (TmZoneStr=="AST"){ return "-0400";}
  if (TmZoneStr=="AWDT"){ return "+0900";}
  if (TmZoneStr=="AWST"){ return "+0800";}
  if (TmZoneStr=="B"){ return "+0200";}
  if (TmZoneStr=="BST"){ return "+0100";}
  if (TmZoneStr=="C"){ return "+0300";}
  if (TmZoneStr=="CDT"){ return "-0500";}
  if (TmZoneStr=="CDT"){ return "+1030";}
  if (TmZoneStr=="CEDT"){ return "+0200";}
  if (TmZoneStr=="CEST"){ return "+0200";}
  if (TmZoneStr=="CET"){ return "+0100";}
  if (TmZoneStr=="CST"){ return "-0600";}
  if (TmZoneStr=="CST"){ return "+1030";}
  if (TmZoneStr=="CST"){ return "+0930";}
  if (TmZoneStr=="CXT"){ return "+0700";}
  if (TmZoneStr=="D"){ return "+0400";}
  if (TmZoneStr=="E"){ return "+0500";}
  if (TmZoneStr=="EDT"){ return "-0400";}
  if (TmZoneStr=="EDT"){ return "+1100";}
  if (TmZoneStr=="EEDT"){ return "+0300";}
  if (TmZoneStr=="EEST"){ return "+0300";}
  if (TmZoneStr=="EET"){ return "+0200";}
  if (TmZoneStr=="EST"){ return "-0500";}
  if (TmZoneStr=="EST"){ return "+1100";}
  if (TmZoneStr=="EST"){ return "+1000";}
  if (TmZoneStr=="F"){ return "+0600";}
  if (TmZoneStr=="G"){ return "+0700";}
  if (TmZoneStr=="GMT"){ return "+0000";}
  if (TmZoneStr=="H"){ return "+0800";}
  if (TmZoneStr=="HAA"){ return "-0300";}
  if (TmZoneStr=="HAC"){ return "-0500";}
  if (TmZoneStr=="HADT"){ return "-0900";}
  if (TmZoneStr=="HAE"){ return "-0400";}
  if (TmZoneStr=="HAP"){ return "-0700";}
  if (TmZoneStr=="HAR"){ return "-0600";}
  if (TmZoneStr=="HAST"){ return "-1000";}
  if (TmZoneStr=="HAT"){ return "-0230";}
  if (TmZoneStr=="HAY"){ return "-0800";}
  if (TmZoneStr=="HNA"){ return "-0400";}
  if (TmZoneStr=="HNC"){ return "-0600";}
  if (TmZoneStr=="HNE"){ return "-0500";}
  if (TmZoneStr=="HNP"){ return "-0800";}
  if (TmZoneStr=="HNR"){ return "-0700";}
  if (TmZoneStr=="HNT"){ return "-0330";}
  if (TmZoneStr=="HNY"){ return "-0900";}
  if (TmZoneStr=="I"){ return "+0900";}
  if (TmZoneStr=="IST"){ return "+0100";}
  if (TmZoneStr=="K"){ return "+1000";}
  if (TmZoneStr=="L"){ return "+1100";}
  if (TmZoneStr=="M"){ return "+1200";}
  if (TmZoneStr=="MDT"){ return "-0600";}
  if (TmZoneStr=="MESZ"){ return "+0200";}
  if (TmZoneStr=="MEZ"){ return "+0100";}
  if (TmZoneStr=="MSD"){ return "+0400";}
  if (TmZoneStr=="MSK"){ return "+0300";}
  if (TmZoneStr=="MST"){ return "-0700";}
  if (TmZoneStr=="N"){ return "-0100";}
  if (TmZoneStr=="NDT"){ return "-0230";}
  if (TmZoneStr=="NFT"){ return "+ 11:30";}
  if (TmZoneStr=="NST"){ return "-0330";}
  if (TmZoneStr=="O"){ return "-0200";}
  if (TmZoneStr=="P"){ return "-0300";}
  if (TmZoneStr=="PDT"){ return "-0700";}
  if (TmZoneStr=="PST"){ return "-0800";}
  if (TmZoneStr=="Q"){ return "-0400";}
  if (TmZoneStr=="R"){ return "-0500";}
  if (TmZoneStr=="S"){ return "-0600";}
  if (TmZoneStr=="T"){ return "-0700";}
  if (TmZoneStr=="U"){ return "-0800";}
  if (TmZoneStr=="UTC"){ return "+0000";}
  if (TmZoneStr=="V"){ return "-0900";}
  if (TmZoneStr=="W"){ return "-1000";}
  if (TmZoneStr=="WDT"){ return "+0900";}
  if (TmZoneStr=="WEDT"){ return "+0100";}
  if (TmZoneStr=="WEST"){ return "+0100";}
  if (TmZoneStr=="WET"){ return "+0000";}
  if (TmZoneStr=="WST"){ return "+0900";}
  if (TmZoneStr=="WST"){ return "+0800";}
  if (TmZoneStr=="X"){ return "-1100";}
  if (TmZoneStr=="Y"){ return "-1200";}
  if (TmZoneStr=="Z"){ return "+0000";}
  return "-0000";
}
const int TTmInfo::SunN=1; const int TTmInfo::MonN=2;
const int TTmInfo::TueN=3; const int TTmInfo::WedN=4;
const int TTmInfo::ThuN=5; const int TTmInfo::FriN=6;
const int TTmInfo::SatN=7;
const int TTmInfo::JanN=1; const int TTmInfo::FebN=2;
const int TTmInfo::MarN=3; const int TTmInfo::AprN=4;
const int TTmInfo::MayN=5; const int TTmInfo::JunN=6;
const int TTmInfo::JulN=7; const int TTmInfo::AugN=8;
const int TTmInfo::SepN=9; const int TTmInfo::OctN=10;
const int TTmInfo::NovN=11; const int TTmInfo::DecN=12;
int TJulianDate::LastJulianDate=15821004;
int TJulianDate::LastJulianDateN=2299160;
int TJulianDate::GetJulianDateN(int d, int m, int y){
  IAssert(y != 0);
  int julian = -1;
  long jdn;
  if (julian < 0){
    julian = (((y * 100L) + m) * 100 + d  <=  LastJulianDate);}
  if (y < 0){
    y++;}
  if (julian){
    jdn = 367L * y - 7 * (y + 5001L + (m - 9) / 7) / 4
     + 275 * m / 9 + d + 1729777L;
  } else {
    jdn = (long)(d - 32076)
     + 1461L * (y + 4800L + (m - 14) / 12) / 4
     + 367 * (m - 2 - (m - 14) / 12 * 12) / 12
     - 3 * ((y + 4900L + (m - 14) / 12) / 100) / 4
     + 1;
  }
  return (int) jdn;
}
void TJulianDate::GetCalendarDate(int jdn, int& dd, int& mm, int& yy){
  int julian = -1;
  long x, z, m, d, y;
  long daysPer400Years = 146097L;
  long fudgedDaysPer4000Years = 1460970L + 31;
  if (julian < 0){
    julian = (jdn <= LastJulianDateN);}
  x = jdn + 68569L;
  if (julian){
    x+=38;
    daysPer400Years = 146100L;
    fudgedDaysPer4000Years = 1461000L + 1;
  }
  z = 4 * x / daysPer400Years;
  x = x - (daysPer400Years * z + 3) / 4;
  y = 4000 * (x + 1) / fudgedDaysPer4000Years;
  x = x - 1461 * y / 4 + 31;
  m = 80 * x / 2447;
  d = x - 2447 * m / 80;
  x = m / 11;
  m = m + 2 - 12 * x;
  y = 100 * (z - 49) + y + x;
  yy = (int)y;
  mm = (int)m;
  dd = (int)d;
  if (yy <= 0){
   (yy)--;}
}
bool TSecTm::GetTmSec(const int& YearN, const int& MonthN, const int& DayN, const int& HourN, const int& MinN, const int& SecN, uint& AbsSec) {
  AbsSec = 0;






  struct tm Tm;
  Tm.tm_year=YearN-1900; Tm.tm_mon=MonthN-1; Tm.tm_mday=DayN;
  Tm.tm_hour=HourN; Tm.tm_min=MinN; Tm.tm_sec=SecN;
  Tm.tm_wday=1;  Tm.tm_yday=1;
  Tm.tm_isdst=-1;
  return TSecTm::GetTmSec(Tm, AbsSec);
}
time_t TSecTm::MkGmTime(struct tm *t) {
  static const int m_to_d[12] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
  short month, year;
  time_t result;
  month = t->tm_mon;
  year = t->tm_year + month / 12 + 1900;
  month %= 12;
  if (month < 0) {
    year -= 1;
    month += 12; }
  result = (year - 1970) * 365 + (year - 1969) / 4 + m_to_d[month];
  result = (year - 1970) * 365 + m_to_d[month];
  if (month <= 1) { year -= 1; }
  result += (year - 1968) / 4;
  result -= (year - 1900) / 100;
  result += (year - 1600) / 400;
  result += t->tm_mday;
  result -= 1;
  result *= 24;
  result += t->tm_hour;
  result *= 60;
  result += t->tm_min;
  result *= 60;
  result += t->tm_sec;
  return result;
}
bool TSecTm::GetTmSec(struct tm& Tm, uint& AbsSec) {
  const time_t GmtTime = MkGmTime(&Tm);
  EAssertR(uint(GmtTime) < TUInt::Mx,
    TStr::Fmt("Time out of range: %d/%d/%d %02d:%02d:%02d",
    Tm.tm_year, Tm.tm_mon, Tm.tm_mday, Tm.tm_hour, Tm.tm_min, Tm.tm_sec).CStr());
  AbsSec = uint(GmtTime);
  return GmtTime >= 0;
}
bool TSecTm::GetTmStruct(const uint& AbsSec, struct tm& Tm) {
  const time_t TimeT = time_t(AbsSec);
  #if defined(GLib_MSC)
  return _gmtime64_s(&Tm, &TimeT) == 0;
  #elif defined(GLib_BCB)
  Tm=*gmtime(&TimeT); return true;
  #else
  return gmtime_r(&TimeT, &Tm) != NULL;
  #endif
}
TSecTm::TSecTm(const int& YearN, const int& MonthN, const int& DayN,
 const int& HourN, const int& MinN, const int& SecN) : AbsSecs(TUInt::Mx){
  GetTmSec(YearN, MonthN, DayN, HourN, MinN, SecN, AbsSecs.Val);
}
TSecTm::TSecTm(const TTm& Tm): AbsSecs(
 TSecTm(Tm.GetYear(), Tm.GetMonth(), Tm.GetDay(), Tm.GetHour(),
   Tm.GetMin(), Tm.GetSec()).GetAbsSecs()) { }
TSecTm::TSecTm(const PXmlTok& XmlTok) {
  const int Year = XmlTok->GetIntArgVal("Year");
  const int Month = XmlTok->GetIntArgVal("Month");
  const int Day = XmlTok->GetIntArgVal("Day");
  const int Hour = XmlTok->GetIntArgVal("Hour");
  const int Min = XmlTok->GetIntArgVal("Min");
  const int Sec = XmlTok->GetIntArgVal("Sec");
  AbsSecs = TSecTm(Year, Month, Day, Hour, Min, Sec).GetAbsSecs();
}
PXmlTok TSecTm::GetXmlTok() const {
  PXmlTok NodeTok = TXmlTok::New("NodeTime");
  NodeTok->AddArg("Year", GetYearN());
  NodeTok->AddArg("Month", GetMonthN());
  NodeTok->AddArg("Day", GetDayN());
  NodeTok->AddArg("Hour", GetHourN());
  NodeTok->AddArg("Min", GetMinN());
  NodeTok->AddArg("Sec", GetSecN());
  return NodeTok;
}
TStr TSecTm::GetStr(const TLoc& Loc) const {
  if (IsDef()) {
    struct tm Tm;
    IAssert(GetTmStruct(AbsSecs(), Tm));

    return TStr::Fmt("%s %s %d %02d:%02d:%02d %d",
      TTmInfo::GetDayOfWeekNm(Tm.tm_wday + 1, Loc).CStr(),
      TTmInfo::GetMonthNm(Tm.tm_mon + 1, Loc).CStr(),
      Tm.tm_mday, Tm.tm_hour, Tm.tm_min, Tm.tm_sec, Tm.tm_year+1900);
  } else {
    return "Undef";
  }
}
TStr TSecTm::GetStr(const TTmUnit& TmUnit) const {
  if (TmUnit == tmuYear) {
    return TInt::GetStr(GetYearN()); }
  else if (TmUnit == tmuMonth) {
    return TStr::Fmt("%04d-%02d", GetYearN(), GetMonthN()); }
  else if (TmUnit == tmuDay) {
    return TStr::Fmt("%04d-%02d-%02d", GetYearN(), GetMonthN(), GetDayN()); }
  else {
    return TStr::Fmt("%04d-%02d-%02d %02d:%02d:%02d",
      GetYearN(), GetMonthN(), GetDayN(), GetHourN(), GetMinN(), GetSecN());
  }
}
TStr TSecTm::GetDtStr(const TLoc& Loc) const {
  if (IsDef()){
    struct tm Tm;
    IAssert(GetTmStruct(AbsSecs(), Tm));
    return TStr::Fmt("%s %s %d %d",
      TTmInfo::GetDayOfWeekNm(Tm.tm_wday + 1, Loc).CStr(),
      TTmInfo::GetMonthNm(Tm.tm_mon + 1, Loc).CStr(), Tm.tm_year+1900);
  } else {
    return "Undef";
  }
}
TStr TSecTm::GetDtMdyStr() const {
  struct tm Tm;
  IAssert(GetTmStruct(AbsSecs(), Tm));
  return TStr::Fmt("%02d/%02d%/%04d", Tm.tm_mon+1, Tm.tm_mday, Tm.tm_year+1900);
}
TStr TSecTm::GetDtYmdStr() const {
  struct tm Tm;
  IAssert(GetTmStruct(AbsSecs(), Tm));
  return TStr::Fmt("%04d-%02d-%02d", Tm.tm_year+1900, Tm.tm_mon+1, Tm.tm_mday);
}
TStr TSecTm::GetYmdTmStr() const {
  struct tm Tm;
  IAssert(GetTmStruct(AbsSecs(), Tm));
  return TStr::Fmt("%04d-%02d-%02d %02d:%02d:%02d", Tm.tm_year+1900, Tm.tm_mon+1, Tm.tm_mday, Tm.tm_hour, Tm.tm_min, Tm.tm_sec);
}
TStr TSecTm::GetYmdTmStr2() const {
  struct tm Tm;
  IAssert(GetTmStruct(AbsSecs(), Tm));
  return TStr::Fmt("%04d-%02d-%02d-%02d:%02d:%02d", Tm.tm_year+1900, Tm.tm_mon+1, Tm.tm_mday, Tm.tm_hour, Tm.tm_min, Tm.tm_sec);
}
TStr TSecTm::GetTmStr() const {
  if (IsDef()){
    struct tm Tm;
    IAssert(GetTmStruct(AbsSecs(), Tm));
    return TStr::Fmt("%02d:%02d:%02d", Tm.tm_hour, Tm.tm_min, Tm.tm_sec);
  } else {
    return "Undef";
  }
}
TStr TSecTm::GetTmMinStr() const {
  if (IsDef()){
    struct tm Tm;
    IAssert(GetTmStruct(AbsSecs(), Tm));
    return TStr::Fmt("%02d:%02d", Tm.tm_min, Tm.tm_sec);
  } else {
    return "Undef";
  }
}
TStr TSecTm::GetDtTmSortStr() const {
  return
    TInt::GetStr(GetYearN(), "%04d")+"/"+
    TInt::GetStr(GetMonthN(), "%02d")+"/"+
    TInt::GetStr(GetDayN(), "%02d")+" "+
    TInt::GetStr(GetHourN(), "%02d")+":"+
    TInt::GetStr(GetMinN(), "%02d")+":"+
    TInt::GetStr(GetSecN(), "%02d");
}
TStr TSecTm::GetDtTmSortFNmStr() const {
  return
    TInt::GetStr(GetYearN(), "%04d")+"-"+
    TInt::GetStr(GetMonthN(), "%02d")+"-"+
    TInt::GetStr(GetDayN(), "%02d")+"_"+
    TInt::GetStr(GetHourN(), "%02d")+"-"+
    TInt::GetStr(GetMinN(), "%02d")+"-"+
    TInt::GetStr(GetSecN(), "%02d");
}
int TSecTm::GetYearN() const {
  struct tm Tm;
  IAssert(IsDef() && GetTmStruct(AbsSecs(), Tm));
  return Tm.tm_year+1900;
}
int TSecTm::GetMonthN() const {
  struct tm Tm;
  IAssert(IsDef() && GetTmStruct(AbsSecs(), Tm));
  return Tm.tm_mon+1;
}
TStr TSecTm::GetMonthNm(const TLoc& Loc) const {
  struct tm Tm;
  IAssert(IsDef() && GetTmStruct(AbsSecs(), Tm));
  return TTmInfo::GetMonthNm(Tm.tm_mon+1, Loc);
}
int TSecTm::GetDayN() const {
  struct tm Tm;
  IAssert(IsDef() && GetTmStruct(AbsSecs(), Tm));
  return Tm.tm_mday;
}
int TSecTm::GetDayOfWeekN() const {
  struct tm Tm;
  IAssert(IsDef() && GetTmStruct(AbsSecs(), Tm));
  return Tm.tm_wday + 1;
}
TStr TSecTm::GetDayOfWeekNm(const TLoc& Loc) const {
  struct tm Tm;
  IAssert(IsDef() && GetTmStruct(AbsSecs(), Tm));
  return TTmInfo::GetDayOfWeekNm(Tm.tm_wday+1, Loc);
}
int TSecTm::GetHourN() const {
  struct tm Tm;
  IAssert(IsDef() && GetTmStruct(AbsSecs(), Tm));
  return Tm.tm_hour;
}
int TSecTm::GetMinN() const {
  struct tm Tm;
  IAssert(IsDef() && GetTmStruct(AbsSecs(), Tm));
  return Tm.tm_min;
}
int TSecTm::GetSecN() const {
  struct tm Tm;
  IAssert(IsDef() && GetTmStruct(AbsSecs(), Tm));
  return Tm.tm_sec;
}
void TSecTm::GetComps(int& Year, int& Month, int& Day, int& Hour, int& Min, int& Sec) const {
  struct tm Tm;
  EAssert(IsDef() && GetTmStruct(AbsSecs(), Tm));
  Year = Tm.tm_year+1900;
  Month = Tm.tm_mon+1;
  Day = Tm.tm_mday;
  Hour = Tm.tm_hour;
  Min = Tm.tm_min;
  Sec = Tm.tm_sec;
}
TSecTm TSecTm::Round(const TTmUnit& TmUnit) const {
  if (TmUnit == tmu1Sec) { return *this; }
  struct tm Time;
  IAssert(IsDef() && GetTmStruct(AbsSecs(), Time));
  switch (TmUnit) {
    case tmu1Min : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, Time.tm_hour, Time.tm_min, 0);
    case tmu10Min : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, Time.tm_hour, 10*(Time.tm_min/10), 0);
    case tmu15Min : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, Time.tm_hour, 15*(Time.tm_min/15), 0);
    case tmu30Min : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, Time.tm_hour, 30*(Time.tm_min/30), 0);
    case tmu1Hour : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, Time.tm_hour, 0, 0);
    case tmu2Hour : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, 2*(Time.tm_hour/2), 0, 0);
    case tmu4Hour : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, 4*(Time.tm_hour/4), 0, 0);
    case tmu6Hour : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, 6*(Time.tm_hour/6), 0, 0);
    case tmu12Hour : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, 12*(Time.tm_hour/12), 0, 0);
    case tmuDay : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, 0, 0, 0);
    case tmuMonth : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, 1, 0, 0, 0);
    case tmuYear : return TSecTm(Time.tm_year+1900, 1, 1, 0, 0, 0);
    case tmuWeek : { int dd=1, mm=1, yy=1;

      const int Day = TJulianDate::GetJulianDateN(Time.tm_mday, Time.tm_mon+1, 1900+Time.tm_year);
      TJulianDate::GetCalendarDate(3+7*(Day/7), dd, mm, yy);  return TSecTm(yy, mm, dd, 0, 0, 0); }
    default : Fail;
  }
  return TSecTm();
}
uint TSecTm::GetInUnits(const TTmUnit& TmUnit) const {
  static const int DayZero = TJulianDate::GetJulianDateN(1, 1, 1970);
  if (TmUnit == tmu1Sec) { return AbsSecs; }
  struct tm Time;
  IAssert(IsDef() && GetTmStruct(AbsSecs(), Time));
  switch (TmUnit) {
    case tmu1Min : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, Time.tm_hour, Time.tm_min, 0).GetAbsSecs()/60;
    case tmu10Min : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, Time.tm_hour, 10*(Time.tm_min/10), 0).GetAbsSecs()/(10*60);
    case tmu15Min : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, Time.tm_hour, 15*(Time.tm_min/15), 0).GetAbsSecs()/(15*60);
    case tmu30Min : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, Time.tm_hour, 30*(Time.tm_min/30), 0).GetAbsSecs()/(30*60);
    case tmu1Hour : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, Time.tm_hour, 0, 0).GetAbsSecs()/3600;
    case tmu2Hour : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, 2*(Time.tm_hour/2), 0, 0).GetAbsSecs()/(2*3600);
    case tmu4Hour : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, 4*(Time.tm_hour/4), 0, 0).GetAbsSecs()/(4*3600);
    case tmu6Hour : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, 6*(Time.tm_hour/6), 0, 0).GetAbsSecs()/(6*3600);
    case tmu12Hour : return TSecTm(Time.tm_year+1900, Time.tm_mon+1, Time.tm_mday, 12*(Time.tm_hour/12), 0, 0).GetAbsSecs()/(12*3600);
    case tmuDay : return TJulianDate::GetJulianDateN(Time.tm_mday, Time.tm_mon+1, 1900+Time.tm_year) - DayZero;
    case tmuWeek : return (TJulianDate::GetJulianDateN(Time.tm_mday, Time.tm_mon+1, 1900+Time.tm_year)-DayZero)/7;
    case tmuMonth : return 12*(Time.tm_year-70)+Time.tm_mon+1;
    case tmuYear : return Time.tm_year+1900;
    default : Fail;
  }
  return TUInt::Mx;
}
TStr TSecTm::GetDayPart() const {
  const int Hour = GetHourN();
  if (0 <= Hour && Hour < 6) { return "Night"; }
  else if (6 <= Hour && Hour < 12) { return "Morning"; }
  else if (12 <= Hour && Hour < 18) { return "Afternoon"; }
  else if (18 <= Hour && Hour < 24) { return "Evening"; }
  return "";
}
uint TSecTm::GetDSecs(const TSecTm& SecTm1, const TSecTm& SecTm2){
  IAssert(SecTm1.IsDef()&&SecTm2.IsDef());
  const time_t Time1= time_t(SecTm1.AbsSecs());
  const time_t Time2= time_t(SecTm2.AbsSecs());
  return uint(difftime(Time2, Time1));
}
TSecTm TSecTm::GetZeroWeekTm(){
  TSecTm ZeroWeekTm=GetZeroTm();
  while (ZeroWeekTm.GetDayOfWeekN()!=TTmInfo::MonN){
    ZeroWeekTm.AddDays(1);}
  return ZeroWeekTm;
}
TSecTm TSecTm::GetCurTm(){
  const time_t TmSec = time(NULL);
  struct tm LocTm;
  uint AbsSec = TUInt::Mx;
  #if defined(GLib_MSN)
  localtime_s(&LocTm, &TmSec);
  #elif defined(GLib_BCB)
  LocTm = *localtime(&TmSec);
  #else
  LocTm = *localtime(&TmSec);
  #endif
  IAssert(TSecTm::GetTmSec(LocTm, AbsSec));
  return TSecTm(AbsSec);
}
TSecTm TSecTm::GetDtTmFromHmsStr(const TStr& HmsStr){
  int HmsStrLen=HmsStr.Len();

  TChA ChA; int ChN=0;
  while ((ChN<HmsStrLen)&&(HmsStr[ChN]!=':')){ChA+=HmsStr[ChN]; ChN++;}
  TStr HourStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<HmsStrLen)&&(HmsStr[ChN]!=':')){ChA+=HmsStr[ChN]; ChN++;}
  TStr MinStr=ChA;

  ChA.Clr(); ChN++;
  while (ChN<HmsStrLen){ChA+=HmsStr[ChN]; ChN++;}
  TStr SecStr=ChA;

  int HourN=HourStr.GetInt();
  int MinN=MinStr.GetInt();
  int SecN=SecStr.GetInt();

  TSecTm Tm=TSecTm::GetZeroTm();
  Tm.AddHours(HourN);
  Tm.AddMins(MinN);
  Tm.AddSecs(SecN);
  return Tm;
}
TSecTm TSecTm::GetDtTmFromMdyStr(const TStr& MdyStr){
  int MdyStrLen=MdyStr.Len();

  TChA ChA; int ChN=0;
  while ((ChN<MdyStrLen)&&(MdyStr[ChN]!='/')){
    ChA+=MdyStr[ChN]; ChN++;}
  TStr MonthStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<MdyStrLen)&&(MdyStr[ChN]!='/')){
    ChA+=MdyStr[ChN]; ChN++;}
  TStr DayStr=ChA;

  ChA.Clr(); ChN++;
  while (ChN<MdyStrLen){
    ChA+=MdyStr[ChN]; ChN++;}
  TStr YearStr=ChA;

  int MonthN=MonthStr.GetInt();
  int DayN=DayStr.GetInt();
  int YearN=YearStr.GetInt();
  if (YearN<1000){
    if (YearN<70){YearN+=2000;} else {YearN+=1900;}}

  return GetDtTm(YearN, MonthN, DayN);
}
TSecTm TSecTm::GetDtTmFromDmyStr(const TStr& DmyStr){
  int DmyStrLen=DmyStr.Len();

  TChA ChA; int ChN=0;
  while ((ChN<DmyStrLen)&&(DmyStr[ChN]!='/')&&(DmyStr[ChN]!='-')){
    ChA+=DmyStr[ChN]; ChN++;}
  TStr DayStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<DmyStrLen)&&(DmyStr[ChN]!='/')&&(DmyStr[ChN]!='-')){
    ChA+=DmyStr[ChN]; ChN++;}
  TStr MonthStr=ChA;

  ChA.Clr(); ChN++;
  while (ChN<DmyStrLen){
    ChA+=DmyStr[ChN]; ChN++;}
  TStr YearStr=ChA;

  int DayN=DayStr.GetInt(-1);
  int MonthN=MonthStr.GetInt(-1);
  int YearN=YearStr.GetInt(-1);
  if (MonthN == -1){
    MonthN = TTmInfo::GetMonthN(MonthStr.ToCap()); }
  if ((DayN==-1)||(MonthN==-1)||(YearN==-1)){
    return TSecTm();
  } else {
    if (YearN<1000){
      if (YearN<70){YearN+=2000;} else {YearN+=1900;}}

    return GetDtTm(YearN, MonthN, DayN);
  }
  return TSecTm();
}
TSecTm TSecTm::GetDtTmFromMdyHmsPmStr(const TStr& MdyHmsPmStr,
 const char& DateSepCh, const char& TimeSepCh){
  int MdyHmsPmStrLen=MdyHmsPmStr.Len();

  TChA ChA; int ChN=0;
  while ((ChN<MdyHmsPmStrLen)&&(MdyHmsPmStr[ChN]!=DateSepCh)){
    ChA+=MdyHmsPmStr[ChN]; ChN++;}
  TStr MonthStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<MdyHmsPmStrLen)&&(MdyHmsPmStr[ChN]!=DateSepCh)){
    ChA+=MdyHmsPmStr[ChN]; ChN++;}
  TStr DayStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<MdyHmsPmStrLen)&&(MdyHmsPmStr[ChN]!=' ')){
    ChA+=MdyHmsPmStr[ChN]; ChN++;}
  TStr YearStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<MdyHmsPmStrLen)&&(MdyHmsPmStr[ChN]!=TimeSepCh)){
    ChA+=MdyHmsPmStr[ChN]; ChN++;}
  TStr HourStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<MdyHmsPmStrLen)&&(MdyHmsPmStr[ChN]!=TimeSepCh)){
    ChA+=MdyHmsPmStr[ChN]; ChN++;}
  TStr MinStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<MdyHmsPmStrLen)&&(MdyHmsPmStr[ChN]!=' ')){
    ChA+=MdyHmsPmStr[ChN]; ChN++;}
  TStr SecStr=ChA;

  ChA.Clr(); ChN++;
  while (ChN<MdyHmsPmStrLen){
    ChA+=MdyHmsPmStr[ChN]; ChN++;}
  TStr AmPmStr=ChA;

  int MonthN=MonthStr.GetInt();
  int DayN=DayStr.GetInt();
  int YearN=YearStr.GetInt();
  int HourN; int MinN; int SecN;
  if (HourStr.IsInt()){
    HourN=HourStr.GetInt();
    MinN=MinStr.GetInt();
    SecN=SecStr.GetInt();
    if (AmPmStr=="AM"){} else if (AmPmStr=="PM"){HourN+=12;} else {Fail;}
  } else {
    HourN=0; MinN=0; SecN=0;
  }

  TSecTm Tm=TSecTm::GetDtTm(YearN, MonthN, DayN);
  Tm.AddHours(HourN);
  Tm.AddMins(MinN);
  Tm.AddSecs(SecN);
  return Tm;
}
TSecTm TSecTm::GetDtTmFromYmdHmsStr(const TStr& YmdHmsPmStr,
 const char& DateSepCh, const char& TimeSepCh){
  int YmdHmsPmStrLen=YmdHmsPmStr.Len();

  TChA ChA; int ChN=0;
  while ((ChN<YmdHmsPmStrLen)&&(YmdHmsPmStr[ChN]!=DateSepCh)){
    ChA+=YmdHmsPmStr[ChN]; ChN++;}
  TStr YearStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<YmdHmsPmStrLen)&&(YmdHmsPmStr[ChN]!=DateSepCh)){
    ChA+=YmdHmsPmStr[ChN]; ChN++;}
  TStr MonthStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<YmdHmsPmStrLen)&&(YmdHmsPmStr[ChN]!=' ')){
    ChA+=YmdHmsPmStr[ChN]; ChN++;}
  TStr DayStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<YmdHmsPmStrLen)&&(YmdHmsPmStr[ChN]!=TimeSepCh)){
    ChA+=YmdHmsPmStr[ChN]; ChN++;}
  TStr HourStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<YmdHmsPmStrLen)&&(YmdHmsPmStr[ChN]!=TimeSepCh)){
    ChA+=YmdHmsPmStr[ChN]; ChN++;}
  TStr MinStr=ChA;

  ChA.Clr(); ChN++;
  while (ChN<YmdHmsPmStrLen){
    ChA+=YmdHmsPmStr[ChN]; ChN++;}
  TStr SecStr=ChA;

  int MonthN=MonthStr.GetInt();
  int DayN=DayStr.GetInt();
  int YearN=YearStr.GetInt();
  int HourN; int MinN; int SecN;
  if (HourStr.IsInt()){
    HourN=HourStr.GetInt();
    MinN=MinStr.GetInt();
    SecN=SecStr.GetInt();
  } else {
    HourN=0; MinN=0; SecN=0;
  }

  TSecTm Tm=TSecTm::GetDtTm(YearN, MonthN, DayN);
  Tm.AddHours(HourN);
  Tm.AddMins(MinN);
  Tm.AddSecs(SecN);
  return Tm;
}
TSecTm TSecTm::GetDtTmFromStr(const TChA& YmdHmsPmStr, const int& YearId, const int& MonId,
 const int& DayId, const int& HourId, const int& MinId, const int& SecId) {
  TChA Tmp = YmdHmsPmStr;
  TVec<char *> FldV;

  for (char *c = (char *) Tmp.CStr(); *c; c++) {
    if (TCh::IsNum(*c)) {
      FldV.Add(c);
      while (TCh::IsNum(*c)) { c++; }
      c--;
    } else { *c = 0; }
  }
  const int Y = atoi(FldV[YearId]);
  const int M = atoi(FldV[MonId]);
  const int D = atoi(FldV[DayId]);
  const int H = atoi(FldV[HourId]);
  const int m = atoi(FldV[MinId]);
  const int S = atoi(FldV[SecId]);
  IAssert(Y>0 && M>0 && D>0 && M<13 && D<32);
  IAssert(H>=0 && H<24 && m>=0 && m<60 && S>=0 && S<60);
  return TSecTm(Y,M,D,H,m,S);
}
TSecTm TSecTm::GetDtTm(const int& YearN, const int& MonthN, const int& DayN){
  uint AbsSecs;
  TSecTm::GetTmSec(YearN, MonthN, DayN, 0, 0, 0, AbsSecs);
  return TSecTm(AbsSecs);
}
TSecTm TSecTm::GetDtTm(const TSecTm& Tm){
  int DaySecs=Tm.GetHourN()*3600+Tm.GetMinN()*60+Tm.GetSecN();
  TSecTm DtTm(Tm.AbsSecs-DaySecs);
  return DtTm;
}
TSecTm TSecTm::LoadTxt(TILx& Lx){
  return TSecTm(Lx.GetInt());
}
void TSecTm::SaveTxt(TOLx& Lx) const {
  IAssert(int(AbsSecs) < TInt::Mx);
  Lx.PutInt((int)AbsSecs);
}
TStr TTm::GetStr(const bool& MSecP) const {
  TChA ChA;
  ChA+=TInt::GetStr(Year, "%04d"); ChA+='-';
  ChA+=TInt::GetStr(Month, "%02d"); ChA+='-';
  ChA+=TInt::GetStr(Day, "%02d"); ChA+=' ';
  ChA+=TInt::GetStr(Hour, "%02d"); ChA+=':';
  ChA+=TInt::GetStr(Min, "%02d"); ChA+=':';
  ChA+=TInt::GetStr(Sec, "%02d");
  if (MSecP){ChA+='.'; ChA+=TInt::GetStr(MSec, "%03d");}
  return ChA;
}
TStr TTm::GetYMDDashStr() const {
  TChA ChA;
  ChA+=TInt::GetStr(Year, "%04d");
  ChA+='-'; ChA+=TInt::GetStr(Month, "%02d");
  ChA+='-'; ChA+=TInt::GetStr(Day, "%02d");
  return ChA;
}
TStr TTm::GetHMSTColonDotStr(const bool& FullP, const bool& MSecP) const {
  TChA ChA;
  ChA+=TInt::GetStr(Hour, "%02d");
  ChA+=':'; ChA+=TInt::GetStr(Min, "%02d");
  if (FullP||((Sec!=0)||(MSec!=0))){
    ChA+=':'; ChA+=TInt::GetStr(Sec, "%02d");
    if ((MSecP)&&(FullP||(MSec!=0))){
      ChA+='.'; ChA+=TInt::GetStr(MSec, "%d");
    }
  }
  return ChA;
}
TStr TTm::GetIdStr() const {
  TChA ChA;
  ChA+=TInt::GetStr(Year%100, "%02d");
  ChA+=TInt::GetStr(Month, "%02d");
  ChA+=TInt::GetStr(Day, "%02d");
  ChA+=TInt::GetStr(Hour, "%02d");
  ChA+=TInt::GetStr(Min, "%02d");
  ChA+=TInt::GetStr(Sec, "%02d");
  ChA+=TInt::GetStr(MSec, "%03d");
  return ChA;
}
void TTm::AddTime(const int& Hours,
 const int& Mins, const int& Secs, const int& MSecs){
  uint64 TmMSecs=TTm::GetMSecsFromTm(*this);
  TmMSecs+=(uint64(Hours)*uint64(3600)*uint64(1000));
  TmMSecs+=(uint64(Mins)*uint64(60)*uint64(1000));
  TmMSecs+=(uint64(Secs)*uint64(1000));
  TmMSecs+=uint64(MSecs);
  *this=GetTmFromMSecs(TmMSecs);
}
void TTm::SubTime(const int& Hours,
 const int& Mins, const int& Secs, const int& MSecs){
  uint64 TmMSecs=TTm::GetMSecsFromTm(*this);
  TmMSecs-=(uint64(Hours)*uint64(3600)*uint64(1000));
  TmMSecs-=(uint64(Mins)*uint64(60)*uint64(1000));
  TmMSecs-=(uint64(Secs)*uint64(1000));
  TmMSecs-=(uint64(MSecs));
  *this=GetTmFromMSecs(TmMSecs);
}
TTm TTm::GetCurUniTm(){
  return TSysTm::GetCurUniTm();
}
TTm TTm::GetUniqueCurUniTm(){
  static TTm LastUniqueTm=TSysTm::GetCurUniTm();
  TTm CurUniqueTm=TSysTm::GetCurUniTm();
  if (CurUniqueTm<LastUniqueTm){CurUniqueTm=LastUniqueTm;}
  if (CurUniqueTm==LastUniqueTm){CurUniqueTm.AddTime(0, 0, 0, 1);}
  LastUniqueTm=CurUniqueTm;
  return CurUniqueTm;
}
TTm TTm::GetUniqueCurUniTm(const int& UniqueSpaces, const int& UniqueSpaceN){
  static uint64 LastMUniqueTmMSecs=TSysTm::GetCurUniMSecs();

  Assert(UniqueSpaces>=1&&UniqueSpaceN>=0&&UniqueSpaceN<UniqueSpaces);

  uint64 CurUniqueTmMSecs=TSysTm::GetCurUniMSecs();
  if (CurUniqueTmMSecs<LastMUniqueTmMSecs){CurUniqueTmMSecs=LastMUniqueTmMSecs;}

  CurUniqueTmMSecs-=CurUniqueTmMSecs%UniqueSpaces; CurUniqueTmMSecs+=UniqueSpaceN;

  if (CurUniqueTmMSecs<=LastMUniqueTmMSecs){
    CurUniqueTmMSecs+=UniqueSpaces;
  }

  LastMUniqueTmMSecs=CurUniqueTmMSecs;
  return GetTmFromMSecs(CurUniqueTmMSecs);
}
TTm TTm::GetCurLocTm(){
  return TSysTm::GetCurLocTm();
}
uint64 TTm::GetCurUniMSecs(){
  return TSysTm::GetCurUniMSecs();
}
uint64 TTm::GetCurLocMSecs(){
  return TSysTm::GetCurLocMSecs();
}
uint64 TTm::GetMSecsFromTm(const TTm& Tm){
  return TSysTm::GetMSecsFromTm(Tm);
}
TTm TTm::GetTmFromMSecs(const uint64& MSecs){
  return TSysTm::GetTmFromMSecs(MSecs);
}
uint TTm::GetMSecsFromOsStart(){
  return TSysTm::GetMSecsFromOsStart();
}
uint64 TTm::GetPerfTimerFq(){
  return TSysTm::GetPerfTimerFq();
}
uint64 TTm::GetPerfTimerTicks(){
  return TSysTm::GetPerfTimerTicks();
}
void TTm::GetDiff(const TTm& Tm1, const TTm& Tm2, int& Days,
	  int& Hours, int& Mins, int& Secs, int& MSecs) {
	const uint64 DiffMSecs = TTm::GetDiffMSecs(Tm1, Tm2);
	const uint64 DiffSecs = DiffMSecs / 1000;
	const uint64 DiffMins = DiffSecs / 60;
	const uint64 DiffHours = DiffMins / 60;	
	MSecs = int(DiffMSecs % 1000);
	Secs = int(DiffSecs % 60);
	Mins = int(DiffMins % 60);
	Hours = int(DiffHours % 24);
	Days = int((int)DiffHours / 24);
}
uint64 TTm::GetDiffMSecs(const TTm& Tm1, const TTm& Tm2){
  uint64 Tm1MSecs=GetMSecsFromTm(Tm1);
  uint64 Tm2MSecs=GetMSecsFromTm(Tm2);
  if (Tm1MSecs>Tm2MSecs){
    return Tm1MSecs-Tm2MSecs;
  } else {
    return Tm2MSecs-Tm1MSecs;
  }
}
TTm TTm::GetLocTmFromUniTm(const TTm& Tm){
  return TSysTm::GetLocTmFromUniTm(Tm);
}
TTm TTm::GetUniTmFromLocTm(const TTm& Tm){
  return TSysTm::GetUniTmFromLocTm(Tm);
}
TTm TTm::GetTmFromWebLogTimeStr(const TStr& TimeStr,
 const char TimeSepCh, const char MSecSepCh){
  int TimeStrLen=TimeStr.Len();

  TChA ChA; int ChN=0;
  while ((ChN<TimeStrLen)&&(TimeStr[ChN]!=TimeSepCh)){
    ChA+=TimeStr[ChN]; ChN++;}
  TStr HourStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<TimeStrLen)&&(TimeStr[ChN]!=TimeSepCh)){
    ChA+=TimeStr[ChN]; ChN++;}
  TStr MinStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<TimeStrLen)&&(TimeStr[ChN]!=MSecSepCh)){
    ChA+=TimeStr[ChN]; ChN++;}
  TStr SecStr=ChA;

  ChA.Clr(); ChN++;
  while (ChN<TimeStrLen){
    ChA+=TimeStr[ChN]; ChN++;}
  TStr MSecStr=ChA;

  int HourN=HourStr.GetInt(0);
  int MinN=MinStr.GetInt(0);
  int SecN=SecStr.GetInt(0);
  int MSecN=MSecStr.GetInt(0);

  TTm Tm(-1, -1, -1, -1, HourN, MinN, SecN, MSecN);

  return Tm;
}
TTm TTm::GetTmFromWebLogDateTimeStr(const TStr& DateTimeStr,
 const char DateSepCh, const char TimeSepCh, const char MSecSepCh,
 const char DateTimeSepCh){
  int DateTimeStrLen=DateTimeStr.Len();

  TChA ChA; int ChN=0;
  while ((ChN<DateTimeStrLen)&&(DateTimeStr[ChN]!=DateSepCh)){
    ChA+=DateTimeStr[ChN]; ChN++;}
  TStr YearStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<DateTimeStrLen)&&(DateTimeStr[ChN]!=DateSepCh)){
    ChA+=DateTimeStr[ChN]; ChN++;}
  TStr MonthStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<DateTimeStrLen)&&(DateTimeStr[ChN]!=DateTimeSepCh)){
    ChA+=DateTimeStr[ChN]; ChN++;}
  TStr DayStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<DateTimeStrLen)&&(DateTimeStr[ChN]!=TimeSepCh)){
    ChA+=DateTimeStr[ChN]; ChN++;}
  TStr HourStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<DateTimeStrLen)&&(DateTimeStr[ChN]!=TimeSepCh)){
    ChA+=DateTimeStr[ChN]; ChN++;}
  TStr MinStr=ChA;

  ChA.Clr(); ChN++;
  while ((ChN<DateTimeStrLen)&&(DateTimeStr[ChN]!=MSecSepCh)){
    ChA+=DateTimeStr[ChN]; ChN++;}
  TStr SecStr=ChA;

  ChA.Clr(); ChN++;
  while (ChN<DateTimeStrLen){
    ChA+=DateTimeStr[ChN]; ChN++;}
  TStr MSecStr=ChA;

  int YearN=YearStr.GetInt(-1);
  int MonthN=MonthStr.GetInt(-1);
  int DayN=DayStr.GetInt(-1);
  int HourN=HourStr.GetInt(0);
  int MinN=MinStr.GetInt(0);
  int SecN=SecStr.GetInt(0);
  int MSecN=MSecStr.GetInt(0);

  TTm Tm;
  if ((YearN!=-1)&&(MonthN!=-1)&&(DayN!=-1)){
    Tm=TTm(YearN, MonthN, DayN, -1, HourN, MinN, SecN, MSecN);
  }

  return Tm;
}
TTm TTm::GetTmFromIdStr(const TStr& IdStr){

  TChA IdChA=IdStr;
  if (IdChA.Len()==14){
    IdChA.Ins(0, "0");}

  IAssert(IdChA.Len()==15);
  for (int ChN=0; ChN<IdChA.Len(); ChN++){
    IAssert(TCh::IsNum(IdChA[ChN]));}

  int YearN=2000+(TStr(IdChA[0])+TStr(IdChA[1])).GetInt();
  int MonthN=(TStr(IdChA[2])+TStr(IdChA[3])).GetInt();
  int DayN=(TStr(IdChA[4])+TStr(IdChA[5])).GetInt();
  int HourN=(TStr(IdChA[6])+TStr(IdChA[7])).GetInt();
  int MinN=(TStr(IdChA[8])+TStr(IdChA[9])).GetInt();
  int SecN=(TStr(IdChA[10])+TStr(IdChA[11])).GetInt();
  int MSecN=(TStr(IdChA[12])+TStr(IdChA[13])+TStr(IdChA[14])).GetInt();
  TTm Tm=TTm(YearN, MonthN, DayN, -1, HourN, MinN, SecN, MSecN);
  return Tm;
}
uint TTm::GetDateTimeInt(const int& Year, const int& Month,
      const int& Day, const int& Hour, const int& Min, const int& Sec) {
	return TSecTm(Year, Month, Day, Hour, Min, Sec).GetAbsSecs();
}
uint TTm::GetDateIntFromTm(const TTm& Tm) {
    return Tm.IsDef() ? GetDateTimeInt(Tm.GetYear(), Tm.GetMonth(), Tm.GetDay()) : 0;
}
uint TTm::GetMonthIntFromTm(const TTm& Tm) {
    return Tm.IsDef() ? GetDateTimeInt(Tm.GetYear(), Tm.GetMonth()) : 0;
}
uint TTm::GetYearIntFromTm(const TTm& Tm) {
    return Tm.IsDef() ? GetDateTimeInt(Tm.GetYear()) : 0;
}
uint TTm::GetDateTimeIntFromTm(const TTm& Tm) {
    return Tm.IsDef() ? 
		GetDateTimeInt(Tm.GetYear(), Tm.GetMonth(),
        Tm.GetDay(), Tm.GetHour(), Tm.GetMin(), Tm.GetSec()) : 0;
}
TTm TTm::GetTmFromDateTimeInt(const uint& DateTimeInt) {
	if (DateTimeInt == 0) { return TTm(); }
	return TTm(TSecTm(DateTimeInt));
}
TSecTm TTm::GetSecTmFromDateTimeInt(const uint& DateTimeInt) {
	if (DateTimeInt == 0) { return TSecTm(); }
	return TSecTm(DateTimeInt);
}
int TTmProfiler::AddTimer(const TStr& TimerNm) {
	MxNmLen = TInt::GetMx(MxNmLen, TimerNm.Len());
	return TimerH.AddKey(TimerNm); 
}
void TTmProfiler::ResetAll() {
    int TimerId = GetTimerIdFFirst();
	while (GetTimerIdFNext(TimerId)) {
		ResetTimer(TimerId);
	}
}
double TTmProfiler::GetTimerSumSec() const {
	double Sum = 0.0;
    int TimerId = GetTimerIdFFirst();
	while (GetTimerIdFNext(TimerId)) {
		Sum += GetTimerSec(TimerId);
	}
    return Sum;
}
double TTmProfiler::GetTimerSec(const int& TimerId) const {
    return TimerH[TimerId].GetSec();
}
void TTmProfiler::PrintReport(const TStr& ProfileNm) const {
    const double TimerSumSec = GetTimerSumSec();
	printf("-- %s --\n", ProfileNm.CStr());
    printf("Sum: (%.2f sec):\n", TimerSumSec);
    int TimerId = GetTimerIdFFirst();
	while (GetTimerIdFNext(TimerId)) {

        TStr TimerNm = GetTimerNm(TimerId);
        TimerNm = TStr::GetSpaceStr(TimerNm.Len() - MxNmLen) + TimerNm;

        if (TimerSumSec > 0.0) {
            const double TimerSec = GetTimerSec(TimerId);
            const double TimerPerc =  TimerSec / TimerSumSec * 100.0;
            printf(" %s: %.2fs [%.2f%%]\n", TimerNm.CStr(), TimerSec, TimerPerc);
        } else {
            printf(" %s: -\n", TimerNm.CStr());
        }
    }
	printf("--\n");
}
