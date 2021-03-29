#include "bd.h"
#include <new>
typedef int TUniVecIdx;
typedef enum TUnicodeErrorHandling_
{
	uehIgnore = 0,
	uehThrow = 1,
	uehReplace = 2,
	uehAbort = 3
}
TUnicodeErrorHandling;
class TUnicodeException
{
public:
	TStr message;
	size_t srcIdx;
	int srcChar;
	TUnicodeException(size_t srcIdx_, int srcChar_, const TStr& message_) :
		message(message_), srcIdx(srcIdx_), srcChar(srcChar_) { }
};
typedef enum TUniByteOrder_
{
	boMachineEndian = 0,
	boLittleEndian = 1,
	boBigEndian = 2
}
TUniByteOrder;
typedef enum TUtf16BomHandling_
{
	bomAllowed = 0,
	bomRequired = 1,
	bomIgnored = 2
}
TUtf16BomHandling;
class TUniCodec
{
public:
	enum { DefaultReplacementChar = 0xfffd };
	int replacementChar;
	TUnicodeErrorHandling errorHandling;
	bool strict;
	bool skipBom;
	TUniCodec() : replacementChar(DefaultReplacementChar), errorHandling(uehIgnore), strict(false), skipBom(true)
	{
	}
	TUniCodec(TUnicodeErrorHandling errorHandling_, bool strict_, int replacementChar_, bool skipBom_) :
		replacementChar(replacementChar_), errorHandling(errorHandling_), strict(strict_), skipBom(skipBom_)
	{
	}
protected:
	enum {
#define DefineByte(b7, b6, b5, b4, b3, b2, b1, b0) _ ## b7 ## b6 ## b5 ## b4 ## _ ## b3 ## b2 ## b1 ## b0 = (b7 << 7) | (b6 << 6) | (b5 << 5) | (b4 << 4) | (b3 << 3) | (b2 << 2) | (b1 << 1) | b0
		DefineByte(1, 0, 0, 0, 0, 0, 0, 0),
		DefineByte(1, 1, 0, 0, 0, 0, 0, 0),
		DefineByte(1, 1, 1, 0, 0, 0, 0, 0),
		DefineByte(1, 1, 1, 1, 0, 0, 0, 0),
		DefineByte(1, 1, 1, 1, 1, 0, 0, 0),
		DefineByte(1, 1, 1, 1, 1, 1, 0, 0),
		DefineByte(1, 1, 1, 1, 1, 1, 1, 0),
		DefineByte(0, 0, 1, 1, 1, 1, 1, 1),
		DefineByte(0, 0, 0, 1, 1, 1, 1, 1),
		DefineByte(0, 0, 0, 0, 1, 1, 1, 1),
		DefineByte(0, 0, 0, 0, 0, 1, 1, 1),
		DefineByte(0, 0, 0, 0, 0, 0, 1, 1)
#undef DefineByte
	};
	typedef TUniVecIdx TVecIdx;
	friend class TUniCaseFolding;
	friend class TUnicode;
public:
	template<typename TSrcVec, typename TDestCh>
	size_t DecodeUtf8(
		const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, const bool clrDest = true) const;
	template<typename TSrcVec, typename TDestCh>
	size_t DecodeUtf8(const TSrcVec& src, TVec<TDestCh>& dest, const bool clrDest = true) const { return DecodeUtf8(src, 0, src.Len(), dest, clrDest); }
	template<typename TSrcVec, typename TDestCh>
	size_t EncodeUtf8(
		const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, const bool clrDest = true) const;
	template<typename TSrcVec, typename TDestCh>
	size_t EncodeUtf8(const TSrcVec& src, TVec<TDestCh>& dest, const bool clrDest = true) const { return EncodeUtf8(src, 0, src.Len(), dest, clrDest); }
	template<typename TSrcVec> TStr EncodeUtf8Str(const TSrcVec& src, size_t srcIdx, const size_t srcCount) const { TVec<char> temp; EncodeUtf8(src, srcIdx, srcCount, temp); TStr retVal = &(temp[0]); return retVal; }
	template<typename TSrcVec> TStr EncodeUtf8Str(const TSrcVec& src) const { TVec<char> temp; EncodeUtf8(src, temp); temp.Add(0); TStr retVal = &(temp[0]); return retVal; }
protected:
	enum {
		Utf16FirstSurrogate = 0xd800,
		Utf16SecondSurrogate = 0xdc00
	};
	static bool IsMachineLittleEndian();
public:
	template<typename TSrcVec, typename TDestCh>
	size_t DecodeUtf16FromBytes(
		const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, const bool clrDest,
		const TUtf16BomHandling bomHandling = bomAllowed,
		const TUniByteOrder defaultByteOrder = boMachineEndian) const;
	template<typename TSrcVec, typename TDestCh>
	size_t DecodeUtf16FromWords(
		const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, bool clrDest,
		const TUtf16BomHandling bomHandling = bomAllowed,
		const TUniByteOrder defaultByteOrder = boMachineEndian) const;
	template<typename TSrcVec, typename TDestCh>
	size_t EncodeUtf16ToWords(
		const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, const bool clrDest, const bool insertBom,
		const TUniByteOrder destByteOrder = boMachineEndian) const;
	template<typename TSrcVec, typename TDestCh>
	size_t EncodeUtf16ToBytes(
		const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, const bool clrDest, const bool insertBom,
		const TUniByteOrder destByteOrder = boMachineEndian) const;
protected:
	static uint GetRndUint(TRnd& rnd);
	static uint GetRndUint(TRnd& rnd, uint minVal, uint maxVal);
protected:
	void TestUtf8(bool decode, size_t expectedRetVal, bool expectedThrow, const TIntV& src, const TIntV& expectedDest, FILE *f);
	void TestDecodeUtf8(TRnd& rnd, const TStr& testCaseDesc);
public:
	void TestUtf8();
protected:
	void WordsToBytes(const TIntV& src, TIntV& dest);
	void TestUtf16(bool decode, size_t expectedRetVal, bool expectedThrow, const TIntV& src, const TIntV& expectedDest,
		const TUtf16BomHandling bomHandling, const TUniByteOrder defaultByteOrder, const bool insertBom,
		FILE *f);
	static inline int SwapBytes(int x) {
		return ((x >> 8) & 0xff) | ((x & 0xff) << 8); }
	void TestDecodeUtf16(TRnd& rnd, const TStr& testCaseDesc,
		const TUtf16BomHandling bomHandling,
		const TUniByteOrder defaultByteOrder,
		const bool insertBom);
public:
	void TestUtf16();
};
typedef THash<TInt, TIntV> TIntIntVH;
class TUniCaseFolding
{
protected:
	TIntH cfCommon, cfSimple, cfTurkic;
	TIntIntVH cfFull;
	template<typename TSrcDat, typename TDestDat>
	inline static void AppendVector(const TVec<TSrcDat>& src, TVec<TDestDat>& dest) {
		for (int i = 0; i < src.Len(); i++) dest.Add(src[i]); }
	friend class TUniChDb;
	typedef TUniVecIdx TVecIdx;
public:
	TUniCaseFolding() { }
	explicit TUniCaseFolding(TSIn& SIn) : cfCommon(SIn), cfSimple(SIn), cfTurkic(SIn), cfFull(SIn) { SIn.LoadCs(); }
	void Load(TSIn& SIn) { cfCommon.Load(SIn); cfSimple.Load(SIn); cfFull.Load(SIn); cfTurkic.Load(SIn); SIn.LoadCs(); }
	void Save(TSOut& SOut) const { cfCommon.Save(SOut); cfSimple.Save(SOut); cfFull.Save(SOut); cfTurkic.Save(SOut); SOut.SaveCs(); }
	void Clr() { cfCommon.Clr(); cfSimple.Clr(); cfFull.Clr(); cfTurkic.Clr(); }
	void LoadTxt(const TStr& fileName);
	template<typename TSrcVec, typename TDestCh>
	void Fold(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, const bool clrDest, const bool full, const bool turkic) const
	{
		for (const size_t srcEnd = srcIdx + srcCount; srcIdx < srcEnd; )
		{
			int c = src[TVecIdx(srcIdx)], i; srcIdx++;
			if (turkic && ((i = cfTurkic.GetKeyId(c)) >= 0)) { dest.Add(cfTurkic[i]); continue; }
			if (full && ((i = cfFull.GetKeyId(c)) >= 0)) { AppendVector(cfFull[i], dest); continue; }
			if ((! full) && ((i = cfSimple.GetKeyId(c)) >= 0)) { dest.Add(cfSimple[i]); continue; }
			i = cfCommon.GetKeyId(c); if (i >= 0) dest.Add(cfCommon[i]); else dest.Add(c);
		}
	}
	template<typename TSrcVec>
	void FoldInPlace(TSrcVec& src, size_t srcIdx, const size_t srcCount, const bool turkic) const
	{
		for (const size_t srcEnd = srcIdx + srcCount; srcIdx < srcEnd; srcIdx++)
		{
			int c = src[TVecIdx(srcIdx)], i;
			if (turkic && ((i = cfTurkic.GetKeyId(c)) >= 0)) { src[TVecIdx(srcIdx)] = cfTurkic[i]; continue; }
			if ((i = cfSimple.GetKeyId(c)) >= 0) { src[TVecIdx(srcIdx)] = cfSimple[i]; continue; }
			i = cfCommon.GetKeyId(c); if (i >= 0) src[TVecIdx(srcIdx)] = cfCommon[i];
		}
	}
protected:
	void Test(const TIntV& src, const TIntV& expectedDest, const bool full, const bool turkic, FILE *f);
public:
	void Test();
};
class TCodecBase;
typedef TPt<TCodecBase> PCodecBase;
typedef TVec<PCodecBase> TCodecBaseV;
class TCodecBase
{
protected:
	TCRef CRef;
	friend class TPt<TCodecBase>;
public:
	virtual ~TCodecBase() { }
	template<class TCodecImpl>
	static PCodecBase New(); /* {
		return new TCodecWrapper<TCodecImpl>(); } */
	virtual TStr GetName() const = 0;
	virtual void Test() const { }
	virtual size_t ToUnicode(const TIntV& src, size_t srcIdx, const size_t srcCount, TIntV& dest, const bool clrDest = true) const = 0;
	virtual size_t ToUnicode(const TStr& src, size_t srcIdx, const size_t srcCount, TIntV& dest, const bool clrDest = true) const = 0;
	size_t ToUnicode(const TIntV& src, TIntV& dest, const bool clrDest = true) const { return ToUnicode(src, 0, src.Len(), dest, clrDest); }
	size_t ToUnicode(const TStr& src, TIntV& dest, const bool clrDest = true) const { return ToUnicode(src, 0, src.Len(), dest, clrDest); }
	virtual size_t FromUnicode(const TIntV& src, size_t srcIdx, const size_t srcCount, TIntV& dest, const bool clrDest = true) const = 0;
	virtual size_t FromUnicode(const TIntV& src, size_t srcIdx, const size_t srcCount, TChA& dest, const bool clrDest = true) const = 0;
	virtual size_t FromUnicode(const TIntV& src, size_t srcIdx, const size_t srcCount, TStr& dest, const bool clrDest = true) const = 0;
	size_t FromUnicode(const TIntV& src, TIntV& dest, const bool clrDest = true) const { return FromUnicode(src, 0, src.Len(), dest, clrDest); }
	size_t FromUnicode(const TIntV& src, TChA& dest, const bool clrDest = true) const { return FromUnicode(src, 0, src.Len(), dest, clrDest); }
	size_t FromUnicode(const TIntV& src, TStr& dest, const bool clrDest = true) const { return FromUnicode(src, 0, src.Len(), dest, clrDest); }
};
template<class TCodecImpl_>
class TCodecWrapper : public TCodecBase
{
public:
	typedef TCodecImpl_ TCodecImpl;
	TCodecImpl impl;
public:
	virtual TStr GetName() const { return impl.GetName(); }
	virtual void Test() const { impl.Test(); }
	virtual size_t ToUnicode(const TIntV& src, size_t srcIdx, const size_t srcCount, TIntV& dest, const bool clrDest = true) const {
		return impl.ToUnicode(src, srcIdx, srcCount, dest, clrDest); }
	virtual size_t ToUnicode(const TStr& src, size_t srcIdx, const size_t srcCount, TIntV& dest, const bool clrDest = true) const {
		return impl.ToUnicode(src, srcIdx, srcCount, dest, clrDest); }
	virtual size_t FromUnicode(const TIntV& src, size_t srcIdx, const size_t srcCount, TIntV& dest, const bool clrDest = true) const {
		return impl.FromUnicode(src, srcIdx, srcCount, dest, clrDest); }
	virtual size_t FromUnicode(const TIntV& src, size_t srcIdx, const size_t srcCount, TChA& dest, const bool clrDest = true) const {
		return impl.FromUnicode(src, srcIdx, srcCount, dest, clrDest); }
	virtual size_t FromUnicode(const TIntV& src, size_t srcIdx, const size_t srcCount, TStr& dest, const bool clrDest = true) const {
		TChA buf; size_t retVal = impl.FromUnicode(src, srcIdx, srcCount, buf, false);
		if (clrDest) dest += buf.CStr(); else dest = buf.CStr();
		return retVal; }
};
template<class TCodecImpl>
PCodecBase TCodecBase::New() {
  return new TCodecWrapper<TCodecImpl>();
}
template<class TVector_>
class TVecElt
{
};
template<class TDat>
class TVecElt<TVec<TDat> >
{
public:
	typedef TVec<TDat> TVector;
	typedef TDat TElement;
	static inline void Add(TVector& vector, const TElement& element) { vector.Add(element); }
};
template<>
class TVecElt<TChA>
{
public:
	typedef TChA TVector;
	typedef char TElement;
	static inline void Add(TVector& vector, const TElement& element) { vector += element; }
};
class TEncoding_ISO8859_1
{
public:
	static inline TStr GetName() { return "ISO-8859-1"; }
	static int ToUnicode(int c) { Assert(0 <= c && c <= 255); return c; }
	static int FromUnicode(int c) { if (0 <= c && c <= 255) return c; else return -1; }
};
class TEncoding_ISO8859_2
{
public:
	static inline TStr GetName() { return "ISO-8859-2"; }
	static const int toUnicodeTable[6 * 16], fromUnicodeTable1[14 * 16], fromUnicodeTable2[2 * 16];
	static int ToUnicode(int c) { Assert(0 <= c && c <= 255);
		if (c < 0xa0) return c; else return toUnicodeTable[c - 0xa0]; }
	static int FromUnicode(int c) {
		if (0 <= c && c < 0xa0) return c;
		else if (0xa0 <= c && c < 0x180) return fromUnicodeTable1[c - 0xa0];
		else if (0x2c0 <= c && c < 0x2e0) return fromUnicodeTable2[c - 0x2c0];
		else return -1; }
};
class TEncoding_ISO8859_3
{
public:
	static inline TStr GetName() { return "ISO-8859-3"; }
	static const int toUnicodeTable[6 * 16], fromUnicodeTable1[14 * 16], fromUnicodeTable2[2];
	static int ToUnicode(int c) { Assert(0 <= c && c <= 255);
		if (c < 0xa0) return c; else return toUnicodeTable[c - 0xa0]; }
	static int FromUnicode(int c) {
		if (0 <= c && c < 0xa0) return c;
		else if (0xa0 <= c && c < 0x180) return fromUnicodeTable1[c - 0xa0];
		else if (0x2d8 <= c && c < 0x2da) return fromUnicodeTable2[c - 0x2d8];
		else return -1; }
};
class TEncoding_ISO8859_4
{
public:
	static inline TStr GetName() { return "ISO-8859-4"; }
	static const int toUnicodeTable[6 * 16], fromUnicodeTable1[14 * 16], fromUnicodeTable2[2 * 16];
	static int ToUnicode(int c) { Assert(0 <= c && c <= 255);
		if (c < 0xa0) return c; else return toUnicodeTable[c - 0xa0]; }
	static int FromUnicode(int c) {
		if (0 <= c && c < 0xa0) return c;
		else if (0xa0 <= c && c < 0x180) return fromUnicodeTable1[c - 0xa0];
		else if (0x2c0 <= c && c < 0x2e0) return fromUnicodeTable2[c - 0x2c0];
		else return -1; }
};
class TEncoding_YuAscii
{
public:
	static const int uniChars[10], yuAsciiChars[10];
	static inline TStr GetName() { return "YU-ASCII"; }
	static int ToUnicode(int c) { Assert(0 <= c && c <= 255);
		for (int i = 0; i < int(sizeof(yuAsciiChars) / sizeof(yuAsciiChars[0])); i++)
			if (c == yuAsciiChars[i]) return uniChars[i];
		return c; }
	static int FromUnicode(int c) {
		for (int i = 0; i < int(sizeof(uniChars) / sizeof(uniChars[0])); i++)
			if (c == uniChars[i]) return yuAsciiChars[i];
			else if(c == yuAsciiChars[i]) return -1;
		if (0 <= c && c <= 255) return c; else return -1; }
};
class TEncoding_CP437
{
public:
	static inline TStr GetName() { return "CP437"; }
	static const int toUnicodeTable[8 * 16], fromUnicodeTable1[6 * 16], fromUnicodeTable2[4 * 16], fromUnicodeTable3[6 * 16], fromUnicodeTable4[11 * 16];
	static int ToUnicode(int c) { Assert(0 <= c && c <= 255);
		if (c < 0x80) return c; else return toUnicodeTable[c - 0x80]; }
	static int FromUnicode(int c) {
		if (0 <= c && c < 0x80) return c;
		else if (0xa0 <= c && c < 0x100) return fromUnicodeTable1[c - 0xa0];
		else if (0x390 <= c && c < 0x3d0) return fromUnicodeTable2[c - 0x390];
		else if (0x2210 <= c && c < 0x2270) return fromUnicodeTable3[c - 0x2210];
		else if (0x2500 <= c && c < 0x25b0) return fromUnicodeTable4[c - 0x2500];
		else if (c == 0x192) return 0x9f;
		else if (c == 0x207f) return 0xfc;
		else if (c == 0x20a7) return 0x9e;
		else if (c == 0x2310) return 0xa9;
		else if (c == 0x2320) return 0xf4;
		else if (c == 0x2321) return 0xf5;
		else return -1; }
};
class TEncoding_CP852
{
public:
	static inline TStr GetName() { return "CP852"; }
	static const int toUnicodeTable[8 * 16], fromUnicodeTable1[14 * 16], fromUnicodeTable2[2 * 16], fromUnicodeTable3[11 * 16];
	static int ToUnicode(int c) { Assert(0 <= c && c <= 255);
		if (c < 0x80) return c; else return toUnicodeTable[c - 0x80]; }
	static int FromUnicode(int c) {
		if (0 <= c && c < 0x80) return c;
		else if (0xa0 <= c && c < 0x180) return fromUnicodeTable1[c - 0xa0];
		else if (0x2c0 <= c && c < 0x2e0) return fromUnicodeTable2[c - 0x2c0];
		else if (0x2500 <= c && c < 0x25b0) return fromUnicodeTable3[c - 0x2500];
		else return -1; }
};
class TEncoding_CP1250
{
public:
	static inline TStr GetName() { return "CP1250"; }
	static const int toUnicodeTable[8 * 16], fromUnicodeTable1[14 * 16], fromUnicodeTable2[2 * 16], fromUnicodeTable3[3 * 16];
	static int ToUnicode(int c) { Assert(0 <= c && c <= 255);
		if (c < 0x80) return c; else return toUnicodeTable[c - 0x80]; }
	static int FromUnicode(int c) {
		if (0 <= c && c < 0x80) return c;
		else if (0xa0 <= c && c < 0x180) return fromUnicodeTable1[c - 0xa0];
		else if (0x2c0 <= c && c < 0x2e0) return fromUnicodeTable2[c - 0x2c0];
		else if (0x2010 <= c && c < 0x2040) return fromUnicodeTable3[c - 0x2010];
		else if (c == 0x20ac) return 0x80;
		else if (c == 0x2122) return 0x99;
		else return -1; }
};
template<class TEncoding_>
class T8BitCodec
{
protected:
	typedef TUniVecIdx TVecIdx;
public:
	typedef TEncoding_ TEncoding;
	TUnicodeErrorHandling errorHandling;
	int replacementChar;
	T8BitCodec() : errorHandling(uehIgnore), replacementChar(TUniCodec::DefaultReplacementChar) { }
	T8BitCodec(TUnicodeErrorHandling errorHandling_, int replacementChar_ = TUniCodec::DefaultReplacementChar) :
		errorHandling(errorHandling_), replacementChar(replacementChar_) { }
	static TStr GetName() { return TEncoding::GetName(); }
	void Test() const
	{
		int nDecoded = 0;
		for (int c = 0; c <= 255; c++) {
			int cu = TEncoding::ToUnicode(c); if (cu == -1) continue;
			nDecoded++;
			IAssert(0 <= cu && cu < 0x110000);
			int c2 = TEncoding::FromUnicode(cu);
			IAssert(c2 == c); }
		int nEncoded = 0;
		for (int cu = 0; cu < 0x110000; cu++) {
			int c = TEncoding::FromUnicode(cu); if (c == -1) continue;
			nEncoded++;
			IAssert(0 <= c && c <= 255);
			int cu2 = TEncoding::ToUnicode(c);
			IAssert(cu2 == cu); }
		IAssert(nDecoded == nEncoded);
	}
	template<typename TSrcVec, typename TDestCh>
	size_t ToUnicode(
		const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, const bool clrDest = true) const
	{
		if (clrDest) dest.Clr();
		size_t toDo = srcCount;
		while (toDo-- > 0) {
			int chSrc = ((int) src[TVecIdx(srcIdx)]) & 0xff; srcIdx++;
			int chDest = TEncoding::ToUnicode(chSrc);
			dest.Add(chDest); }
		return srcCount;
	}
	template<typename TSrcVec, typename TDestCh>
	size_t ToUnicode(const TSrcVec& src, TVec<TDestCh>& dest, const bool clrDest = true) const { return ToUnicode(src, 0, src.Len(), dest, clrDest); }
	size_t ToUnicode(const TIntV& src, TIntV& dest, const bool clrDest = true) const { return ToUnicode(src, 0, src.Len(), dest, clrDest); }
	size_t ToUnicode(const TStr& src, TIntV& dest, const bool clrDest = true) const { return ToUnicode(src, 0, src.Len(), dest, clrDest); }
	template<typename TSrcVec, typename TDestVec>
	size_t FromUnicode(
		const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TDestVec& dest, const bool clrDest = true) const
	{
		typedef typename TVecElt<TDestVec>::TElement TDestCh;
		if (clrDest) dest.Clr();
		size_t toDo = srcCount, nEncoded = 0;
		while (toDo-- > 0) {
			int chSrc = (int) src[TVecIdx(srcIdx)]; srcIdx++;
			int chDest = TEncoding::FromUnicode(chSrc);
			if (chDest < 0) {
				switch (errorHandling) {
				case uehThrow: throw TUnicodeException(srcIdx - 1, chSrc, "Invalid character for encoding into " + GetName() + ".");
				case uehAbort: return nEncoded;
				case uehReplace: TVecElt<TDestVec>::Add(dest, TDestCh(replacementChar)); continue;
				case uehIgnore: continue;
				default: Fail; } }
			TVecElt<TDestVec>::Add(dest, TDestCh(chDest)); nEncoded++; }
		return nEncoded;
	}
	template<typename TSrcVec, typename TDestVec>
	size_t FromUnicode(const TSrcVec& src, TDestVec& dest, const bool clrDest = true) const { return FromUnicode(src, 0, src.Len(), dest, clrDest); }
	size_t UniToStr(const TIntV& src, size_t srcIdx, const size_t srcCount, TStr& dest, const bool clrDest = true) const {
		TChA buf; size_t retVal = FromUnicode(src, srcIdx, srcCount, buf, false);
		if (clrDest) dest += buf.CStr(); else dest = buf.CStr();
		return retVal; }
	size_t UniToStr(const TIntV& src, TStr& dest, const bool clrDest = true) const { return UniToStr(src, 0, src.Len(), dest, clrDest); }
};
typedef T8BitCodec<TEncoding_ISO8859_1> TCodec_ISO8859_1;
typedef T8BitCodec<TEncoding_ISO8859_2> TCodec_ISO8859_2;
typedef T8BitCodec<TEncoding_ISO8859_3> TCodec_ISO8859_3;
typedef T8BitCodec<TEncoding_ISO8859_4> TCodec_ISO8859_4;
typedef T8BitCodec<TEncoding_CP852> TCodec_CP852;
typedef T8BitCodec<TEncoding_CP437> TCodec_CP437;
typedef T8BitCodec<TEncoding_CP1250> TCodec_CP1250;
typedef T8BitCodec<TEncoding_YuAscii> TCodec_YuAscii;
typedef enum TUniChCategory_
{
#define DefineUniCat(cat, c) uc ## cat = (int(uchar(c)) & 0xff)
	DefineUniCat(Letter, 'L'),
	DefineUniCat(Mark, 'M'),
	DefineUniCat(Number, 'N'),
	DefineUniCat(Punctuation, 'P'),
	DefineUniCat(Symbol, 'S'),
	DefineUniCat(Separator, 'Z'),
	DefineUniCat(Other, 'C')
#undef DefineUniCat
}
TUniChCategory;
typedef enum TUniChSubCategory_
{
#define DefineUniSubCat(cat, subCat, c) uc ## cat ## subCat = ((uc ## cat) << 8) | (int(uchar(c)) & 0xff)
	DefineUniSubCat(Letter, Uppercase, 'u'),
	DefineUniSubCat(Letter, Lowercase, 'l'),
	DefineUniSubCat(Letter, Titlecase, 't'),
	DefineUniSubCat(Letter, Modifier, 'm'),
	DefineUniSubCat(Letter, Other, 'o'),
	DefineUniSubCat(Mark, Nonspacing, 'n'),
	DefineUniSubCat(Mark, SpacingCombining, 'c'),
	DefineUniSubCat(Mark, Enclosing, 'e'),
	DefineUniSubCat(Number, DecimalDigit, 'd'),
	DefineUniSubCat(Number, Letter, 'l'),
	DefineUniSubCat(Number, Other, 'o'),
	DefineUniSubCat(Punctuation, Connector, 'c'),
	DefineUniSubCat(Punctuation, Dash, 'd'),
	DefineUniSubCat(Punctuation, Open, 's'),
	DefineUniSubCat(Punctuation, Close, 'e'),
	DefineUniSubCat(Punctuation, InitialQuote, 'i'),
	DefineUniSubCat(Punctuation, FinalQuote, 'f'),
	DefineUniSubCat(Punctuation, Other, 'o'),
	DefineUniSubCat(Symbol, Math, 'm'),
	DefineUniSubCat(Symbol, Currency, 'c'),
	DefineUniSubCat(Symbol, Modifier, 'k'),
	DefineUniSubCat(Symbol, Other, 'o'),
	DefineUniSubCat(Separator, Space, 's'),
	DefineUniSubCat(Separator, Line, 'l'),
	DefineUniSubCat(Separator, Paragraph, 'p'),
	DefineUniSubCat(Other, Control, 'c'),
	DefineUniSubCat(Other, Format, 'f'),
	DefineUniSubCat(Other, Surrogate, 's'),
	DefineUniSubCat(Other, PrivateUse, 'o'),
	DefineUniSubCat(Other, NotAssigned, 'n')
}
TUniChSubCategory;
typedef enum TUniChFlags_
{
	ucfCompatibilityDecomposition = 1,
	ucfCompositionExclusion = 1 << 1,
	ucfWbFormat = 1 << 2,
	ucfWbKatakana = 1 << 3,
	ucfWbALetter = 1 << 4,
	ucfWbMidLetter = 1 << 5,
	ucfWbMidNum = 1 << 6,
	ucfWbNumeric = 1 << 7,
	ucfWbExtendNumLet = 1 << 8,
	ucfSbSep = 1 << 9,
	ucfSbFormat = 1 << 10,
	ucfSbSp = 1 << 11,
	ucfSbLower = 1 << 12,
	ucfSbUpper = 1 << 13,
	ucfSbOLetter = 1 << 14,
	ucfSbNumeric = 1 << 15,
	ucfSbATerm = 1 << 16,
	ucfSbSTerm = 1 << 17,
	ucfSbClose = 1 << 18,
	ucfSbMask = ucfSbSep | ucfSbFormat | ucfSbSp | ucfSbLower | ucfSbUpper | ucfSbOLetter | ucfSbNumeric | ucfSbATerm | ucfSbSTerm | ucfSbClose,
	ucfWbMask = ucfWbFormat | ucfWbKatakana | ucfWbALetter | ucfWbMidLetter | ucfWbMidNum | ucfWbNumeric | ucfWbExtendNumLet | ucfSbSep,
	ucfDcpAlphabetic = 1 << 19,
	ucfDcpDefaultIgnorableCodePoint = 1 << 20,
	ucfDcpLowercase = 1 << 21,
	ucfDcpGraphemeBase = 1 << 22,
	ucfDcpGraphemeExtend = 1 << 23,
	ucfDcpIdStart = 1 << 24,
	ucfDcpIdContinue = 1 << 25,
	ucfDcpMath = 1 << 26,
	ucfDcpUppercase = 1 << 27,
	ucfDcpXidStart = 1 << 28,
	ucfDcpXidContinue = 1 << 29,
	ucfDcpMask = ucfDcpAlphabetic | ucfDcpDefaultIgnorableCodePoint | ucfDcpLowercase | ucfDcpGraphemeBase | ucfDcpGraphemeExtend |
		ucfDcpIdStart | ucfDcpIdContinue | ucfDcpMath | ucfDcpUppercase | ucfDcpXidStart | ucfDcpXidContinue,
}
TUniChFlags;
typedef enum TUniChProperties_
{
	ucfPrAsciiHexDigit = 1,
	ucfPrBidiControl = 2,
	ucfPrDash = 4,
	ucfPrDeprecated = 8,
	ucfPrDiacritic = 0x10,
	ucfPrExtender = 0x20,
	ucfPrGraphemeLink = 0x40,
	ucfPrHexDigit = 0x80,
	ucfPrHyphen = 0x100,
	ucfPrIdeographic = 0x200,
	ucfPrJoinControl = 0x400,
	ucfPrLogicalOrderException = 0x800,
	ucfPrNoncharacterCodePoint = 0x1000,
	ucfPrPatternSyntax = 0x2000,
	ucfPrPatternWhiteSpace = 0x4000,
	ucfPrQuotationMark = 0x8000,
	ucfPrSoftDotted = 0x10000,
	ucfPrSTerm = 0x20000,
	ucfPrTerminalPunctuation = 0x40000,
	ucfPrVariationSelector = 0x80000,
	ucfPrWhiteSpace = 0x100000
}
TUniChProperties;
typedef enum TUniChPropertiesX_
{
	ucfPxOtherAlphabetic = 1,
	ucfPxOtherDefaultIgnorableCodePoint = 2,
	ucfPxOtherGraphemeExtend = 4,
	ucfPxOtherIdContinue = 8,
	ucfPxOtherIdStart = 0x10,
	ucfPxOtherLowercase = 0x20,
	ucfPxOtherMath = 0x40,
	ucfPxOtherUppercase = 0x80,
	ucfPxIdsBinaryOperator = 0x100,
	ucfPxIdsTrinaryOperator = 0x200,
	ucfPxRadical = 0x400,
	ucfPxUnifiedIdeograph = 0x800
}
TUniChPropertiesX;
class TUniChInfo
{
public:
	enum {
		ccStarter = 0,
		ccOverlaysAndInterior = 1,
		ccNuktas = 7,
		ccHiraganaKatakanaVoicingMarks = 8,
		ccViramas = 9,
		ccFixedPositionStart = 10,
		ccFixedPositionEnd = 199,
		ccBelowLeftAttached = 200,
		ccBelowAttached = 202,
		ccBelowRightAttached = 204,
		ccLeftAttached = 208,
		ccRightAttached = 210,
		ccAboveLeftAttached = 212,
		ccAboveAttached = 214,
		ccAboveRightAttached = 216,
		ccBelowLeft = 218,
		ccBelow = 220,
		ccBelowRight = 222,
		ccLeft = 224,
		ccRight = 226,
		ccAboveLeft = 228,
		ccAbove = 230,
		ccAboveRight = 232,
		ccDoubleBelow = 233,
		ccDoubleAbove = 234,
		ccBelowIotaSubscript = 240,
		ccInvalid = 255
	};
	char chCat, chSubCat;
	uchar combClass;
	TUniChCategory cat;
	TUniChSubCategory subCat;
	signed char script;
	int simpleUpperCaseMapping, simpleLowerCaseMapping, simpleTitleCaseMapping;
	int decompOffset;
	int nameOffset;
	int flags;
	int properties;
	int propertiesX;
	ushort lineBreak;
	static inline ushort GetLineBreakCode(char c1, char c2) { return ((static_cast<ushort>(static_cast<uchar>(c1)) & 0xff) << 8) | ((static_cast<ushort>(static_cast<uchar>(c2)) & 0xff)); }
	static const ushort LineBreak_Unknown, LineBreak_ComplexContext, LineBreak_Numeric, LineBreak_InfixNumeric, LineBreak_Quotation;
public:
	void InitAfterLoad() {
		cat = (TUniChCategory) chCat;
		subCat = (TUniChSubCategory) (((static_cast<int>(static_cast<uchar>(chCat)) & 0xff) << 8) | (static_cast<int>(static_cast<uchar>(chSubCat)) & 0xff)); }
	void SetCatAndSubCat(const TUniChSubCategory catAndSubCat) {
		cat = (TUniChCategory) ((int(catAndSubCat) >> 8) & 0xff);
		subCat = catAndSubCat;
		chCat = (char) cat; chSubCat = (char) (int(subCat) & 0xff); }
	friend class TUniChDb;
	static inline void LoadUShort(TSIn& SIn, ushort& u) { SIn.LoadBf(&u, sizeof(u)); }
	static inline void LoadSChar(TSIn& SIn, signed char& u) { SIn.LoadBf(&u, sizeof(u)); }
	static inline void SaveUShort(TSOut& SOut, ushort u) { SOut.SaveBf(&u, sizeof(u)); }
	static inline void SaveSChar(TSOut& SOut, signed char u) { SOut.SaveBf(&u, sizeof(u)); }
public:
	void Save(TSOut& SOut) const {
		SOut.Save(chCat); SOut.Save(chSubCat); SOut.Save(combClass); SaveSChar(SOut, script);
		SOut.Save(simpleUpperCaseMapping); SOut.Save(simpleLowerCaseMapping); SOut.Save(simpleTitleCaseMapping);
		SOut.Save(decompOffset); SOut.Save(nameOffset);
		SOut.Save(flags); SOut.Save(properties); SOut.Save(propertiesX); SaveUShort(SOut, lineBreak); }
	void Load(TSIn& SIn) {
		SIn.Load(chCat); SIn.Load(chSubCat); SIn.Load(combClass); LoadSChar(SIn, script);
		SIn.Load(simpleUpperCaseMapping); SIn.Load(simpleLowerCaseMapping); SIn.Load(simpleTitleCaseMapping);
		SIn.Load(decompOffset); SIn.Load(nameOffset);
		SIn.Load(flags); SIn.Load(properties); SIn.Load(propertiesX); LoadUShort(SIn, lineBreak); InitAfterLoad(); }
	explicit TUniChInfo(TSIn& SIn) { Load(SIn); }
	TUniChInfo() : chCat(char(ucOther)), chSubCat(char(ucOtherNotAssigned & 0xff)), combClass(ccInvalid),
		script(-1),simpleUpperCaseMapping(-1), simpleLowerCaseMapping(-1), simpleTitleCaseMapping(-1),
		decompOffset(-1), nameOffset(-1), flags(0), properties(0), propertiesX(0), lineBreak(LineBreak_Unknown) {
		InitAfterLoad(); }
	bool IsDcpFlag(const TUniChFlags flag) const { Assert((flag & ucfDcpMask) == flag); return (flags & flag) == flag; }
	void ClrDcpFlags() { flags = flags & ~ucfDcpMask; }
	void SetDcpFlag(const TUniChFlags flag) { Assert((flag & ucfDcpMask) == flag); flags |= flag; }
	bool IsAlphabetic() const { return IsDcpFlag(ucfDcpAlphabetic); }
	bool IsUppercase() const { return IsDcpFlag(ucfDcpUppercase); }
	bool IsLowercase() const { return IsDcpFlag(ucfDcpLowercase); }
	bool IsMath() const { return IsDcpFlag(ucfDcpMath); }
	bool IsDefaultIgnorable() const { return IsDcpFlag(ucfDcpDefaultIgnorableCodePoint); }
	bool IsGraphemeBase() const { return IsDcpFlag(ucfDcpGraphemeBase); }
	bool IsGraphemeExtend() const { return IsDcpFlag(ucfDcpGraphemeExtend); }
	bool IsIdStart() const { return IsDcpFlag(ucfDcpIdStart); }
	bool IsIdContinue() const { return IsDcpFlag(ucfDcpIdContinue); }
	bool IsXidStart() const { return IsDcpFlag(ucfDcpXidStart); }
	bool IsXidContinue() const { return IsDcpFlag(ucfDcpXidContinue); }
	bool IsProperty(const TUniChProperties flag) const { return (properties & flag) == flag; }
	void SetProperty(const TUniChProperties flag) { properties |= flag; }
	bool IsAsciiHexDigit() const { return IsProperty(ucfPrAsciiHexDigit); }
	bool IsBidiControl() const { return IsProperty(ucfPrBidiControl); }
	bool IsDash() const { return IsProperty(ucfPrDash); }
	bool IsDeprecated() const { return IsProperty(ucfPrDeprecated); }
	bool IsDiacritic() const { return IsProperty(ucfPrDiacritic); }
	bool IsExtender() const { return IsProperty(ucfPrExtender); }
	bool IsGraphemeLink() const { return IsProperty(ucfPrGraphemeLink); }
	bool IsHexDigit() const { return IsProperty(ucfPrHexDigit); }
	bool IsHyphen() const { return IsProperty(ucfPrHyphen); }
	bool IsIdeographic() const { return IsProperty(ucfPrIdeographic); }
	bool IsJoinControl() const { return IsProperty(ucfPrJoinControl); }
	bool IsLogicalOrderException() const { return IsProperty(ucfPrLogicalOrderException); }
	bool IsNoncharacter() const { return IsProperty(ucfPrNoncharacterCodePoint); }
	bool IsQuotationMark() const { return IsProperty(ucfPrQuotationMark); }
	bool IsSoftDotted() const { return IsProperty(ucfPrSoftDotted); }
	bool IsSTerminal() const { return IsProperty(ucfPrSTerm); }
	bool IsTerminalPunctuation() const { return IsProperty(ucfPrTerminalPunctuation); }
	bool IsVariationSelector() const { return IsProperty(ucfPrVariationSelector); }
	bool IsWhiteSpace() const { return IsProperty(ucfPrWhiteSpace); }
	bool IsPropertyX(const TUniChPropertiesX flag) const { return (propertiesX & flag) == flag; }
	void SetPropertyX(const TUniChPropertiesX flag) { propertiesX |= flag; }
	bool IsCompositionExclusion() const { return (flags & ucfCompositionExclusion) == ucfCompositionExclusion; }
	bool IsCompatibilityDecomposition() const { return (flags & ucfCompatibilityDecomposition) == ucfCompatibilityDecomposition; }
	bool IsWbFlag(const TUniChFlags flag) const { Assert((flag & ucfWbMask) == flag); return (flags & flag) == flag; }
	void ClrWbAndSbFlags() { flags = flags & ~(ucfWbMask | ucfSbMask); }
	void SetWbFlag(const TUniChFlags flag) { Assert((flag & ucfWbMask) == flag); flags |= flag; }
	int GetWbFlags() const { return flags & ucfWbMask; }
	bool IsWbFormat() const { return IsWbFlag(ucfWbFormat); }
	TStr GetWbFlagsStr() const { return GetWbFlagsStr(GetWbFlags()); }
	static TStr GetWbFlagsStr(const int flags) { return TStr("") + (flags & ucfWbALetter ? "A" : "") +
		(flags & ucfWbFormat ? "F" : "") + (flags & ucfWbKatakana ? "K" : "") + (flags & ucfWbMidLetter ? "M" : "") +
		(flags & ucfWbMidNum ? "m" : "") + (flags & ucfWbNumeric ? "N" : "") + (flags & ucfWbExtendNumLet ? "E" : ""); }
	bool IsSbFlag(const TUniChFlags flag) const { Assert((flag & ucfSbMask) == flag); return (flags & flag) == flag; }
	void SetSbFlag(const TUniChFlags flag) { Assert((flag & ucfSbMask) == flag); flags |= flag; }
	int GetSbFlags() const { return flags & ucfSbMask; }
	bool IsSbFormat() const { return IsSbFlag(ucfSbFormat); }
	TStr GetSbFlagsStr() const { return GetSbFlagsStr(GetSbFlags()); }
	static TStr GetSbFlagsStr(const int flags) { return TStr("") + (flags & ucfSbSep ? "S" : "") +
		(flags & ucfSbFormat ? "F" : "") + (flags & ucfSbSp ? "_" : "") + (flags & ucfSbLower ? "L" : "") +
		(flags & ucfSbUpper ? "U" : "") + (flags & ucfSbOLetter ? "O" : "") + (flags & ucfSbNumeric ? "N" : "") +
		(flags & ucfSbATerm ? "A" : "") + (flags & ucfSbSTerm ? "T" : "") + (flags & ucfSbClose ? "C" : ""); }
	bool IsSbSep() const { return (flags & ucfSbSep) == ucfSbSep; }
	bool IsGbExtend() const { return IsGraphemeExtend(); }
	bool IsCased() const { return IsUppercase() || IsLowercase() || (subCat == ucLetterTitlecase); }
	TUniChCategory GetCat() const { return (TUniChCategory) cat; }
	TUniChSubCategory GetSubCat() const { return (TUniChSubCategory) subCat; }
	bool IsCurrency() const { return subCat == ucSymbolCurrency; }
	bool IsPrivateUse() const { return subCat == ucOtherPrivateUse; }
	bool IsSurrogate() const { return subCat == ucOtherSurrogate; }
	inline static bool IsValidSubCat(const char chCat, const char chSubCat) {
		static const char s[] = "LuLlLtLmLoMnMcMeNdNlNoPcPdPsPePiPfPoSmScSkSoZsZlZpCcCfCsCoCn";
		for (const char *p = s; *p; p += 2)
			if (chCat == p[0] && chSubCat == p[1]) return true;
		return false; }
};
template<typename TItem_>
class TUniTrie
{
public:
	typedef TItem_ TItem;
protected:
	class TNode {
	public:
		TItem item;
		int child, sib;
		bool terminal;
		TNode() : child(-1), sib(-1), terminal(false) { }
		TNode(const TItem& item_, const int child_, const int sib_, const bool terminal_) : item(item_), child(child_), sib(sib_), terminal(terminal_) { }
	};
	typedef TVec<TNode> TNodeV;
	typedef TPair<TItem, TItem> TItemPr;
	typedef TTriple<TItem, TItem, TItem> TItemTr;
	typedef TUniVecIdx TVecIdx;
	THash<TItem, TVoid> singles;
	THash<TItemPr, TVoid> pairs;
	THash<TItemTr, TInt> roots;
	TNodeV nodes;
public:
	TUniTrie() { }
	void Clr() { singles.Clr(); pairs.Clr(); roots.Clr(); nodes.Clr(); }
	bool Empty() const { return singles.Empty() && pairs.Empty() && roots.Empty(); }
	bool Has1Gram(const TItem& item) const { return singles.IsKey(item); }
	bool Has2Gram(const TItem& last, const TItem& butLast) const { return pairs.IsKey(TItemPr(last, butLast)); }
	int Get3GramRoot(const TItem& last, const TItem& butLast, const TItem& butButLast) const {
		int keyId = roots.GetKeyId(TItemTr(last, butLast, butButLast));
		if (keyId < 0) return 0; else return roots[keyId]; }
	int GetChild(const int parentIdx, const TItem& item) const {
		for (int childIdx = nodes[parentIdx].child; childIdx >= 0; ) {
			const TNode &node = nodes[childIdx];
			if (node.item == item) return childIdx;
			childIdx = node.sib; }
		return -1; }
	bool IsNodeTerminal(const int nodeIdx) const { return nodes[nodeIdx].terminal; }
	template<typename TSrcVec>
	void Add(const TSrcVec& src, const size_t srcIdx, const size_t srcCount)
	{
		IAssert(srcCount > 0);
		if (srcCount == 1) { singles.AddKey(TItem(src[TVecIdx(srcIdx)])); return; }
		if (srcCount == 2) { pairs.AddKey(TItemPr(TItem(src[TVecIdx(srcIdx + 1)]), TItem(src[TVecIdx(srcIdx)]))); return; }
		size_t srcLast = srcIdx + (srcCount - 1);
		TItemTr tr = TItemTr(TItem(src[TVecIdx(srcLast)]), TItem(src[TVecIdx(srcLast - 1)]), TItem(src[TVecIdx(srcLast - 2)]));
		int keyId = roots.GetKeyId(tr), curNodeIdx = -1;
		if (keyId >= 0) curNodeIdx = roots[keyId];
		else { curNodeIdx = nodes.Add(TNode(TItem(0), -1, -1, false)); roots.AddDat(tr, curNodeIdx); }
		if (srcCount > 3) for (size_t srcPos = srcLast - 3; ; )
		{
			const TItem curItem = src[TVecIdx(srcPos)];
			int childNodeIdx = nodes[curNodeIdx].child;
			while (childNodeIdx >= 0) {
				TNode &childNode = nodes[childNodeIdx];
				if (childNode.item == curItem) break;
				childNodeIdx = childNode.sib; }
			if (childNodeIdx < 0) {
				childNodeIdx = nodes.Add(TNode(curItem, -1, nodes[curNodeIdx].child, false));
				nodes[curNodeIdx].child = childNodeIdx; }
			curNodeIdx = childNodeIdx;
			if (srcPos == srcIdx) break; else srcPos--;
		}
		nodes[curNodeIdx].terminal = true;
	}
	template<typename TSrcVec>
	void Add(const TSrcVec& src) { Add(src, 0, (size_t) src.Len()); }
};
class TUniChDb
{
protected:
	void InitAfterLoad();
	typedef TUniVecIdx TVecIdx;
public:
	THash<TInt, TUniChInfo> h;
	TStrPool charNames;
	TStrIntH scripts;
	TIntV decompositions;
	THash<TIntPr, TInt> inverseDec;
	TUniCaseFolding caseFolding;
	TIntIntVH specialCasingLower, specialCasingUpper, specialCasingTitle;
	int scriptUnknown;
	TUniChDb() : scriptUnknown(-1) { }
	explicit TUniChDb(TSIn& SIn) { Load(SIn); }
	void Clr() {
		h.Clr(); charNames.Clr(); decompositions.Clr(); inverseDec.Clr(); caseFolding.Clr();
		specialCasingLower.Clr(); specialCasingUpper.Clr(); specialCasingTitle.Clr();
		scripts.Clr(); }
	void Save(TSOut& SOut) const {
		h.Save(SOut); charNames.Save(SOut); decompositions.Save(SOut);
		inverseDec.Save(SOut); caseFolding.Save(SOut); scripts.Save(SOut);
		specialCasingLower.Save(SOut); specialCasingUpper.Save(SOut); specialCasingTitle.Save(SOut);
		SOut.SaveCs(); }
	void Load(TSIn& SIn) {
		h.Load(SIn); charNames.~TStrPool(); new (&charNames) TStrPool(SIn);
		decompositions.Load(SIn);
		inverseDec.Load(SIn); caseFolding.Load(SIn); scripts.Load(SIn);
		specialCasingLower.Load(SIn); specialCasingUpper.Load(SIn); specialCasingTitle.Load(SIn);
		SIn.LoadCs(); InitAfterLoad(); }
	void LoadBin(const TStr& fnBin) {
		PSIn SIn = TFIn::New(fnBin); Load(*SIn); }
	void Test(const TStr& basePath);
	static TStr GetCaseFoldingFn() { return "CaseFolding.txt"; }
	static TStr GetSpecialCasingFn() { return "SpecialCasing.txt"; }
	static TStr GetUnicodeDataFn() { return "UnicodeData.txt"; }
	static TStr GetCompositionExclusionsFn() { return "CompositionExclusions.txt"; }
	static TStr GetScriptsFn() { return "Scripts.txt"; }
	static TStr GetDerivedCorePropsFn() { return "DerivedCoreProperties.txt"; }
	static TStr GetLineBreakFn() { return "LineBreak.txt"; }
	static TStr GetPropListFn() { return "PropList.txt"; }
	static TStr GetAuxiliaryDir() { return "auxiliary"; }
	static TStr GetWordBreakTestFn() { return "WordBreakTest.txt"; }
	static TStr GetWordBreakPropertyFn() { return "WordBreakProperty.txt"; }
	static TStr GetSentenceBreakTestFn() { return "SentenceBreakTest.txt"; }
	static TStr GetSentenceBreakPropertyFn() { return "SentenceBreakProperty.txt"; }
	static TStr GetNormalizationTestFn() { return "NormalizationTest.txt"; }
	static TStr GetBinFn() { return "UniChDb.bin"; }
	static TStr GetScriptNameUnknown() { return "Unknown"; }
	static TStr GetScriptNameKatakana() { return "Katakana"; }
	static TStr GetScriptNameHiragana() { return "Hiragana"; }
	const TStr& GetScriptName(const int scriptId) const { return scripts.GetKey(scriptId); }
	int GetScriptByName(const TStr& scriptName) const { return scripts.GetKeyId(scriptName); }
	int GetScript(const TUniChInfo& ci) const { int s = ci.script; if (s < 0) s = scriptUnknown; return s; }
	int GetScript(const int cp) const { int i = h.GetKeyId(cp); if (i < 0) return scriptUnknown; else return GetScript(h[i]); }
	const char *GetCharName(const int cp) const { int i = h.GetKeyId(cp); if (i < 0) return 0; int ofs = h[i].nameOffset; return ofs < 0 ? 0 : charNames.GetCStr(ofs); }
	TStr GetCharNameS(const int cp) const {
		const char *p = GetCharName(cp); if (p) return p;
		char buf[20]; sprintf(buf, "U+%04x", cp); return TStr(buf); }
	template<class TSrcVec> void PrintCharNames(FILE *f, const TSrcVec& src, size_t srcIdx, const size_t srcCount, const TStr& prefix) const {
		if (! f) f = stdout;
		for (const size_t srcEnd = srcIdx + srcCount; srcIdx < srcEnd; srcIdx++) {
			fprintf(f, "%s", prefix.CStr());
			int cp = src[TVecIdx(srcIdx)]; fprintf(f, (cp >= 0x10000 ? "U+%05x" : "U+%04x "), cp);
			fprintf(f, " %s\n", GetCharNameS(cp).CStr()); }}
	template<class TSrcVec> void PrintCharNames(FILE *f, const TSrcVec& src, const TStr& prefix) const { PrintCharNames(f, src, 0, src.Len(), prefix); }
	bool IsGetChInfo(const int cp, TUniChInfo& ChInfo) {
		int i = h.GetKeyId(cp);
		if (i < 0) return false; else { ChInfo=h[i]; return true; }}
	TUniChCategory GetCat(const int cp) const { int i = h.GetKeyId(cp); if (i < 0) return ucOther; else return h[i].cat; }
	TUniChSubCategory GetSubCat(const int cp) const { int i = h.GetKeyId(cp); if (i < 0) return ucOtherNotAssigned; else return h[i].subCat; }
	bool IsWbFlag(const int cp, const TUniChFlags flag) const { int i = h.GetKeyId(cp); if (i < 0) return false; else return h[i].IsWbFlag(flag); }
	int GetWbFlags(const int cp) const { int i = h.GetKeyId(cp); if (i < 0) return 0; else return h[i].GetWbFlags(); }
	bool IsSbFlag(const int cp, const TUniChFlags flag) const { int i = h.GetKeyId(cp); if (i < 0) return false; else return h[i].IsSbFlag(flag); }
	int GetSbFlags(const int cp) const { int i = h.GetKeyId(cp); if (i < 0) return 0; else return h[i].GetSbFlags(); }
#define ___UniFwd1(name) bool name(const int cp) const { int i = h.GetKeyId(cp); if (i < 0) return false; else return h[i].name(); }
#define ___UniFwd2(name1, name2) ___UniFwd1(name1) ___UniFwd1(name2)
#define ___UniFwd3(name1, name2, name3) ___UniFwd2(name1, name2) ___UniFwd1(name3)
#define ___UniFwd4(name1, name2, name3, name4) ___UniFwd3(name1, name2, name3) ___UniFwd1(name4)
#define ___UniFwd5(name1, name2, name3, name4, name5) ___UniFwd4(name1, name2, name3, name4) ___UniFwd1(name5)
#define DECLARE_FORWARDED_PROPERTY_METHODS \
	___UniFwd5(IsAsciiHexDigit, IsBidiControl, IsDash, IsDeprecated, IsDiacritic) \
	___UniFwd5(IsExtender, IsGraphemeLink, IsHexDigit, IsHyphen, IsIdeographic)  \
	___UniFwd5(IsJoinControl, IsLogicalOrderException, IsNoncharacter, IsQuotationMark, IsSoftDotted)  \
	___UniFwd4(IsSTerminal, IsTerminalPunctuation, IsVariationSelector, IsWhiteSpace)  \
	___UniFwd5(IsAlphabetic, IsUppercase, IsLowercase, IsMath, IsDefaultIgnorable)  \
	___UniFwd4(IsGraphemeBase, IsGraphemeExtend, IsIdStart, IsIdContinue)  \
	___UniFwd2(IsXidStart, IsXidContinue)  \
	___UniFwd3(IsCompositionExclusion, IsCompatibilityDecomposition, IsSbSep)  \
	___UniFwd1(IsGbExtend)  \
	___UniFwd2(IsCased, IsCurrency)
	DECLARE_FORWARDED_PROPERTY_METHODS
#undef ___UniFwd1
	bool IsPrivateUse(const int cp) const {
		int i = h.GetKeyId(cp); if (i >= 0) return h[i].IsPrivateUse();
		return (0xe000 <= cp && cp <= 0xf8ff) ||
			(0xf0000 <= cp && cp <= 0xffffd) || (0x100000 <= cp && cp <= 0x10fffd); }
	bool IsSurrogate(const int cp) const {
		int i = h.GetKeyId(cp); if (i >= 0) return h[i].IsSurrogate();
		return 0xd800 <= cp && cp <= 0xdcff; }
	int GetCombiningClass(const int cp) const { int i = h.GetKeyId(cp); if (i < 0) return TUniChInfo::ccStarter; else return h[i].combClass; }
	enum {
        HangulSBase = 0xAC00, HangulLBase = 0x1100, HangulVBase = 0x1161, HangulTBase = 0x11A7,
        HangulLCount = 19, HangulVCount = 21, HangulTCount = 28,
        HangulNCount = HangulVCount * HangulTCount,
        HangulSCount = HangulLCount * HangulNCount
	};
protected:
	static bool IsWbIgnored(const TUniChInfo& ci) { return ci.IsGbExtend() || ci.IsWbFormat(); }
	bool IsWbIgnored(const int cp) const { int i = h.GetKeyId(cp); if (i < 0) return false; else return IsWbIgnored(h[i]); }
	template<typename TSrcVec> void WbFindCurOrNextNonIgnored(const TSrcVec& src, size_t& position, const size_t srcEnd) const {
		while (position < srcEnd && IsWbIgnored(src[TVecIdx(position)])) position++; }
	template<typename TSrcVec> void WbFindNextNonIgnored(const TSrcVec& src, size_t& position, const size_t srcEnd) const {
		if (position >= srcEnd) return;
		position++; while (position < srcEnd && IsWbIgnored(src[TVecIdx(position)])) position++; }
	template<typename TSrcVec> void WbFindNextNonIgnoredS(const TSrcVec& src, size_t& position, const size_t srcEnd) const {
		if (position >= srcEnd) return;
		if (IsSbSep(src[TVecIdx(position)])) { position++; return; }
		position++; while (position < srcEnd && IsWbIgnored(src[TVecIdx(position)])) position++; }
	template<typename TSrcVec> bool WbFindPrevNonIgnored(const TSrcVec& src, const size_t srcStart, size_t& position) const {
		if (position <= srcStart) return false;
		while (position > srcStart) {
			position--; if (! IsWbIgnored(src[TVecIdx(position)])) return true; }
		return false; }
	void TestWbFindNonIgnored(const TIntV& src) const;
	void TestWbFindNonIgnored() const;
public:
	template<typename TSrcVec>
	bool FindNextWordBoundary(const TSrcVec& src, const size_t srcIdx, const size_t srcCount, size_t &position) const;
	template<typename TSrcVec>
	void FindWordBoundaries(const TSrcVec& src, const size_t srcIdx, const size_t srcCount, TBoolV& dest) const;
protected:
	void TestFindNextWordOrSentenceBoundary(const TStr& basePath, bool sentence);
protected:
	TUniTrie<TInt> sbExTrie;
	template<typename TSrcVec>
	bool CanSentenceEndHere(const TSrcVec& src, const size_t srcIdx, const size_t position) const;
public:
	template<typename TSrcVec>
	bool FindNextSentenceBoundary(const TSrcVec& src, const size_t srcIdx, const size_t srcCount, size_t &position) const;
	template<typename TSrcVec>
	void FindSentenceBoundaries(const TSrcVec& src, const size_t srcIdx, const size_t srcCount, TBoolV& dest) const;
	void SbEx_Clr() { sbExTrie.Clr(); }
	template<class TSrcVec> void SbEx_Add(const TSrcVec& v) { sbExTrie.Add(v); }
	void SbEx_Add(const TStr& s) {
          TIntV v; int n = s.Len(); v.Gen(n); for (int i = 0; i < n; i++) v[i] = int(uchar(s[i])); SbEx_Add(v); }
	void SbEx_AddUtf8(const TStr& s) { TUniCodec codec; TIntV v; codec.DecodeUtf8(s, v); SbEx_Add(v); }
	int SbEx_AddMulti(const TStr& words, const bool wordsAreUtf8 = true) { TStrV vec; words.SplitOnAllCh('|', vec);
		for (int i = 0; i < vec.Len(); i++) if (wordsAreUtf8) SbEx_AddUtf8(vec[i]); else SbEx_Add(vec[i]);
		return vec.Len(); }
	void SbEx_Set(const TUniTrie<TInt>& newTrie) { sbExTrie = newTrie; }
	int SbEx_SetStdEnglish() {
		static const TStr data = "Ms|Mrs|Mr|Rev|Dr|Prof|Gov|Sen|Rep|Gen|Brig|Col|Capt|Lieut|Lt|Sgt|Pvt|Cmdr|Adm|Corp|St|Mt|Ft|e.g|e. g.|i.e|i. e|ib|ibid|s.v|s. v|s.vv|s. vv";
		SbEx_Clr(); return SbEx_AddMulti(data, false); }
protected:
	template<typename TDestCh>
	void AddDecomposition(const int codePoint, TVec<TDestCh>& dest, const bool compatibility) const;
public:
	template<typename TSrcVec, typename TDestCh>
	void Decompose(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
			TVec<TDestCh>& dest, bool compatibility, bool clrDest = true) const;
	template<typename TSrcVec, typename TDestCh>
	void Decompose(const TSrcVec& src, TVec<TDestCh>& dest, bool compatibility, bool clrDest = true) const {
		Decompose(src, 0, src.Len(), dest, compatibility, clrDest); }
	template<typename TSrcVec, typename TDestCh>
	void Compose(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
			TVec<TDestCh>& dest, bool clrDest = true) const;
	template<typename TSrcVec, typename TDestCh>
	void Compose(const TSrcVec& src, TVec<TDestCh>& dest, bool clrDest = true) const {
		Compose(src, 0, src.Len(), dest, clrDest); }
	template<typename TSrcVec, typename TDestCh>
	void DecomposeAndCompose(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
			TVec<TDestCh>& dest, bool compatibility, bool clrDest = true) const;
	template<typename TSrcVec, typename TDestCh>
	void DecomposeAndCompose(const TSrcVec& src, TVec<TDestCh>& dest, bool compatibility, bool clrDest = true) const {
		DecomposeAndCompose(src, 0, src.Len(), dest, compatibility, clrDest); }
	template<typename TSrcVec, typename TDestCh>
	size_t ExtractStarters(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
			TVec<TDestCh>& dest, bool clrDest = true) const;
	template<typename TSrcVec, typename TDestCh>
	size_t ExtractStarters(const TSrcVec& src, TVec<TDestCh>& dest, bool clrDest = true) const {
		return ExtractStarters(src, 0, src.Len(), dest, clrDest); }
	template<typename TSrcVec>
	size_t ExtractStarters(TSrcVec& src) const {
		TIntV temp; size_t retVal = ExtractStarters(src, temp);
		src.Clr(); for (int i = 0; i < temp.Len(); i++) src.Add(temp[i]);
		return retVal; }
protected:
	void TestComposition(const TStr& basePath);
protected:
	void InitWordAndSentenceBoundaryFlags(const TStr& basePath);
	void InitScripts(const TStr& basePath);
	void InitLineBreaks(const TStr& basePath);
	void InitDerivedCoreProperties(const TStr& basePath);
	void InitPropList(const TStr& basePath);
	void InitSpecialCasing(const TStr& basePath);
	void LoadTxt_ProcessDecomposition(TUniChInfo& ci, TStr s);
public:
	void LoadTxt(const TStr& basePath);
	void SaveBin(const TStr& fnBinUcd);
public:
	typedef enum TCaseConversion_ { ccLower = 0, ccUpper = 1, ccTitle = 2, ccMax = 3 } TCaseConversion;
	template<typename TSrcVec, typename TDestCh> void GetCaseConverted(const TSrcVec& src, size_t srcIdx, const size_t srcCount, TVec<TDestCh>& dest, const bool clrDest, const TCaseConversion how, const bool turkic, const bool lithuanian) const;
	template<typename TSrcVec, typename TDestCh> void GetLowerCase(const TSrcVec& src, size_t srcIdx, const size_t srcCount, TVec<TDestCh>& dest, const bool clrDest = true, const bool turkic = false, const bool lithuanian = false) const { GetCaseConverted(src, srcIdx, srcCount, dest, clrDest, ccLower, turkic, lithuanian); }
	template<typename TSrcVec, typename TDestCh> void GetUpperCase(const TSrcVec& src, size_t srcIdx, const size_t srcCount, TVec<TDestCh>& dest, const bool clrDest = true, const bool turkic = false, const bool lithuanian = false) const { GetCaseConverted(src, srcIdx, srcCount, dest, clrDest, ccUpper, turkic, lithuanian); }
	template<typename TSrcVec, typename TDestCh> void GetTitleCase(const TSrcVec& src, size_t srcIdx, const size_t srcCount, TVec<TDestCh>& dest, const bool clrDest = true, const bool turkic = false, const bool lithuanian = false) const { GetCaseConverted(src, srcIdx, srcCount, dest, clrDest, ccTitle, turkic, lithuanian); }
	template<typename TSrcVec, typename TDestCh> void GetLowerCase(const TSrcVec& src, TVec<TDestCh>& dest, const bool clrDest = true, const bool turkic = false, const bool lithuanian = false) const { GetLowerCase(src, 0, src.Len(), dest, clrDest, turkic, lithuanian); }
	template<typename TSrcVec, typename TDestCh> void GetUpperCase(const TSrcVec& src, TVec<TDestCh>& dest, const bool clrDest = true, const bool turkic = false, const bool lithuanian = false) const { GetUpperCase(src, 0, src.Len(), dest, clrDest, turkic, lithuanian); }
	template<typename TSrcVec, typename TDestCh> void GetTitleCase(const TSrcVec& src, TVec<TDestCh>& dest, const bool clrDest = true, const bool turkic = false, const bool lithuanian = false) const { GetTitleCase(src, 0, src.Len(), dest, clrDest, turkic, lithuanian); }
	template<typename TSrcVec, typename TDestCh> void GetSimpleCaseConverted(const TSrcVec& src, size_t srcIdx, const size_t srcCount, TVec<TDestCh>& dest, const bool clrDest, const TCaseConversion how) const;
	template<typename TSrcVec, typename TDestCh> void GetSimpleLowerCase(const TSrcVec& src, size_t srcIdx, const size_t srcCount, TVec<TDestCh>& dest, const bool clrDest = true) const { GetSimpleCaseConverted(src, srcIdx, srcCount, dest, clrDest, ccLower); }
	template<typename TSrcVec, typename TDestCh> void GetSimpleUpperCase(const TSrcVec& src, size_t srcIdx, const size_t srcCount, TVec<TDestCh>& dest, const bool clrDest = true) const { GetSimpleCaseConverted(src, srcIdx, srcCount, dest, clrDest, ccUpper); }
	template<typename TSrcVec, typename TDestCh> void GetSimpleTitleCase(const TSrcVec& src, size_t srcIdx, const size_t srcCount, TVec<TDestCh>& dest, const bool clrDest = true) const { GetSimpleCaseConverted(src, srcIdx, srcCount, dest, clrDest, ccTitle); }
	template<typename TSrcVec, typename TDestCh> void GetSimpleLowerCase(const TSrcVec& src, TVec<TDestCh>& dest, const bool clrDest = true) const { GetSimpleLowerCase(src, 0, src.Len(), dest, clrDest); }
	template<typename TSrcVec, typename TDestCh> void GetSimpleUpperCase(const TSrcVec& src, TVec<TDestCh>& dest, const bool clrDest = true) const { GetSimpleUpperCase(src, 0, src.Len(), dest, clrDest); }
	template<typename TSrcVec, typename TDestCh> void GetSimpleTitleCase(const TSrcVec& src, TVec<TDestCh>& dest, const bool clrDest = true) const { GetSimpleTitleCase(src, 0, src.Len(), dest, clrDest); }
	template<typename TSrcVec> void ToSimpleCaseConverted(TSrcVec& src, size_t srcIdx, const size_t srcCount, const TCaseConversion how) const;
	template<typename TSrcVec> void ToSimpleUpperCase(TSrcVec& src, size_t srcIdx, const size_t srcCount) const { ToSimpleCaseConverted(src, srcIdx, srcCount, ccUpper); }
	template<typename TSrcVec> void ToSimpleLowerCase(TSrcVec& src, size_t srcIdx, const size_t srcCount) const { ToSimpleCaseConverted(src, srcIdx, srcCount, ccLower); }
	template<typename TSrcVec> void ToSimpleTitleCase(TSrcVec& src, size_t srcIdx, const size_t srcCount) const { ToSimpleCaseConverted(src, srcIdx, srcCount, ccTitle); }
	template<typename TSrcVec> void ToSimpleUpperCase(TSrcVec& src) const { ToSimpleUpperCase(src, 0, src.Len()); }
	template<typename TSrcVec> void ToSimpleLowerCase(TSrcVec& src) const { ToSimpleLowerCase(src, 0, src.Len()); }
	template<typename TSrcVec> void ToSimpleTitleCase(TSrcVec& src) const { ToSimpleTitleCase(src, 0, src.Len()); }
public:
	friend class TUniCaseFolding;
	template<typename TSrcVec, typename TDestCh>
	void GetCaseFolded(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, const bool clrDest, const bool full, const bool turkic = false) const { caseFolding.Fold(src, srcIdx, srcCount, dest, clrDest, full, turkic); }
	template<typename TSrcVec, typename TDestCh>
	void GetCaseFolded(const TSrcVec& src, TVec<TDestCh>& dest, const bool clrDest = true, const bool full = true, const bool turkic = false) const {
		GetCaseFolded(src, 0, src.Len(), dest, clrDest, full, turkic); }
	template<typename TSrcVec> void ToCaseFolded(TSrcVec& src, size_t srcIdx, const size_t srcCount, const bool turkic = false) const { caseFolding.FoldInPlace(src, srcIdx, srcCount, turkic); }
	template<typename TSrcVec> void ToCaseFolded(TSrcVec& src, const bool turkic = false) const { ToCaseFolded(src, 0, src.Len(), turkic); }
protected:
	void TestCaseConversion(const TStr& source, const TStr& trueLc, const TStr& trueTc, const TStr& trueUc, bool turkic, bool lithuanian);
	void TestCaseConversions();
protected:
	class TUcdFileReader
	{
	protected:
		TChA buf;
	public:
		TChA comment;
	protected:
		FILE *f;
		int putBackCh;
		int GetCh() {
			if (putBackCh >= 0) { int c = putBackCh; putBackCh = EOF; return c; }
			return fgetc(f); }
		void PutBack(int c) { Assert(putBackCh == EOF); putBackCh = c; }
		bool ReadNextLine() {
			buf.Clr(); comment.Clr();
			bool inComment = false, first = true;
			while (true) {
				int c = GetCh();
				if (c == EOF) return ! first;
				else if (c == 13) {
					c = GetCh(); if (c != 10) PutBack(c);
					return true; }
				else if (c == 10) return true;
				else if (c == '#') inComment = true;
				if (! inComment) buf += char(c);
				else comment += char(c); }
				}
	private:
		TUcdFileReader& operator = (const TUcdFileReader& r) { Fail; return *((TUcdFileReader *) 0); }
		TUcdFileReader(const TUcdFileReader& r) { Fail; }
	public:
		TUcdFileReader() : f(0) { }
		TUcdFileReader(const TStr& fileName) : f(0), putBackCh(EOF) { Open(fileName); }
		void Open(const TStr& fileName) { Close(); f = fopen(fileName.CStr(), "rt"); IAssertR(f, fileName); putBackCh = EOF; }
		void Close() { putBackCh = EOF; if (f) { fclose(f); f = 0; }}
		~TUcdFileReader() { Close(); }
		bool GetNextLine(TStrV& dest) {
			dest.Clr();
			while (true) {
				if (! ReadNextLine()) return false;
				TStr line = buf; line.ToTrunc();
				if (line.Len() <= 0) continue;
				line.SplitOnAllCh(';', dest, false);
				for (int i = 0; i < dest.Len(); i++) dest[i].ToTrunc();
				return true; }}
		static int ParseCodePoint(const TStr& s) {
			int c; bool ok = s.IsHexInt(true, 0, 0x10ffff, c); IAssertR(ok, s); return c; }
		static void ParseCodePointList(const TStr& s, TIntV& dest, bool ClrDestP = true) {
			if (ClrDestP) dest.Clr();
			TStrV parts; s.SplitOnWs(parts);
			for (int i = 0; i < parts.Len(); i++) {
				int c; bool ok = parts[i].IsHexInt(true, 0, 0x10ffff, c); IAssertR(ok, s);
				dest.Add(c); } }
		static void ParseCodePointRange(const TStr& s, int& from, int &to) {
			int i = s.SearchStr(".."); if (i < 0) { from = ParseCodePoint(s); to = from; return; }
			from = ParseCodePoint(s.GetSubStr(0, i - 1));
			to = ParseCodePoint(s.GetSubStr(i + 2, s.Len() - 1)); }
	};
	class TSubcatHelper
	{
	public:
		bool hasCat; TUniChSubCategory subCat;
		TStrH invalidCatCodes;
		TUniChDb &owner;
		TSubcatHelper(TUniChDb &owner_) : owner(owner_) { }
		void ProcessComment(TUniChDb::TUcdFileReader &reader)
		{
			hasCat = false; subCat = ucOtherNotAssigned;
			if (reader.comment.Len() > 3)
			{
				IAssert(reader.comment[0] == '#');
				IAssert(reader.comment[1] == ' ');
				char chCat = reader.comment[2], chSubCat = reader.comment[3];
				if (reader.comment.Len() > 4) IAssert(isspace(uchar(reader.comment[4])));
				if (TUniChInfo::IsValidSubCat(chCat, chSubCat)) {
					hasCat = true; subCat = (TUniChSubCategory) ((int(uchar(chCat)) << 8) | (int(uchar(chSubCat)))); }
				else invalidCatCodes.AddKey(TStr(chCat) + TStr(chSubCat));
			}
		}
		void SetCat(const int cp) {
			int i = owner.h.GetKeyId(cp); IAssert(i >= 0);
			IAssert(owner.h[i].subCat == ucOtherNotAssigned);
			IAssert(hasCat);
			owner.h[i].SetCatAndSubCat(subCat); }
		void TestCat(const int cp) {
			if (! hasCat) return;
			int i = owner.h.GetKeyId(cp); IAssert(i >= 0);
			IAssert(owner.h[i].subCat == subCat); }
		~TSubcatHelper()
		{
			if (invalidCatCodes.IsKey("L&")) invalidCatCodes.DelKey("L&");
			if (! invalidCatCodes.Empty()) {
				printf("Invalid cat code(s) in the comments: ");
				for (int i = invalidCatCodes.FFirstKeyId(); invalidCatCodes.FNextKeyId(i); )
					printf(" \"%s\"", invalidCatCodes.GetKey(i).CStr());
				printf("\n"); }
		}
	};
};
class TUnicode
{
public:
	TUniCodec codec;
	TUniChDb ucd;
	TUnicode() { Init(); }
	explicit TUnicode(const TStr& fnBinUcd) { ucd.LoadBin(fnBinUcd); Init(); }
	void Init() { InitCodecs(); }
	int DecodeUtf8(const TIntV& src, TIntV& dest) const { return (int) codec.DecodeUtf8(src, dest); }
	int DecodeUtf8(const TStr& src, TIntV& dest) const { return (int) codec.DecodeUtf8(src, dest); }
	int EncodeUtf8(const TIntV& src, TIntV& dest) const { return (int) codec.EncodeUtf8(src, dest); }
	TStr EncodeUtf8Str(const TIntV& src) const { return codec.EncodeUtf8Str(src); }
	static void EncodeUtf8(const uint& Ch, TChA& Dest);
	static TStr EncodeUtf8(const uint& Ch);
	int DecodeUtf16FromBytes(const TIntV& src, TIntV& dest,
		const TUtf16BomHandling bomHandling = bomAllowed,
		const TUniByteOrder defaultByteOrder = boMachineEndian) const {
			return (int) codec.DecodeUtf16FromBytes(src, 0, src.Len(), dest, true, bomHandling, defaultByteOrder); }
	int DecodeUtf16FromWords(const TIntV& src, TIntV& dest,
		const TUtf16BomHandling bomHandling = bomAllowed,
		const TUniByteOrder defaultByteOrder = boMachineEndian) const {
			return (int) codec.DecodeUtf16FromWords(src, 0, src.Len(), dest, true, bomHandling, defaultByteOrder); }
	int EncodeUtf16ToWords(const TIntV& src, TIntV& dest, const bool insertBom,
		const TUniByteOrder destByteOrder = boMachineEndian) const {
			return (int) codec.EncodeUtf16ToWords(src, 0, src.Len(), dest, true, insertBom, destByteOrder); }
	int EncodeUtf16ToBytes(const TIntV& src, TIntV& dest, const bool insertBom,
		const TUniByteOrder destByteOrder = boMachineEndian) const {
			return (int) codec.EncodeUtf16ToBytes(src, 0, src.Len(), dest, true, insertBom, destByteOrder); }
	T8BitCodec<TEncoding_ISO8859_1> iso8859_1;
	T8BitCodec<TEncoding_ISO8859_2> iso8859_2;
	T8BitCodec<TEncoding_ISO8859_3> iso8859_3;
	T8BitCodec<TEncoding_ISO8859_4> iso8859_4;
	T8BitCodec<TEncoding_YuAscii> yuAscii;
	T8BitCodec<TEncoding_CP1250> cp1250;
	T8BitCodec<TEncoding_CP852> cp852;
	T8BitCodec<TEncoding_CP437> cp437;
protected:
	THash<TStr, PCodecBase> codecs;
	static inline TStr NormalizeCodecName(const TStr& name) {
		TStr s = name.GetLc(); s.ChangeStrAll("_", ""); s.ChangeStrAll("-", ""); return s; }
public:
	void RegisterCodec(const TStr& nameList, const PCodecBase& codec) {
		TStrV names; nameList.SplitOnWs(names);
		for (int i = 0; i < names.Len(); i++)
			codecs.AddDat(NormalizeCodecName(names[i]), codec); }
	void UnregisterCodec(const TStr& nameList) {
		TStrV names; nameList.SplitOnWs(names);
		for (int i = 0; i < names.Len(); i++)
			codecs.DelKey(NormalizeCodecName(names[i])); }
	void ClrCodecs() { codecs.Clr(); }
	void InitCodecs();
	PCodecBase GetCodec(const TStr& name) const {
		TStr s = NormalizeCodecName(name);
		PCodecBase p; if (! codecs.IsKeyGetDat(s, p)) p.Clr();
		return p; }
	void GetAllCodecs(TCodecBaseV& dest) const {
		dest.Clr();
		for (int i = codecs.FFirstKeyId(); codecs.FNextKeyId(i); ) {
			PCodecBase codec = codecs[i]; bool found = false;
			for (int j = 0; j < dest.Len(); j++) if (dest[j]() == codec()) { found = true; break; }
			if (! found) dest.Add(codec); }}
	bool FindNextWordBoundary(const TIntV& src, int &position) const {
		if (position < 0) { position = 0; return true; }
		size_t position_; bool retVal = ucd.FindNextWordBoundary(src, 0, src.Len(), position_); position = int(position_); return retVal; }
	void FindWordBoundaries(const TIntV& src, TBoolV& dest) const { ucd.FindWordBoundaries(src, 0, src.Len(), dest); }
	bool FindNextSentenceBoundary(const TIntV& src, int &position) const {
		if (position < 0) { position = 0; return true; }
		size_t position_; bool retVal = ucd.FindNextSentenceBoundary(src, 0, src.Len(), position_); position = int(position_); return retVal; }
	void FindSentenceBoundaries(const TIntV& src, TBoolV& dest) const { ucd.FindSentenceBoundaries(src, 0, src.Len(), dest); }
	void ClrSentenceBoundaryExceptions() { ucd.SbEx_Clr(); }
	void UseEnglishSentenceBoundaryExceptions() { ucd.SbEx_SetStdEnglish(); }
	void Decompose(const TIntV& src, TIntV& dest, bool compatibility) const { ucd.Decompose(src, dest, compatibility, true); }
	void Compose(const TIntV& src, TIntV& dest) const { return ucd.Compose(src, dest, true); }
	void DecomposeAndCompose(const TIntV& src, TIntV& dest, bool compatibility) const { return ucd.DecomposeAndCompose(src, dest, compatibility); }
	int ExtractStarters(const TIntV& src, TIntV& dest) const { return (int) ucd.ExtractStarters(src, dest); }
	int ExtractStarters(TIntV& src) const { return (int) ucd.ExtractStarters(src); }
public:
	typedef TUniChDb::TCaseConversion TCaseConversion;
	void GetLowerCase(const TIntV& src, TIntV& dest) const { ucd.GetLowerCase(src, dest, true, false, false); }
	void GetUpperCase(const TIntV& src, TIntV& dest) const { ucd.GetUpperCase(src, dest, true, false, false); }
	void GetTitleCase(const TIntV& src, TIntV& dest) const { ucd.GetTitleCase(src, dest, true, false, false); }
	void GetSimpleLowerCase(const TIntV& src, TIntV& dest) const { ucd.GetSimpleLowerCase(src, dest, true); }
	void GetSimpleUpperCase(const TIntV& src, TIntV& dest) const { ucd.GetSimpleUpperCase(src, dest, true); }
	void GetSimpleTitleCase(const TIntV& src, TIntV& dest) const { ucd.GetSimpleTitleCase(src, dest, true); }
	void ToSimpleUpperCase(TIntV& src) const { ucd.ToSimpleUpperCase(src); }
	void ToSimpleLowerCase(TIntV& src) const { ucd.ToSimpleLowerCase(src); }
	void ToSimpleTitleCase(TIntV& src) const { ucd.ToSimpleTitleCase(src); }
	void GetCaseFolded(const TIntV& src, TIntV& dest, const bool full = true) const { return ucd.GetCaseFolded(src, dest, true, full, false); }
	void ToCaseFolded(TIntV& src) const { return ucd.ToCaseFolded(src, false); }
	TStr GetUtf8CaseFolded(const TStr& s) const {
		bool isAscii = true;
		for (int i = 0, n = s.Len(); i < n; i++) if (uchar(s[i]) >= 128) { isAscii = false; break; }
		if (isAscii) return s.GetLc();
		TIntV src; DecodeUtf8(s, src);
		TIntV dest; GetCaseFolded(src, dest);
		return EncodeUtf8Str(dest); }
#define ___UniFwd1(name) bool name(const int cp) const { return ucd.name(cp); }
	DECLARE_FORWARDED_PROPERTY_METHODS
#undef DECLARE_FORWARDED_PROPERTY_METHODS
#undef __UniFwd1
	___UniFwd2(IsPrivateUse, IsSurrogate)
	TUniChCategory GetCat(const int cp) const { return ucd.GetCat(cp); }
	TUniChSubCategory GetSubCat(const int cp) const { return ucd.GetSubCat(cp); }
	const char *GetCharName(const int cp) const { return ucd.GetCharName(cp); }
	TStr GetCharNameS(const int cp) const { return ucd.GetCharNameS(cp); }
};
template<typename TSrcVec, typename TDestCh>
size_t TUniCodec::DecodeUtf8(
	const TSrcVec& src, size_t srcIdx, const size_t srcCount,
	TVec<TDestCh>& dest, const bool clrDest) const
{
	size_t nDecoded = 0;
	if (clrDest) dest.Clr();
	const size_t origSrcIdx = srcIdx;
	const size_t srcEnd = srcIdx + srcCount;
	while (srcIdx < srcEnd)
	{
		const size_t charSrcIdx = srcIdx;
		uint c = src[TVecIdx(srcIdx)] & 0xff; srcIdx++;
		if ((c & _1000_0000) == 0) {
			dest.Add(TDestCh(c)); nDecoded++; continue; }
		else if ((c & _1100_0000) == _1000_0000) {
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(charSrcIdx, c, "Invalid character: 10xxxxxx.");
			case uehAbort: return nDecoded;
			case uehReplace: dest.Add(TDestCh(replacementChar)); continue;
			case uehIgnore: continue;
			default: Fail; } }
		else
		{
			uint nMoreBytes = 0, nBits = 0, minVal = 0;
			if ((c & _1110_0000) == _1100_0000) nMoreBytes = 1, nBits = 5, minVal = 0x80;
			else if ((c & _1111_0000) == _1110_0000) nMoreBytes = 2, nBits = 4, minVal = 0x800;
			else if ((c & _1111_1000) == _1111_0000) nMoreBytes = 3, nBits = 3, minVal = 0x10000;
			else if ((c & _1111_1100) == _1111_1000) nMoreBytes = 4, nBits = 2, minVal = 0x200000;
			else if ((c & _1111_1110) == _1111_1100) nMoreBytes = 5, nBits = 1, minVal = 0x4000000;
			else {
				if (strict)  {
					switch (errorHandling) {
					case uehThrow: throw TUnicodeException(charSrcIdx, c, "Invalid character: 1111111x.");
					case uehAbort: return nDecoded;
					case uehReplace: break;
					case uehIgnore: break;
					default: Fail; } }
				nMoreBytes = 5; nBits = 2; minVal = 0x80000000u; }
			uint cOut = c & ((1 << nBits) - 1);
			bool cancel = false;
			for (uint i = 0; i < nMoreBytes && ! cancel; i++) {
				if (! (srcIdx < srcEnd)) {
					switch (errorHandling) {
					case uehThrow: throw TUnicodeException(charSrcIdx, c, TInt::GetStr(nMoreBytes) + " more bytes expected, only " + TInt::GetStr(int(srcEnd - charSrcIdx - 1)) + " available.");
					case uehAbort: return nDecoded;
					case uehReplace: dest.Add(TDestCh(replacementChar)); cancel = true; continue;
					case uehIgnore: cancel = true; continue;
					default: Fail; } }
				c = src[TVecIdx(srcIdx)] & 0xff; srcIdx++;
				if ((c & _1100_0000) != _1000_0000) {
					switch (errorHandling) {
					case uehThrow: throw TUnicodeException(charSrcIdx, c, "Byte " + TInt::GetStr(i) + " of " + TInt::GetStr(nMoreBytes) + " extra bytes should begin with 10xxxxxx.");
					case uehAbort: return nDecoded;
					case uehReplace: dest.Add(TDestCh(replacementChar)); srcIdx--; cancel = true; continue;
					case uehIgnore: srcIdx--; cancel = true; continue;
					default: Fail; } }
				cOut <<= 6; cOut |= (c & _0011_1111); }
			if (cancel) continue;
			if (strict) {
				bool err1 = (cOut < minVal);
				bool err2 = (nMoreBytes > 3 || (nMoreBytes == 3 && cOut > 0x10ffff));
				if (err1 || err2) switch (errorHandling) {
					case uehThrow:
						if (err1) throw TUnicodeException(charSrcIdx, c, "The codepoint 0x" + TInt::GetStr(cOut, "%08x") + " has been represented by too many bytes (" + TInt::GetStr(nMoreBytes + 1) + ").");
						else if (err2) throw TUnicodeException(charSrcIdx, c, "Invalid multibyte sequence: it decodes into 0x" + TInt::GetStr(cOut, "%08x") + ", but only codepoints 0..0x10ffff are valid.");
						else { Fail; break; }
					case uehAbort: return nDecoded;
					case uehReplace: dest.Add(TDestCh(replacementChar)); continue;
					case uehIgnore: continue;
					default: Fail; } }
			if (! (skipBom && (cOut == 0xfffe || cOut == 0xfeff) && charSrcIdx == origSrcIdx)) {
				dest.Add(cOut); nDecoded++; }
		}
	}
	return nDecoded;
}
template<typename TSrcVec, typename TDestCh>
size_t TUniCodec::EncodeUtf8(
	const TSrcVec& src, size_t srcIdx, const size_t srcCount,
	TVec<TDestCh>& dest, const bool clrDest) const
{
	size_t nEncoded = 0;
	for (const size_t srcEnd = srcIdx + srcCount; srcIdx < srcEnd; srcIdx++)
	{
		uint c = uint(src[TVecIdx(srcIdx)]);
		bool err = false;
		if (strict && c > 0x10ffff) {
			err = true;
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(srcIdx, c, "Invalid character (0x" + TInt::GetStr(c, "%x") + "; only characters in the range 0..0x10ffff are allowed).");
			case uehAbort: return nEncoded;
			case uehReplace: c = replacementChar; break;
			case uehIgnore: continue;
			default: Fail; } }
		if (c < 0x80u)
			dest.Add(TDestCh(c & 0xffu));
		else if (c < 0x800u) {
			dest.Add(TDestCh(_1100_0000 | ((c >> 6) & _0001_1111)));
			dest.Add(TDestCh(_1000_0000 | (c & _0011_1111))); }
		else if (c < 0x10000u) {
			dest.Add(TDestCh(_1110_0000 | ((c >> 12) & _0000_1111)));
			dest.Add(TDestCh(_1000_0000 | ((c >> 6) & _0011_1111)));
			dest.Add(TDestCh(_1000_0000 | (c & _0011_1111))); }
		else if (c < 0x200000u) {
			dest.Add(TDestCh(_1111_0000 | ((c >> 18) & _0000_0111)));
			dest.Add(TDestCh(_1000_0000 | ((c >> 12) & _0011_1111)));
			dest.Add(TDestCh(_1000_0000 | ((c >> 6) & _0011_1111)));
			dest.Add(TDestCh(_1000_0000 | (c & _0011_1111))); }
		else if (c < 0x4000000u) {
			dest.Add(TDestCh(_1111_1000 | ((c >> 24) & _0000_0011)));
			dest.Add(TDestCh(_1000_0000 | ((c >> 18) & _0011_1111)));
			dest.Add(TDestCh(_1000_0000 | ((c >> 12) & _0011_1111)));
			dest.Add(TDestCh(_1000_0000 | ((c >> 6) & _0011_1111)));
			dest.Add(TDestCh(_1000_0000 | (c & _0011_1111))); }
		else {
			dest.Add(TDestCh(_1111_1100 | ((c >> 30) & _0000_0011)));
			dest.Add(TDestCh(_1000_0000 | ((c >> 24) & _0011_1111)));
			dest.Add(TDestCh(_1000_0000 | ((c >> 18) & _0011_1111)));
			dest.Add(TDestCh(_1000_0000 | ((c >> 12) & _0011_1111)));
			dest.Add(TDestCh(_1000_0000 | ((c >> 6) & _0011_1111)));
			dest.Add(TDestCh(_1000_0000 | (c & _0011_1111))); }
		if (! err) nEncoded++;
	}
	return nEncoded;
}
template<typename TSrcVec, typename TDestCh>
size_t TUniCodec::DecodeUtf16FromBytes(
	const TSrcVec& src, size_t srcIdx, const size_t srcCount,
	TVec<TDestCh>& dest, const bool clrDest,
	const TUtf16BomHandling bomHandling,
	const TUniByteOrder defaultByteOrder) const
{
	IAssert(srcCount % 2 == 0);
	IAssert(bomHandling == bomAllowed || bomHandling == bomRequired || bomHandling == bomIgnored);
	IAssert(defaultByteOrder == boMachineEndian || defaultByteOrder == boBigEndian || defaultByteOrder == boLittleEndian);
	if (clrDest) dest.Clr();
	size_t nDecoded = 0;
	if (srcCount <= 0) return nDecoded;
	const size_t origSrcIdx = srcIdx, srcEnd = srcIdx + srcCount;
	bool littleEndian = false;
	bool leDefault = (defaultByteOrder == boLittleEndian || (defaultByteOrder == boMachineEndian && IsMachineLittleEndian()));
	if (bomHandling == bomIgnored) littleEndian = leDefault;
	else if (bomHandling == bomAllowed || bomHandling == bomRequired)
	{
		int byte1 = uint(src[TVecIdx(srcIdx)]) & 0xff, byte2 = uint(src[TVecIdx(srcIdx + 1)]) & 0xff;
		if (byte1 == 0xfe && byte2 == 0xff) { littleEndian = false; if (skipBom) srcIdx += 2; }
		else if (byte1 == 0xff && byte2 == 0xfe) { littleEndian = true; if (skipBom) srcIdx += 2; }
		else if (bomHandling == bomAllowed) littleEndian = leDefault;
		else {
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(srcIdx, byte1, "BOM expected at the beginning of the input vector (" + TInt::GetStr(byte1, "%02x") + " " + TInt::GetStr(byte2, "%02x") + " found instead).");
			case uehAbort: case uehReplace: case uehIgnore: return size_t(-1);
			default: Fail; } }
	}
	else Fail;
	while (srcIdx < srcEnd)
	{
		const size_t charSrcIdx = srcIdx;
		uint byte1 = uint(src[TVecIdx(srcIdx)]) & 0xff, byte2 = uint(src[TVecIdx(srcIdx + 1)]) & 0xff; srcIdx += 2;
		uint c = littleEndian ? (byte1 | (byte2 << 8)) : (byte2 | (byte1 << 8));
		if (Utf16FirstSurrogate <= c && c <= Utf16FirstSurrogate + 1023)
		{
			if (! (srcIdx + 2 <= srcEnd)) {
				switch (errorHandling) {
				case uehThrow: throw TUnicodeException(charSrcIdx, c, "The second character of a surrogate pair is missing.");
				case uehAbort: return nDecoded;
				case uehReplace: dest.Add(TDestCh(replacementChar)); continue;
				case uehIgnore: continue;
				default: Fail; } }
			uint byte1 = uint(src[TVecIdx(srcIdx)]) & 0xff, byte2 = uint(src[TVecIdx(srcIdx + 1)]) & 0xff; srcIdx += 2;
			uint c2 = littleEndian ? (byte1 | (byte2 << 8)) : (byte2 | (byte1 << 8));
			if (c2 < Utf16SecondSurrogate || Utf16SecondSurrogate + 1023 < c2) {
				switch (errorHandling) {
				case uehThrow: throw TUnicodeException(charSrcIdx + 2, c2, "The second character of a surrogate pair should be in the range " + TInt::GetStr(Utf16SecondSurrogate, "%04x") + ".." + TInt::GetStr(Utf16SecondSurrogate + 1023, "%04x") + ", not " + TInt::GetStr(c2, "04x") + ".");
				case uehAbort: return nDecoded;
				case uehReplace: dest.Add(TDestCh(replacementChar)); srcIdx -= 2; continue;
				case uehIgnore: srcIdx -= 2; continue;
				default: Fail; } }
			uint cc = ((c - Utf16FirstSurrogate) << 10) | (c2 - Utf16SecondSurrogate);
			cc += 0x10000;
			dest.Add(TDestCh(cc)); nDecoded++; continue;
		}
		else if (strict && Utf16SecondSurrogate <= c && c <= Utf16SecondSurrogate + 1023) {
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(charSrcIdx, c, "This 16-bit value should be used only as the second character of a surrogate pair.");
			case uehAbort: return nDecoded;
			case uehReplace: dest.Add(TDestCh(replacementChar)); continue;
			case uehIgnore: continue;
			default: Fail; } }
		if (charSrcIdx == origSrcIdx && (c == 0xfffeu || c == 0xfeffu) && skipBom) continue;
		dest.Add(TDestCh(c)); nDecoded++;
	}
	return nDecoded;
}
template<typename TSrcVec, typename TDestCh>
size_t TUniCodec::DecodeUtf16FromWords(
	const TSrcVec& src, size_t srcIdx, const size_t srcCount,
	TVec<TDestCh>& dest, bool clrDest,
	const TUtf16BomHandling bomHandling,
	const TUniByteOrder defaultByteOrder) const
{
	IAssert(bomHandling == bomAllowed || bomHandling == bomRequired || bomHandling == bomIgnored);
	IAssert(defaultByteOrder == boMachineEndian || defaultByteOrder == boBigEndian || defaultByteOrder == boLittleEndian);
	if (clrDest) dest.Clr();
	size_t nDecoded = 0;
	if (srcCount <= 0) return nDecoded;
	const size_t origSrcIdx = srcIdx, srcEnd = srcIdx + srcCount;
	bool swap = false;
	bool isMachineLe = IsMachineLittleEndian();
	bool isDefaultLe = (defaultByteOrder == boLittleEndian || (defaultByteOrder == boMachineEndian && isMachineLe));
	if (bomHandling == bomIgnored) swap = (isDefaultLe != isMachineLe);
	else if (bomHandling == bomAllowed || bomHandling == bomRequired)
	{
		int c = uint(src[TVecIdx(srcIdx)]) & 0xffff;
		if (c == 0xfeff) { swap = false; if (skipBom) srcIdx += 1; }
		else if (c == 0xfffe) { swap = true; if (skipBom) srcIdx += 1; }
		else if (bomHandling == bomAllowed) swap = (isMachineLe != isDefaultLe);
		else {
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(srcIdx, c, "BOM expected at the beginning of the input vector (" + TInt::GetStr(c, "%04x") + " found instead).");
			case uehAbort: case uehReplace: case uehIgnore: return size_t(-1);
			default: Fail; } }
	}
	else Fail;
	while (srcIdx < srcEnd)
	{
		const size_t charSrcIdx = srcIdx;
		uint c = uint(src[TVecIdx(srcIdx)]) & 0xffffu; srcIdx++;
		if (swap) c = ((c >> 8) & 0xff) | ((c & 0xff) << 8);
		if (Utf16FirstSurrogate <= c && c <= Utf16FirstSurrogate + 1023)
		{
			if (! (srcIdx < srcEnd)) {
				switch (errorHandling) {
				case uehThrow: throw TUnicodeException(charSrcIdx, c, "The second character of a surrogate pair is missing.");
				case uehAbort: return nDecoded;
				case uehReplace: dest.Add(TDestCh(replacementChar)); continue;
				case uehIgnore: continue;
				default: Fail; } }
			uint c2 = uint(src[TVecIdx(srcIdx)]) & 0xffffu; srcIdx++;
			if (swap) c2 = ((c2 >> 8) & 0xff) | ((c2 & 0xff) << 8);
			if (c2 < Utf16SecondSurrogate || Utf16SecondSurrogate + 1023 < c2) {
				switch (errorHandling) {
				case uehThrow: throw TUnicodeException(charSrcIdx + 1, c2, "The second character of a surrogate pair should be in the range " + TInt::GetStr(Utf16SecondSurrogate, "%04x") + ".." + TInt::GetStr(Utf16SecondSurrogate + 1023, "%04x") + ", not " + TInt::GetStr(c2, "04x") + ".");
				case uehAbort: return nDecoded;
				case uehReplace: dest.Add(TDestCh(replacementChar)); srcIdx -= 1; continue;
				case uehIgnore: srcIdx -= 1; continue;
				default: Fail; } }
			uint cc = ((c - Utf16FirstSurrogate) << 10) | (c2 - Utf16SecondSurrogate);
			cc += 0x10000;
			dest.Add(TDestCh(cc)); nDecoded++; continue;
		}
		else if (strict && Utf16SecondSurrogate <= c && c <= Utf16SecondSurrogate + 1023) {
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(charSrcIdx, c, "This 16-bit value should be used only as the second character of a surrogate pair.");
			case uehAbort: return nDecoded;
			case uehReplace: dest.Add(TDestCh(replacementChar)); continue;
			case uehIgnore: continue;
			default: Fail; } }
		if (charSrcIdx == origSrcIdx && (c == 0xfffeu || c == 0xfeffu) && skipBom) continue;
		dest.Add(TDestCh(c)); nDecoded++;
	}
	return nDecoded;
}
template<typename TSrcVec, typename TDestCh>
size_t TUniCodec::EncodeUtf16ToWords(
	const TSrcVec& src, size_t srcIdx, const size_t srcCount,
	TVec<TDestCh>& dest, const bool clrDest, const bool insertBom,
	const TUniByteOrder destByteOrder) const
{
	bool isMachineLe = IsMachineLittleEndian();
	bool swap = (destByteOrder == boLittleEndian && ! isMachineLe) || (destByteOrder == boBigEndian && isMachineLe);
	size_t nEncoded = 0, srcEnd = srcIdx + srcCount;
	if (insertBom) { dest.Add(TDestCh(swap ? 0xfffeu : 0xfeffu)); nEncoded++; }
	while (srcIdx < srcEnd)
	{
		uint c = uint(src[TVecIdx(srcIdx)]); srcIdx++;
		if (! (c <= 0x10ffffu)) {
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(srcIdx - 1, c, "UTF-16 only supports characters in the range 0..10ffff (not " + TUInt::GetStr(c, "%08x") + ").");
			case uehAbort: return nEncoded;
			case uehReplace: dest.Add(TDestCh(swap ? SwapBytes(replacementChar) : replacementChar)); continue;
			case uehIgnore: continue;
			default: Fail; } }
		if (Utf16FirstSurrogate <= c && c < Utf16FirstSurrogate + 1023) {
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(srcIdx - 1, c, "UTF-16 cannot encode " + TUInt::GetStr(c, "%04x") + " as it belongs to the first surrogate range (" + TUInt::GetStr(Utf16FirstSurrogate, "%04x") + ".." + TUInt::GetStr(Utf16FirstSurrogate + 1023, "%04x") + ").");
			case uehAbort: return nEncoded;
			case uehReplace: dest.Add(TDestCh(swap ? SwapBytes(replacementChar) : replacementChar)); continue;
			case uehIgnore: continue;
			default: Fail; } }
		if (Utf16SecondSurrogate <= c && c < Utf16SecondSurrogate + 1023) {
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(srcIdx - 1, c, "The character " + TUInt::GetStr(c, "%04x") + " belongs to the second surrogate range (" + TUInt::GetStr(Utf16FirstSurrogate, "%04x") + ".." + TUInt::GetStr(Utf16FirstSurrogate + 1023, "%04x") + "), which is not allowed with strict == true.");
			case uehAbort: return nEncoded;
			case uehReplace: dest.Add(TDestCh(swap ? SwapBytes(replacementChar) : replacementChar)); continue;
			case uehIgnore: continue;
			default: Fail; } }
		if (c <= 0xffffu) {
			if (swap) c = ((c >> 8) & 0xff) | ((c & 0xff) << 8);
			dest.Add(TDestCh(c)); nEncoded++; continue; }
		c -= 0x10000u; IAssert( c <= 0xfffffu);
		uint c1 = (c >> 10) & 1023, c2 = c & 1023;
		c1 += Utf16FirstSurrogate; c2 += Utf16SecondSurrogate;
		if (swap) {
			c1 = ((c1 >> 8) & 0xff) | ((c1 & 0xff) << 8);
			c2 = ((c2 >> 8) & 0xff) | ((c2 & 0xff) << 8); }
		dest.Add(TDestCh(c1));
		dest.Add(TDestCh(c2));
		nEncoded++; continue;
	}
	return nEncoded;
}
template<typename TSrcVec, typename TDestCh>
size_t TUniCodec::EncodeUtf16ToBytes(
	const TSrcVec& src, size_t srcIdx, const size_t srcCount,
	TVec<TDestCh>& dest, const bool clrDest, const bool insertBom,
	const TUniByteOrder destByteOrder) const
{
	bool isDestLe = (destByteOrder == boLittleEndian || (destByteOrder == boMachineEndian && IsMachineLittleEndian()));
	size_t nEncoded = 0, srcEnd = srcIdx + srcCount;
	if (insertBom) { dest.Add(isDestLe ? 0xff : 0xfe); dest.Add(isDestLe ? 0xfe : 0xff); nEncoded++; }
	while (srcIdx < srcEnd)
	{
		uint c = uint(src[TVecIdx(srcIdx)]); srcIdx++;
		if (! (c <= 0x10ffffu)) {
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(srcIdx - 1, c, "UTF-16 only supports characters in the range 0..10ffff (not " + TUInt::GetStr(c, "%08x") + ").");
			case uehAbort: return nEncoded;
#define ___OutRepl if (isDestLe) { dest.Add(replacementChar & 0xff); dest.Add((replacementChar >> 8) & 0xff); } else { dest.Add((replacementChar >> 8) & 0xff); dest.Add(replacementChar & 0xff); }
			case uehReplace: ___OutRepl; continue;
			case uehIgnore: continue;
			default: Fail; } }
		if (Utf16FirstSurrogate <= c && c < Utf16FirstSurrogate + 1023) {
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(srcIdx - 1, c, "UTF-16 cannot encode " + TUInt::GetStr(c, "%04x") + " as it belongs to the first surrogate range (" + TUInt::GetStr(Utf16FirstSurrogate, "%04x") + ".." + TUInt::GetStr(Utf16FirstSurrogate + 1023, "%04x") + ").");
			case uehAbort: return nEncoded;
			case uehReplace: ___OutRepl; continue;
			case uehIgnore: continue;
			default: Fail; } }
		if (Utf16SecondSurrogate <= c && c < Utf16SecondSurrogate + 1023) {
			switch (errorHandling) {
			case uehThrow: throw TUnicodeException(srcIdx - 1, c, "The character " + TUInt::GetStr(c, "%04x") + " belongs to the second surrogate range (" + TUInt::GetStr(Utf16FirstSurrogate, "%04x") + ".." + TUInt::GetStr(Utf16FirstSurrogate + 1023, "%04x") + "), which is not allowed with strict == true.");
			case uehAbort: return nEncoded;
			case uehReplace: ___OutRepl; continue;
			case uehIgnore: continue;
			default: Fail; } }
#undef ___OutRepl
		if (c <= 0xffffu) {
			if (isDestLe) { dest.Add(c & 0xff); dest.Add((c >> 8) & 0xff); }
			else { dest.Add((c >> 8) & 0xff); dest.Add(c & 0xff); }
			nEncoded++; continue; }
		c -= 0x10000u; IAssert( c <= 0xfffffu);
		uint c1 = (c >> 10) & 1023, c2 = c & 1023;
		c1 += Utf16FirstSurrogate; c2 += Utf16SecondSurrogate;
		if (isDestLe) { dest.Add(c1 & 0xff); dest.Add((c1 >> 8) & 0xff); dest.Add(c2 & 0xff); dest.Add((c2 >> 8) & 0xff); }
		else { dest.Add((c1 >> 8) & 0xff); dest.Add(c1 & 0xff); dest.Add((c2 >> 8) & 0xff); dest.Add(c2 & 0xff); }
		nEncoded++; continue;
	}
	return nEncoded;
}
template<typename TSrcVec>
bool TUniChDb::FindNextWordBoundary(const TSrcVec& src, const size_t srcIdx, const size_t srcCount, size_t &position) const
{
	if (position < srcIdx) { position = srcIdx; return true; }
	const size_t srcEnd = srcIdx + srcCount;
	if (position >= srcEnd) return false;
	size_t origPos = position;
	if (IsWbIgnored(src[TVecIdx(position)])) {
		if (! WbFindPrevNonIgnored(src, srcIdx, position))
			position = origPos;
	}
	size_t posPrev = position;
	if (! WbFindPrevNonIgnored(src, srcIdx, posPrev)) posPrev = position;
	if (position == origPos && position + 1 < srcEnd && IsSbSep(src[TVecIdx(position)]) && IsWbIgnored(src[TVecIdx(position + 1)])) { position += 1; return true; }
	size_t posNext = position; WbFindNextNonIgnored(src, posNext, srcEnd);
	size_t posNext2;
	int cPrev = (posPrev < position ? (int) src[TVecIdx(posPrev)] : -1), cCur = (position < srcEnd ? (int) src[TVecIdx(position)] : -1);
	int cNext = (position < posNext && posNext < srcEnd ? (int) src[TVecIdx(posNext)] : -1);
	int wbfPrev = GetWbFlags(cPrev), wbfCur = GetWbFlags(cCur), wbfNext = GetWbFlags(cNext);
	int cNext2, wbfNext2;
	for ( ; position < srcEnd; posPrev = position, position = posNext, posNext = posNext2,
							   cPrev = cCur, cCur = cNext, cNext = cNext2,
							   wbfPrev = wbfCur, wbfCur = wbfNext, wbfNext = wbfNext2)
	{
		posNext2 = posNext; WbFindNextNonIgnored(src, posNext2, srcEnd);
		cNext2 = (posNext < posNext2 && posNext2 < srcEnd ? (int) src[TVecIdx(posNext2)] : -1);
		wbfNext2 = GetWbFlags(cNext2);
#define TestCurNext(curFlag, nextFlag) if ((wbfCur & curFlag) == curFlag && (wbfNext & nextFlag) == nextFlag) continue
#define TestCurNext2(curFlag, nextFlag, next2Flag) if ((wbfCur & curFlag) == curFlag && (wbfNext & nextFlag) == nextFlag && (wbfNext2 & next2Flag) == next2Flag) continue
#define TestPrevCurNext(prevFlag, curFlag, nextFlag) if ((wbfPrev & prevFlag) == prevFlag && (wbfCur & curFlag) == curFlag && (wbfNext & nextFlag) == nextFlag) continue
		if (cCur == 13 && cNext == 10) continue;
		TestCurNext(ucfWbALetter, ucfWbALetter);
		TestCurNext2(ucfWbALetter, ucfWbMidLetter, ucfWbALetter);
		TestPrevCurNext(ucfWbALetter, ucfWbMidLetter, ucfWbALetter);
		TestCurNext(ucfWbNumeric, ucfWbNumeric);
		TestCurNext(ucfWbALetter, ucfWbNumeric);
		TestCurNext(ucfWbNumeric, ucfWbALetter);
		TestPrevCurNext(ucfWbNumeric, ucfWbMidNum, ucfWbNumeric);
		TestCurNext2(ucfWbNumeric, ucfWbMidNum, ucfWbNumeric);
		TestCurNext(ucfWbKatakana, ucfWbKatakana);
		if ((wbfCur & (ucfWbALetter | ucfWbNumeric | ucfWbKatakana | ucfWbExtendNumLet)) != 0 &&
			(wbfNext & ucfWbExtendNumLet) == ucfWbExtendNumLet) continue;
		if ((wbfCur & ucfWbExtendNumLet) == ucfWbExtendNumLet &&
			(wbfNext & (ucfWbALetter | ucfWbNumeric | ucfWbKatakana)) != 0) continue;
		position = posNext; return true;
#undef TestCurNext
#undef TestCurNext2
#undef TestPrevCurNext
	}
	IAssert(position == srcEnd);
	return true;
}
template<typename TSrcVec>
void TUniChDb::FindWordBoundaries(const TSrcVec& src, const size_t srcIdx, const size_t srcCount, TBoolV& dest) const
{
	if (size_t(dest.Len()) != srcCount + 1) dest.Gen(TVecIdx(srcCount + 1));
	dest.PutAll(false);
	size_t position = srcIdx;
	dest[TVecIdx(position - srcIdx)] = true;
	while (position < srcIdx + srcCount)
	{
		size_t oldPos = position;
		FindNextWordBoundary(src, srcIdx, srcCount, position);
    if (oldPos >= position) {
		  Assert(oldPos < position);
    }
    Assert(position <= srcIdx + srcCount);
		dest[TVecIdx(position - srcIdx)] = true;
	}
	Assert(dest[TVecIdx(srcCount)]);
}
template<typename TSrcVec>
bool TUniChDb::CanSentenceEndHere(const TSrcVec& src, const size_t srcIdx, const size_t position) const
{
	if (sbExTrie.Empty()) return true;
	size_t pos = position;
	if (! WbFindPrevNonIgnored(src, srcIdx, pos)) return true;
	int c = (int) src[TVecIdx(pos)]; int sfb = GetSbFlags(c);
	if ((c & ucfSbSep) == ucfSbSep) {
		if (! WbFindPrevNonIgnored(src, srcIdx, pos)) return true;
		c = (int) src[TVecIdx(pos)]; sfb = GetSbFlags(c); }
	while ((sfb & ucfSbSp) == ucfSbSp) {
		if (! WbFindPrevNonIgnored(src, srcIdx, pos)) return true;
		c = (int) src[TVecIdx(pos)]; sfb = GetSbFlags(c); }
	while ((sfb & ucfSbSp) == ucfSbSp) {
		if (! WbFindPrevNonIgnored(src, srcIdx, pos)) return true;
		c = (int) src[TVecIdx(pos)]; sfb = GetSbFlags(c); }
	while ((sfb & (ucfSbATerm | ucfSbSTerm)) != 0) {
		if (! WbFindPrevNonIgnored(src, srcIdx, pos)) return true;
		c = (int) src[TVecIdx(pos)]; sfb = GetSbFlags(c); }
	int cLast = c, cButLast = -1, cButButLast = -1, len = 1, node = -1;
	while (true)
	{
		bool atEnd = (! WbFindPrevNonIgnored(src, srcIdx, pos));
		c = (atEnd ? -1 : (int) src[TVecIdx(pos)]);
		TUniChCategory cat = GetCat(c);
		if (atEnd || ! (cat == ucLetter || cat == ucNumber || cat == ucSymbol)) {
			if (len == 1) return ! sbExTrie.Has1Gram(cLast);
			if (len == 2) return ! sbExTrie.Has2Gram(cLast, cButLast);
			IAssert(len >= 3); IAssert(node >= 0);
			if (sbExTrie.IsNodeTerminal(node)) return false;
			if (atEnd) return true; }
		if (len == 1) { cButLast = c; len++; }
		else if (len == 2) { cButButLast = c; len++;
			node = sbExTrie.Get3GramRoot(cLast, cButLast, cButButLast);
			if (node < 0) return true; }
		else {
			node = sbExTrie.GetChild(node, c);
			if (node < 0) return true; }
	}
}
template<typename TSrcVec>
bool TUniChDb::FindNextSentenceBoundary(const TSrcVec& src, const size_t srcIdx, const size_t srcCount, size_t &position) const
{
	if (position < srcIdx) { position = srcIdx; return true; }
	const size_t srcEnd = srcIdx + srcCount;
	if (position >= srcEnd) return false;
	size_t origPos = position;
	if (IsWbIgnored(src[TVecIdx(position)])) {
		if (! WbFindPrevNonIgnored(src, srcIdx, position))
			position = origPos;
	}
	size_t posPrev = position;
	if (! WbFindPrevNonIgnored(src, srcIdx, posPrev)) posPrev = position;
	if (position == origPos && position + 1 < srcEnd && IsSbSep(src[TVecIdx(position)]) && IsWbIgnored(src[TVecIdx(position + 1)])) { position += 1; return true; }
	size_t posNext = position; WbFindNextNonIgnored(src, posNext, srcEnd);
	size_t posNext2;
	int cPrev = (posPrev < position ? (int) src[TVecIdx(posPrev)] : -1), cCur = (position < srcEnd ? (int) src[TVecIdx(position)] : -1);
	int cNext = (position < posNext && posNext < srcEnd ? (int) src[TVecIdx(posNext)] : -1);
	int sbfPrev = GetSbFlags(cPrev), sbfCur = GetSbFlags(cCur), sbfNext = GetSbFlags(cNext);
	int cNext2, sbfNext2;
	typedef enum { stInit, stATerm, stATermSp, stATermSep, stSTerm, stSTermSp, stSTermSep } TPeekBackState;
	TPeekBackState backState;
	{
		size_t pos = position;
		bool wasSep = false, wasSp = false, wasATerm = false, wasSTerm = false;
		while (true)
		{
			if (! WbFindPrevNonIgnored(src, srcIdx, pos)) break;
			int cp = (int) src[TVecIdx(pos)]; int sbf = GetSbFlags(cp);
			if ((sbf & ucfSbSep) == ucfSbSep) {
				wasSep = true;
				if (! WbFindPrevNonIgnored(src, srcIdx, pos)) break;
				cp = (int) src[TVecIdx(pos)]; sbf = GetSbFlags(cp); }
			bool stop = false;
			while ((sbf & ucfSbSp) == ucfSbSp) {
				wasSp = true;
				if (! WbFindPrevNonIgnored(src, srcIdx, pos)) { stop = true; break; }
				cp = (int) src[TVecIdx(pos)]; sbf = GetSbFlags(cp); }
			if (stop) break;
			while ((sbf & ucfSbClose) == ucfSbClose) {
				if (! WbFindPrevNonIgnored(src, srcIdx, pos)) { stop = true; break; }
				cp = (int) src[TVecIdx(pos)]; sbf = GetSbFlags(cp); }
			if (stop) break;
			wasATerm = ((sbf & ucfSbATerm) == ucfSbATerm);
			wasSTerm = ((sbf & ucfSbSTerm) == ucfSbSTerm);
			break;
		}
		if (wasATerm) backState = (wasSep ? stATermSep : wasSp ? stATermSp : stATerm);
		else if (wasSTerm) backState = (wasSep ? stSTermSep : wasSp ? stSTermSp : stSTerm);
		else backState = stInit;
	}
	typedef enum { stUnknown, stLower, stNotLower } TPeekAheadState;
	TPeekAheadState aheadState = stUnknown;
	for ( ; position < srcEnd; posPrev = position, position = posNext, posNext = posNext2,
							   cPrev = cCur, cCur = cNext, cNext = cNext2,
							   sbfPrev = sbfCur, sbfCur = sbfNext, sbfNext = sbfNext2)
	{
		posNext2 = posNext; WbFindNextNonIgnored(src, posNext2, srcEnd);
		cNext2 = (posNext < posNext2 && posNext2 < srcEnd ? (int) src[TVecIdx(posNext2)] : -1);
		sbfNext2 = GetSbFlags(cNext2);
#define TestCur(curFlag) ((sbfCur & ucfSb##curFlag) == ucfSb##curFlag)
#define Trans(curFlag, newState) if (TestCur(curFlag)) { backState = st##newState; break; }
		switch (backState) {
			case stInit: Trans(ATerm, ATerm); Trans(STerm, STerm); break;
			case stATerm: Trans(Sp, ATermSp); Trans(Sep, ATermSep); Trans(ATerm, ATerm); Trans(STerm, STerm); Trans(Close, ATerm); backState = stInit; break;
			case stSTerm: Trans(Sp, STermSp); Trans(Sep, STermSep); Trans(ATerm, ATerm); Trans(STerm, STerm); Trans(Close, STerm); backState = stInit; break;
			case stATermSp: Trans(Sp, ATermSp); Trans(Sep, ATermSep); Trans(ATerm, ATerm); Trans(STerm, STerm); backState = stInit; break;
			case stSTermSp: Trans(Sp, STermSp); Trans(Sep, STermSep); Trans(ATerm, ATerm); Trans(STerm, STerm); backState = stInit; break;
			case stATermSep: Trans(ATerm, ATerm); Trans(STerm, STerm); backState = stInit; break;
			case stSTermSep: Trans(ATerm, ATerm); Trans(STerm, STerm); backState = stInit; break;
			default: IAssert(false); }
#undef Trans
#undef TestCur
#define IsPeekAheadSkippable(sbf) ((sbf & (ucfSbOLetter | ucfSbUpper | ucfSbLower | ucfSbSep | ucfSbSTerm | ucfSbATerm)) == 0)
		if (! IsPeekAheadSkippable(sbfCur)) {
			bool isLower = ((sbfCur & ucfSbLower) == ucfSbLower);
			if (aheadState == stLower) IAssert(isLower);
			else if (aheadState == stNotLower) IAssert(! isLower);
			aheadState = stUnknown; }
		if (aheadState == stUnknown)
		{
			size_t pos = posNext;
			while (pos < srcEnd) {
				int cp = (int) src[TVecIdx(pos)]; int sbf = GetSbFlags(cp);
				if (! IsPeekAheadSkippable(sbf)) {
					if ((sbf & ucfSbLower) == ucfSbLower) aheadState = stLower;
					else aheadState = stNotLower;
					break; }
				WbFindNextNonIgnored(src, pos, srcEnd); }
			if (! (pos < srcEnd)) aheadState = stNotLower;
		}
#undef IsPeekAheadSkippable
#define TestCurNext(curFlag, nextFlag) if ((sbfCur & curFlag) == curFlag && (sbfNext & nextFlag) == nextFlag) continue
#define TestCurNext2(curFlag, nextFlag, next2Flag) if ((sbfCur & curFlag) == curFlag && (sbfNext & nextFlag) == nextFlag && (sbfNext2 & next2Flag) == next2Flag) continue
#define TestPrevCurNext(prevFlag, curFlag, nextFlag) if ((sbfPrev & prevFlag) == prevFlag && (sbfCur & curFlag) == curFlag && (sbfNext & nextFlag) == nextFlag) continue
		if (cCur == 13 && cNext == 10) continue;
		if ((sbfCur & ucfSbSep) == ucfSbSep) {
			if (! CanSentenceEndHere(src, srcIdx, position)) continue;
			position = posNext; return true; }
		TestCurNext(ucfSbATerm, ucfSbNumeric);
		TestPrevCurNext(ucfSbUpper, ucfSbATerm, ucfSbUpper);
		if ((backState == stATerm || backState == stATermSp || backState == stSTerm || backState == stSTermSp) &&
			(sbfNext & (ucfSbSTerm | ucfSbATerm)) != 0) continue;
		if ((backState == stATerm || backState == stATermSp) && aheadState == stLower) continue;
		if ((backState == stATerm || backState == stSTerm) && (sbfNext & (ucfSbClose | ucfSbSp | ucfSbSep)) != 0) continue;
		if (backState == stATerm || backState == stATermSp || backState == stATermSep || backState == stSTerm || backState == stSTermSp || backState == stSTermSep) {
			if ((sbfNext & (ucfSbSp | ucfSbSep)) != 0) continue;
			if (! CanSentenceEndHere(src, srcIdx, position)) continue;
			position = posNext; return true; }
		continue;
#undef TestCurNext
#undef TestCurNext2
#undef TestPrevCurNext
	}
	IAssert(position == srcEnd);
	return true;
}
template<typename TSrcVec>
void TUniChDb::FindSentenceBoundaries(const TSrcVec& src, const size_t srcIdx, const size_t srcCount, TBoolV& dest) const
{
	if (size_t(dest.Len()) != srcCount + 1) dest.Gen(TVecIdx(srcCount + 1));
	dest.PutAll(false);
	size_t position = srcIdx;
	dest[TVecIdx(position - srcIdx)] = true;
	while (position < srcIdx + srcCount)
	{
		size_t oldPos = position;
		FindNextSentenceBoundary(src, srcIdx, srcCount, position);
    if (oldPos >= position) {
		  Assert(oldPos < position);
    }
    Assert(position <= srcIdx + srcCount);
		dest[TVecIdx(position - srcIdx)] = true;
	}
	Assert(dest[TVecIdx(srcCount)]);
}
template<typename TSrcVec, typename TDestCh>
void TUniChDb::GetCaseConverted(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
								TVec<TDestCh>& dest, const bool clrDest,
								const TUniChDb::TCaseConversion how,
								const bool turkic, const bool lithuanian) const
{
	const TIntIntVH &specials = (how == ccUpper ? specialCasingUpper : how == ccLower ? specialCasingLower : how == ccTitle ? specialCasingTitle : *((TIntIntVH *) 0));
	if (clrDest) dest.Clr();
	enum {
		GreekCapitalLetterSigma = 0x3a3,
		GreekSmallLetterSigma = 0x3c3,
		GreekSmallLetterFinalSigma = 0x3c2,
		LatinCapitalLetterI = 0x49,
		LatinCapitalLetterJ = 0x4a,
		LatinCapitalLetterIWithOgonek = 0x12e,
		LatinCapitalLetterIWithGrave = 0xcc,
		LatinCapitalLetterIWithAcute = 0xcd,
		LatinCapitalLetterIWithTilde = 0x128,
		LatinCapitalLetterIWithDotAbove = 0x130,
		LatinSmallLetterI = 0x69,
		CombiningDotAbove = 0x307
	};
	bool seenCased = false, seenTwoCased = false; int cpFirstCased = -1;
	size_t nextWordBoundary = srcIdx;
	TBoolV wordBoundaries; bool wbsKnown = false;
	for (const size_t origSrcIdx = srcIdx, srcEnd = srcIdx + srcCount; srcIdx < srcEnd; )
	{
		int cp = src[TVecIdx(srcIdx)]; srcIdx++;
		TUniChDb::TCaseConversion howHere;
		if (how != ccTitle) howHere = how;
		else {
			if (srcIdx - 1 == nextWordBoundary) {
				seenCased = false; seenTwoCased = false; cpFirstCased = -1;
				size_t next = nextWordBoundary; FindNextWordBoundary(src, origSrcIdx, srcCount, next);
				IAssert(next > nextWordBoundary); nextWordBoundary = next; }
			bool isCased = IsCased(cp);
			if (isCased && ! seenCased) { howHere = ccTitle; seenCased = true; cpFirstCased = cp; }
			else { howHere = ccLower;
				if (isCased && seenCased) seenTwoCased = true; }
		}
		if (cp == GreekCapitalLetterSigma && howHere == ccLower)
		{
			if (! wbsKnown) { FindWordBoundaries(src, origSrcIdx, srcCount, wordBoundaries); wbsKnown = true; }
			size_t srcIdx2 = srcIdx; bool casedAfter = false;
			if (how == ccTitle)
				printf("!");
			while (! wordBoundaries[TVecIdx(srcIdx2 - origSrcIdx)])
			{
				int cp2 = src[TVecIdx(srcIdx2)]; srcIdx2++;
				if (IsCased(cp2)) { casedAfter = true; break; }
			}
			if (! casedAfter)
			{
				srcIdx2 = srcIdx - 1; bool casedBefore = false;
				while (! wordBoundaries[TVecIdx(srcIdx2 - origSrcIdx)])
				{
					--srcIdx2; int cp2 = src[TVecIdx(srcIdx2)];
					if (IsCased(cp2)) { casedBefore = true; break; }
				}
				if (casedBefore) {
					dest.Add(GreekSmallLetterFinalSigma); Assert(howHere == ccLower); continue; }
			}
			dest.Add(GreekSmallLetterSigma); continue;
		}
		else if (lithuanian)
		{
			if (howHere == ccLower)
			{
				if (cp == LatinCapitalLetterI || cp == LatinCapitalLetterJ || cp == LatinCapitalLetterIWithOgonek)
				{
					bool moreAbove = false;
					for (size_t srcIdx2 = srcIdx; srcIdx2 < srcEnd; )
					{
						const int cp2 = src[TVecIdx(srcIdx2)]; srcIdx2++;
						const int cc2 = GetCombiningClass(cp2);
						if (cc2 == TUniChInfo::ccStarter) break;
						if (cc2 == TUniChInfo::ccAbove) { moreAbove = true; break; }
					}
					if (moreAbove)
					{
						if (cp == LatinCapitalLetterI) { dest.Add(0x69); dest.Add(0x307); continue; }
						if (cp == LatinCapitalLetterJ) { dest.Add(0x6a); dest.Add(0x307); continue; }
						if (cp == LatinCapitalLetterIWithOgonek) { dest.Add(0x12f); dest.Add(0x307); continue; }
					}
				}
				else if (cp == LatinCapitalLetterIWithGrave) { dest.Add(0x69); dest.Add(0x307); dest.Add(0x300); continue; }
				else if (cp == LatinCapitalLetterIWithAcute) { dest.Add(0x69); dest.Add(0x307); dest.Add(0x301); continue; }
				else if (cp == LatinCapitalLetterIWithTilde) { dest.Add(0x69); dest.Add(0x307); dest.Add(0x303); continue; }
			}
			if (cp == CombiningDotAbove)
			{
				bool afterSoftDotted = false;
				size_t srcIdx2 = srcIdx - 1;
				while (origSrcIdx < srcIdx2)
				{
					--srcIdx2; int cp2 = src[TVecIdx(srcIdx2)];
					int cc2 = GetCombiningClass(cp2);
					if (cc2 == TUniChInfo::ccAbove) break;
					if (cc2 == TUniChInfo::ccStarter) {
						afterSoftDotted = IsSoftDotted(cp2); break; }
				}
				if (afterSoftDotted)
				{
					Assert(lithuanian);
					if (how == ccLower) { dest.Add(0x307); continue; }
					if (how == ccUpper) continue;
					Assert(how == ccTitle);
					Assert(howHere == ccLower);
					if (seenCased && ! seenTwoCased) continue;
					dest.Add(0x307); continue;
				}
			}
		}
		else if (turkic)
		{
			if (cp == LatinCapitalLetterIWithDotAbove) {
				dest.Add(howHere == ccLower ? 0x69 : 0x130); continue; }
			else if (cp == CombiningDotAbove)
			{
				bool afterI = false;
				size_t srcIdx2 = srcIdx - 1;
				while (origSrcIdx < srcIdx2)
				{
					--srcIdx2; int cp2 = src[TVecIdx(srcIdx2)];
					if (cp2 == LatinCapitalLetterI) { afterI = true; break; }
					int cc2 = GetCombiningClass(cp2);
					if (cc2 == TUniChInfo::ccAbove || cc2 == TUniChInfo::ccStarter) break;
				}
				if (afterI) {
					if (how == ccTitle && seenCased && ! seenTwoCased) {
						IAssert(cpFirstCased == LatinCapitalLetterI);
						dest.Add(0x307); continue; }
					if (howHere != ccLower) dest.Add(0x307);
					continue; }
			}
			else if (cp == LatinCapitalLetterI)
			{
				bool beforeDot = false;
				for (size_t srcIdx2 = srcIdx; srcIdx2 < srcEnd; )
				{
					const int cp2 = src[TVecIdx(srcIdx2)]; srcIdx2++;
					if (cp2 == 0x307) { beforeDot = true; break; }
					const int cc2 = GetCombiningClass(cp2);
					if (cc2 == TUniChInfo::ccStarter || cc2 == TUniChInfo::ccAbove) break;
				}
				if (! beforeDot) {
					dest.Add(howHere == ccLower ? 0x131 : 0x49); continue; }
			}
			else if (cp == LatinSmallLetterI)
			{
				dest.Add(howHere == ccLower ? 0x69 : 0x130); continue;
			}
		}
		const TIntIntVH &specHere = (
			howHere == how ? specials :
			howHere == ccLower ? specialCasingLower :
			howHere == ccTitle ? specialCasingTitle :
			howHere == ccUpper ? specialCasingUpper : *((TIntIntVH *) 0));
		int i = specHere.GetKeyId(cp);
		if (i >= 0) { TUniCaseFolding::AppendVector(specHere[i], dest); continue; }
		i = h.GetKeyId(cp);
		if (i >= 0) {
			const TUniChInfo &ci = h[i];
			int cpNew = (
				howHere == ccLower ? ci.simpleLowerCaseMapping :
				howHere == ccUpper ? ci.simpleUpperCaseMapping :
									 ci.simpleTitleCaseMapping);
			if (cpNew < 0) cpNew = cp;
			dest.Add(cpNew); continue; }
		dest.Add(cp);
	}
}
template<typename TSrcVec, typename TDestCh>
void TUniChDb::GetSimpleCaseConverted(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
	TVec<TDestCh>& dest, const bool clrDest, const TCaseConversion how) const
{
	if (clrDest) dest.Clr();
	bool seenCased = false; size_t nextWordBoundary = srcIdx;
	for (const size_t origSrcIdx = srcIdx, srcEnd = srcIdx + srcCount; srcIdx < srcEnd; )
	{
		const int cp = src[TVecIdx(srcIdx)]; srcIdx++;
		int i = h.GetKeyId(cp); if (i < 0) { dest.Add(cp); continue; }
		const TUniChInfo &ci = h[i];
		TUniChDb::TCaseConversion howHere;
		if (how != ccTitle) howHere = how;
		else {
			if (srcIdx - 1 == nextWordBoundary) {
				seenCased = false;
				size_t next = nextWordBoundary; FindNextWordBoundary(src, origSrcIdx, srcCount, next);
				IAssert(next > nextWordBoundary); nextWordBoundary = next; }
			bool isCased = IsCased(cp);
			if (isCased && ! seenCased) { howHere = ccTitle; seenCased = true; }
			else howHere = ccLower;
		}
		int cpNew = (howHere == ccTitle ? ci.simpleTitleCaseMapping : howHere == ccUpper ? ci.simpleUpperCaseMapping : ci.simpleLowerCaseMapping);
		if (cpNew < 0) cpNew = cp;
		dest.Add(cpNew);
	}
}
template<typename TSrcVec>
void TUniChDb::ToSimpleCaseConverted(TSrcVec& src, size_t srcIdx, const size_t srcCount, const TCaseConversion how) const
{
	bool seenCased = false; size_t nextWordBoundary = srcIdx;
	for (const size_t origSrcIdx = srcIdx, srcEnd = srcIdx + srcCount; srcIdx < srcEnd; srcIdx++)
	{
		const int cp = src[TVecIdx(srcIdx)];
		int i = h.GetKeyId(cp); if (i < 0) continue;
		const TUniChInfo &ci = h[i];
		TUniChDb::TCaseConversion howHere;
		if (how != ccTitle) howHere = how;
		else {
			if (srcIdx == nextWordBoundary) {
				seenCased = false;
				size_t next = nextWordBoundary; FindNextWordBoundary(src, origSrcIdx, srcCount, next);
				IAssert(next > nextWordBoundary); nextWordBoundary = next; }
			bool isCased = IsCased(cp);
			if (isCased && ! seenCased) { howHere = ccTitle; seenCased = true; }
			else howHere = ccLower;
		}
		int cpNew = (howHere == ccTitle ? ci.simpleTitleCaseMapping : howHere == ccUpper ? ci.simpleUpperCaseMapping : ci.simpleLowerCaseMapping);
		if (cpNew >= 0) src[TVecIdx(srcIdx)] = cpNew;
	}
}
template<typename TDestCh>
void TUniChDb::AddDecomposition(const int codePoint, TVec<TDestCh>& dest, const bool compatibility) const
{
	if (HangulSBase <= codePoint && codePoint < HangulSBase + HangulSCount)
	{
		const int SIndex = codePoint - HangulSBase;
		const int L = HangulLBase + SIndex / HangulNCount;
		const int V = HangulVBase + (SIndex % HangulNCount) / HangulTCount;
		const int T = HangulTBase + (SIndex % HangulTCount);
		dest.Add(L); dest.Add(V);
		if (T != HangulTBase) dest.Add(T);
		return;
	}
	int i = h.GetKeyId(codePoint); if (i < 0) { dest.Add(codePoint); return; }
	const TUniChInfo &ci = h[i];
	int ofs = ci.decompOffset; if (ofs < 0) { dest.Add(codePoint); return; }
	if ((! compatibility) && ci.IsCompatibilityDecomposition()) { dest.Add(codePoint); return; }
	while (true) {
		int cp = decompositions[ofs++]; if (cp < 0) return;
		AddDecomposition(cp, dest, compatibility); }
}
template<typename TSrcVec, typename TDestCh>
void TUniChDb::Decompose(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, const bool compatibility, bool clrDest) const
{
	if (clrDest) dest.Clr();
	const size_t destStart = dest.Len();
	while (srcIdx < srcCount) {
		AddDecomposition(src[TVecIdx(srcIdx)], dest, compatibility); srcIdx++; }
	for (size_t destIdx = destStart, destEnd = dest.Len(); destIdx < destEnd; )
	{
		size_t j = destIdx;
		int cp = dest[TVecIdx(destIdx)]; destIdx++;
		int cpCls = GetCombiningClass(cp);
		if (cpCls == TUniChInfo::ccStarter) continue;
		while (destStart < j && GetCombiningClass(dest[TVecIdx(j - 1)]) > cpCls) {
			dest[TVecIdx(j)] = dest[TVecIdx(j - 1)]; j--; }
		dest[TVecIdx(j)] = cp;
	}
}
template<typename TSrcVec, typename TDestCh>
void TUniChDb::DecomposeAndCompose(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, bool compatibility, bool clrDest) const
{
	if (clrDest) dest.Clr();
	TIntV temp;
	Decompose(src, srcIdx, srcCount, temp, compatibility);
	Compose(temp, 0, temp.Len(), dest, clrDest);
}
template<typename TSrcVec, typename TDestCh>
void TUniChDb::Compose(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, bool clrDest) const
{
	if (clrDest) dest.Clr();
	bool lastStarterKnown = false;
	size_t lastStarterPos = size_t(-1);
	int cpLastStarter = -1;
	const size_t srcEnd = srcIdx + srcCount;
	int ccMax = -1;
	while (srcIdx < srcEnd)
	{
		const int cp = src[TVecIdx(srcIdx)]; srcIdx++;
		const int cpClass = GetCombiningClass(cp);
		if (lastStarterKnown && ccMax < cpClass)
		{
			int j = inverseDec.GetKeyId(TIntPr(cpLastStarter, cp));
			int cpCombined = -1;
			do {
				if (j >= 0) { cpCombined = inverseDec[j]; break; }
				const int LIndex = cpLastStarter - HangulLBase;
				if (0 <= LIndex && LIndex < HangulLCount) {
					const int VIndex = cp - HangulVBase;
					if (0 <= VIndex && VIndex < HangulVCount) {
						cpCombined = HangulSBase + (LIndex * HangulVCount + VIndex) * HangulTCount;
						break; } }
				const int SIndex = cpLastStarter - HangulSBase;
				if (0 <= SIndex && SIndex < HangulSCount && (SIndex % HangulTCount) == 0)
				{
					const int TIndex = cp - HangulTBase;
					if (0 <= TIndex && TIndex < HangulTCount) {
						cpCombined = cpLastStarter + TIndex;
						break; }
				}
			} while (false);
						if (cpCombined >= 0) {
				dest[TVecIdx(lastStarterPos)] = cpCombined;
				Assert(GetCombiningClass(cpCombined) == TUniChInfo::ccStarter);
								cpLastStarter = cpCombined; continue; }
		}
		if (cpClass == TUniChInfo::ccStarter) {
			lastStarterKnown = true; lastStarterPos = dest.Len(); cpLastStarter = cp; ccMax = cpClass - 1; }
		else if (cpClass > ccMax)
			ccMax = cpClass;
		dest.Add(cp);
	}
}
template<typename TSrcVec, typename TDestCh>
size_t TUniChDb::ExtractStarters(const TSrcVec& src, size_t srcIdx, const size_t srcCount,
		TVec<TDestCh>& dest, bool clrDest) const
{
	if (clrDest) dest.Clr();
	size_t retVal = 0;
	for (const size_t srcEnd = srcIdx + srcCount; srcIdx < srcEnd; srcIdx++) {
		const int cp = src[TVecIdx(srcIdx)];
		if (GetCombiningClass(cp) == TUniChInfo::ccStarter)
			{ dest.Add(cp); retVal++; } }
	return retVal;
}
inline bool AlwaysFalse()
{
	int sum = 0;
	for (int i = 0; i < 5; i++) sum += i;
	return sum > 100;
}
inline bool AlwaysTrue()
{
	int sum = 0;
	for (int i = 0; i < 5; i++) sum += i;
	return sum < 100;
}
