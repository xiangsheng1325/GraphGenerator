void TSparseColMatrix::PMultiply(const TFltVV& B, int ColId, TFltV& Result) const {
    Assert(B.GetRows() >= ColN && Result.Len() >= RowN);
    int i, j; TFlt *ResV = Result.BegI();
    for (i = 0; i < RowN; i++) ResV[i] = 0.0;
    for (j = 0; j < ColN; j++) {
        const TIntFltKdV& ColV = ColSpVV[j]; int len = ColV.Len();
        for (i = 0; i < len; i++) {
            ResV[ColV[i].Key] += ColV[i].Dat * B(j,ColId);
        }
    }
}
void TSparseColMatrix::PMultiply(const TFltV& Vec, TFltV& Result) const {
    Assert(Vec.Len() >= ColN && Result.Len() >= RowN);
    int i, j; TFlt *ResV = Result.BegI();
    for (i = 0; i < RowN; i++) ResV[i] = 0.0;
    for (j = 0; j < ColN; j++) {
        const TIntFltKdV& ColV = ColSpVV[j]; int len = ColV.Len();
        for (i = 0; i < len; i++) {
            ResV[ColV[i].Key] += ColV[i].Dat * Vec[j];
        }
    }
}
void TSparseColMatrix::PMultiplyT(const TFltVV& B, int ColId, TFltV& Result) const {
    Assert(B.GetRows() >= RowN && Result.Len() >= ColN);
    int i, j, len; TFlt *ResV = Result.BegI();
    for (j = 0; j < ColN; j++) {
        const TIntFltKdV& ColV = ColSpVV[j];
        len = ColV.Len(); ResV[j] = 0.0;
        for (i = 0; i < len; i++) {
            ResV[j] += ColV[i].Dat * B(ColV[i].Key, ColId);
        }
    }
}
void TSparseColMatrix::PMultiplyT(const TFltV& Vec, TFltV& Result) const {
    Assert(Vec.Len() >= RowN && Result.Len() >= ColN);
    int i, j, len; TFlt *VecV = Vec.BegI(), *ResV = Result.BegI();
    for (j = 0; j < ColN; j++) {
        const TIntFltKdV& ColV = ColSpVV[j];
        len = ColV.Len(); ResV[j] = 0.0;
        for (i = 0; i < len; i++) {
            ResV[j] += ColV[i].Dat * VecV[ColV[i].Key];
        }
    }
}
TSparseRowMatrix::TSparseRowMatrix(const TStr& MatlabMatrixFNm) {
   FILE *F = fopen(MatlabMatrixFNm.CStr(), "rt");  IAssert(F != NULL);
   TVec<TTriple<TInt, TInt, TSFlt> > MtxV;
   RowN = 0;  ColN = 0;
   while (! feof(F)) {
     int row=-1, col=-1; float val;
     if (fscanf(F, "%d %d %f\n", &row, &col, &val) == 3) {
       IAssert(row > 0 && col > 0);
       MtxV.Add(TTriple<TInt, TInt, TSFlt>(row, col, val));
       RowN = TMath::Mx(RowN, row);
       ColN = TMath::Mx(ColN, col);
     }
   }
   fclose(F);

   MtxV.Sort();
   RowSpVV.Gen(RowN);
   int cnt = 0;
   for (int row = 1; row <= RowN; row++) {
     while (cnt < MtxV.Len() && MtxV[cnt].Val1 == row) {
       RowSpVV[row-1].Add(TIntFltKd(MtxV[cnt].Val2-1, MtxV[cnt].Val3()));
       cnt++;
     }
   }
}
void TSparseRowMatrix::PMultiplyT(const TFltVV& B, int ColId, TFltV& Result) const {
    Assert(B.GetRows() >= RowN && Result.Len() >= ColN);
    for (int i = 0; i < ColN; i++) Result[i] = 0.0;
    for (int j = 0; j < RowN; j++) {
        const TIntFltKdV& RowV = RowSpVV[j]; int len = RowV.Len();
        for (int i = 0; i < len; i++) {
            Result[RowV[i].Key] += RowV[i].Dat * B(j,ColId);
        }
    }
}
void TSparseRowMatrix::PMultiplyT(const TFltV& Vec, TFltV& Result) const {
    Assert(Vec.Len() >= RowN && Result.Len() >= ColN);
    for (int i = 0; i < ColN; i++) Result[i] = 0.0;
    for (int j = 0; j < RowN; j++) {
        const TIntFltKdV& RowV = RowSpVV[j]; int len = RowV.Len();
        for (int i = 0; i < len; i++) {
            Result[RowV[i].Key] += RowV[i].Dat * Vec[j];
        }
    }
}
void TSparseRowMatrix::PMultiply(const TFltVV& B, int ColId, TFltV& Result) const {
    Assert(B.GetRows() >= ColN && Result.Len() >= RowN);
    for (int j = 0; j < RowN; j++) {
        const TIntFltKdV& RowV = RowSpVV[j];
        int len = RowV.Len(); Result[j] = 0.0;
        for (int i = 0; i < len; i++) {
            Result[j] += RowV[i].Dat * B(RowV[i].Key, ColId);
        }
    }
}
void TSparseRowMatrix::PMultiply(const TFltV& Vec, TFltV& Result) const {
    Assert(Vec.Len() >= ColN && Result.Len() >= RowN);
    for (int j = 0; j < RowN; j++) {
        const TIntFltKdV& RowV = RowSpVV[j];
        int len = RowV.Len(); Result[j] = 0.0;
        for (int i = 0; i < len; i++) {
            Result[j] += RowV[i].Dat * Vec[RowV[i].Key];
        }
    }
}
TFullColMatrix::TFullColMatrix(const TStr& MatlabMatrixFNm): TMatrix() {
    TLAMisc::LoadMatlabTFltVV(MatlabMatrixFNm, ColV);
    RowN=ColV[0].Len(); ColN=ColV.Len();
    for (int i = 0; i < ColN; i++) {
        IAssertR(ColV[i].Len() == RowN, TStr::Fmt("%d != %d", ColV[i].Len(), RowN));
    }
}
void TFullColMatrix::PMultiplyT(const TFltVV& B, int ColId, TFltV& Result) const {
    Assert(B.GetRows() >= RowN && Result.Len() >= ColN);
    for (int i = 0; i < ColN; i++) {
        Result[i] = TLinAlg::DotProduct(B, ColId, ColV[i]);
    }
}
void TFullColMatrix::PMultiplyT(const TFltV& Vec, TFltV& Result) const {
    Assert(Vec.Len() >= RowN && Result.Len() >= ColN);
    for (int i = 0; i < ColN; i++) {
        Result[i] = TLinAlg::DotProduct(Vec, ColV[i]);
    }
}
void TFullColMatrix::PMultiply(const TFltVV& B, int ColId, TFltV& Result) const {
    Assert(B.GetRows() >= ColN && Result.Len() >= RowN);
    for (int i = 0; i < RowN; i++) { Result[i] = 0.0; }
    for (int i = 0; i < ColN; i++) {
        TLinAlg::AddVec(B(i, ColId), ColV[i], Result, Result);
    }
}
void TFullColMatrix::PMultiply(const TFltV& Vec, TFltV& Result) const {
    Assert(Vec.Len() >= ColN && Result.Len() >= RowN);
    for (int i = 0; i < RowN; i++) { Result[i] = 0.0; }
    for (int i = 0; i < ColN; i++) {
        TLinAlg::AddVec(Vec[i], ColV[i], Result, Result);
    }
}
double TLinAlg::DotProduct(const TFltV& x, const TFltV& y) {
    IAssertR(x.Len() == y.Len(), TStr::Fmt("%d != %d", x.Len(), y.Len()));
    double result = 0.0; int Len = x.Len();
    for (int i = 0; i < Len; i++)
        result += x[i] * y[i];
    return result;
}
double TLinAlg::DotProduct(const TFltVV& X, int ColIdX, const TFltVV& Y, int ColIdY) {
    Assert(X.GetRows() == Y.GetRows());
    double result = 0.0, len = X.GetRows();
    for (int i = 0; i < len; i++)
        result = result + X(i,ColIdX) * Y(i,ColIdY);
    return result;
}
double TLinAlg::DotProduct(const TFltVV& X, int ColId, const TFltV& Vec) {
    Assert(X.GetRows() == Vec.Len());
    double result = 0.0; int Len = X.GetRows();
    for (int i = 0; i < Len; i++)
        result += X(i,ColId) * Vec[i];
    return result;
}
double TLinAlg::DotProduct(const TIntFltKdV& x, const TIntFltKdV& y) {
    const int xLen = x.Len(), yLen = y.Len();
    double Res = 0.0; int i1 = 0, i2 = 0;
    while (i1 < xLen && i2 < yLen) {
        if (x[i1].Key < y[i2].Key) i1++;
        else if (x[i1].Key > y[i2].Key) i2++;
        else { Res += x[i1].Dat * y[i2].Dat;  i1++;  i2++; }
    }
    return Res;
}
double TLinAlg::DotProduct(const TFltV& x, const TIntFltKdV& y) {
    double Res = 0.0; const int xLen = x.Len(), yLen = y.Len();
    for (int i = 0; i < yLen; i++) {
        const int key = y[i].Key;
        if (key < xLen) Res += y[i].Dat * x[key];
    }
    return Res;
}
double TLinAlg::DotProduct(const TFltVV& X, int ColId, const TIntFltKdV& y) {
    double Res = 0.0; const int n = X.GetRows(), yLen = y.Len();
    for (int i = 0; i < yLen; i++) {
        const int key = y[i].Key;
        if (key < n) Res += y[i].Dat * X(key,ColId);
    }
    return Res;
}
void TLinAlg::LinComb(const double& p, const TFltV& x,
        const double& q, const TFltV& y, TFltV& z) {
    Assert(x.Len() == y.Len() && y.Len() == z.Len());
    const int Len = x.Len();
    for (int i = 0; i < Len; i++) {
        z[i] = p * x[i] + q * y[i]; }
}
void TLinAlg::ConvexComb(const double& p, const TFltV& x, const TFltV& y, TFltV& z) {
    AssertR(0.0 <= p && p <= 1.0, TFlt::GetStr(p));
    LinComb(p, x, 1.0 - p, y, z);
}
void TLinAlg::AddVec(const double& k, const TFltV& x, const TFltV& y, TFltV& z) {
    LinComb(k, x, 1.0, y, z);
}
void TLinAlg::AddVec(const double& k, const TIntFltKdV& x, const TFltV& y, TFltV& z) {
    Assert(y.Len() == z.Len());
    z = y;

    const int xLen = x.Len(), yLen = y.Len();
    for (int i = 0; i < xLen; i++) {
        const int ii = x[i].Key;
        if (ii < yLen) {
            z[ii] = k * x[i].Dat + y[ii];
        }
    }
}
void TLinAlg::AddVec(const double& k, const TIntFltKdV& x, TFltV& y) {
    const int xLen = x.Len(), yLen = y.Len();
    for (int i = 0; i < xLen; i++) {
        const int ii = x[i].Key;
        if (ii < yLen) {
            y[ii] += k * x[i].Dat;
        }
    }
}
void TLinAlg::AddVec(double k, const TFltVV& X, int ColIdX, TFltVV& Y, int ColIdY){
    Assert(X.GetRows() == Y.GetRows());
    const int len = Y.GetRows();
    for (int i = 0; i < len; i++) {
        Y(i,ColIdY) = Y(i,ColIdY) + k * X(i, ColIdX);
    }
}
void TLinAlg::AddVec(double k, const TFltVV& X, int ColId, TFltV& Result){
    Assert(X.GetRows() == Result.Len());
    const int len = Result.Len();
    for (int i = 0; i < len; i++) {
        Result[i] = Result[i] + k * X(i, ColId);
    }
}
void TLinAlg::AddVec(const TIntFltKdV& x, const TIntFltKdV& y, TIntFltKdV& z) {
	TSparseOpsIntFlt::SparseMerge(x, y, z);
}
double TLinAlg::SumVec(const TFltV& x) {
    const int len = x.Len();
    double Res = 0.0;
    for (int i = 0; i < len; i++) {
        Res += x[i];
    }
    return Res;
}
double TLinAlg::SumVec(double k, const TFltV& x, const TFltV& y) {
    Assert(x.Len() == y.Len());
    const int len = x.Len();
    double Res = 0.0;
    for (int i = 0; i < len; i++) {
        Res += k * x[i] + y[i];
    }
    return Res;
}
double TLinAlg::EuclDist2(const TFltV& x, const TFltV& y) {
    Assert(x.Len() == y.Len());
    const int len = x.Len();
    double Res = 0.0;
    for (int i = 0; i < len; i++) {
        Res += TMath::Sqr(x[i] - y[i]);
    }
    return Res;
}
double TLinAlg::EuclDist2(const TFltPr& x, const TFltPr& y) {
    return TMath::Sqr(x.Val1 - y.Val1) + TMath::Sqr(x.Val2 - y.Val2);
}
double TLinAlg::EuclDist(const TFltV& x, const TFltV& y) {
    return sqrt(EuclDist2(x, y));
}
double TLinAlg::EuclDist(const TFltPr& x, const TFltPr& y) {
    return sqrt(EuclDist2(x, y));
}
double TLinAlg::Norm2(const TFltV& x) {
    return DotProduct(x, x);
}
double TLinAlg::Norm(const TFltV& x) {
    return sqrt(Norm2(x));
}
void TLinAlg::Normalize(TFltV& x) {
    const double xNorm = Norm(x);
    if (xNorm > 0.0) { MultiplyScalar(1/xNorm, x, x); }
}
double TLinAlg::Norm2(const TIntFltKdV& x) {
    double Result = 0;
    for (int i = 0; i < x.Len(); i++) {
        Result += TMath::Sqr(x[i].Dat);
    }
    return Result;
}
double TLinAlg::Norm(const TIntFltKdV& x) {
    return sqrt(Norm2(x));
}
void TLinAlg::Normalize(TIntFltKdV& x) {
    MultiplyScalar(1/Norm(x), x, x);
}
double TLinAlg::Norm2(const TFltVV& X, int ColId) {
    return DotProduct(X, ColId, X, ColId);
}
double TLinAlg::Norm(const TFltVV& X, int ColId) {
    return sqrt(Norm2(X, ColId));
}
double TLinAlg::NormL1(const TFltV& x) {
    double norm = 0.0; const int Len = x.Len();
    for (int i = 0; i < Len; i++)
        norm += TFlt::Abs(x[i]);
    return norm;
}
double TLinAlg::NormL1(double k, const TFltV& x, const TFltV& y) {
    Assert(x.Len() == y.Len());
    double norm = 0.0; const int len = x.Len();
    for (int i = 0; i < len; i++) {
        norm += TFlt::Abs(k * x[i] + y[i]);
    }
    return norm;
}
double TLinAlg::NormL1(const TIntFltKdV& x) {
    double norm = 0.0; const int Len = x.Len();
    for (int i = 0; i < Len; i++)
        norm += TFlt::Abs(x[i].Dat);
    return norm;
}
void TLinAlg::NormalizeL1(TFltV& x) {
    const double xNorm = NormL1(x);
    if (xNorm > 0.0) { MultiplyScalar(1/xNorm, x, x); }
}
void TLinAlg::NormalizeL1(TIntFltKdV& x) {
    const double xNorm = NormL1(x);
    if (xNorm > 0.0) { MultiplyScalar(1/xNorm, x, x); }
}
double TLinAlg::NormLinf(const TFltV& x) {
    double norm = 0.0; const int Len = x.Len();
    for (int i = 0; i < Len; i++)
        norm = TFlt::GetMx(TFlt::Abs(x[i]), norm);
    return norm;
}
double TLinAlg::NormLinf(const TIntFltKdV& x) {
    double norm = 0.0; const int Len = x.Len();
    for (int i = 0; i < Len; i++)
        norm = TFlt::GetMx(TFlt::Abs(x[i].Dat), norm);
    return norm;
}
void TLinAlg::NormalizeLinf(TFltV& x) {
    const double xNormLinf = NormLinf(x);
    if (xNormLinf > 0.0) { MultiplyScalar(1.0/xNormLinf, x, x); }
}
void TLinAlg::NormalizeLinf(TIntFltKdV& x) {
    const double xNormLInf = NormLinf(x);
    if (xNormLInf> 0.0) { MultiplyScalar(1.0/xNormLInf, x, x); }
}
void TLinAlg::MultiplyScalar(const double& k, const TFltV& x, TFltV& y) {
    Assert(x.Len() == y.Len());
    int Len = x.Len();
    for (int i = 0; i < Len; i++)
        y[i] = k * x[i];
}
void TLinAlg::MultiplyScalar(const double& k, const TIntFltKdV& x, TIntFltKdV& y) {
    Assert(x.Len() == y.Len());
    int Len = x.Len();
    for (int i = 0; i < Len; i++)
        y[i].Dat = k * x[i].Dat;
}
void TLinAlg::Multiply(const TFltVV& A, const TFltV& x, TFltV& y) {
    Assert(A.GetCols() == x.Len() && A.GetRows() == y.Len());
    int n = A.GetRows(), m = A.GetCols();
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < m; j++)
            y[i] += A(i,j) * x[j];
    }
}
void TLinAlg::Multiply(const TFltVV& A, const TFltV& x, TFltVV& C, int ColId) {
    Assert(A.GetCols() == x.Len() && A.GetRows() == C.GetRows());
    int n = A.GetRows(), m = A.GetCols();
    for (int i = 0; i < n; i++) {
        C(i,ColId) = 0.0;
        for (int j = 0; j < m; j++)
            C(i,ColId) += A(i,j) * x[j];
    }
}
void TLinAlg::Multiply(const TFltVV& A, const TFltVV& B, int ColId, TFltV& y) {
    Assert(A.GetCols() == B.GetRows() && A.GetRows() == y.Len());
    int n = A.GetRows(), m = A.GetCols();
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < m; j++)
            y[i] += A(i,j) * B(j,ColId);
    }
}
void TLinAlg::Multiply(const TFltVV& A, const TFltVV& B, int ColIdB, TFltVV& C, int ColIdC){
    Assert(A.GetCols() == B.GetRows() && A.GetRows() == C.GetRows());
    int n = A.GetRows(), m = A.GetCols();
    for (int i = 0; i < n; i++) {
        C(i,ColIdC) = 0.0;
        for (int j = 0; j < m; j++)
            C(i,ColIdC) += A(i,j) * B(j,ColIdB);
    }
}
void TLinAlg::MultiplyT(const TFltVV& A, const TFltV& x, TFltV& y) {
    Assert(A.GetRows() == x.Len() && A.GetCols() == y.Len());
    int n = A.GetCols(), m = A.GetRows();
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < m; j++)
            y[i] += A(j,i) * x[j];
    }
}
void TLinAlg::Multiply(const TFltVV& A, const TFltVV& B, TFltVV& C) {
    Assert(A.GetRows() == C.GetRows() && B.GetCols() == C.GetCols() && A.GetCols() == B.GetRows());
    int n = C.GetRows(), m = C.GetCols(), l = A.GetCols();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < l; k++)
                sum += A(i,k)*B(k,j);
            C(i,j) = sum;
        }
    }
}
void TLinAlg::Gemm(const double& Alpha, const TFltVV& A, const TFltVV& B, const double& Beta,
		const TFltVV& C, TFltVV& D, const int& TransposeFlags) {
	bool tA = (TransposeFlags & GEMM_A_T) == GEMM_A_T;
	bool tB = (TransposeFlags & GEMM_B_T) == GEMM_B_T;
	bool tC = (TransposeFlags & GEMM_C_T) == GEMM_C_T;

	int a_i = tA ? A.GetRows() : A.GetCols();
	int a_j = tA ? A.GetCols() : A.GetRows();
	int b_i = tB ? A.GetRows() : A.GetCols();
	int b_j = tB ? A.GetCols() : A.GetRows();
	int c_i = tC ? A.GetRows() : A.GetCols();
	int c_j = tC ? A.GetCols() : A.GetRows();
	int d_i = D.GetCols();
	int d_j = D.GetRows();
	

  bool Cnd = a_i == c_j && b_i == c_i && a_i == b_j && c_i == d_i && c_j == d_j;
  if (!Cnd) {
	  Assert(Cnd);
  }
	double Aij, Bij, Cij;

	for (int j = 0; j < a_j; j++) {

		for (int i = 0; i < a_i; i++) {

			double sum = 0;
			for (int k = 0; k < a_i; k++) {
				Aij = tA ? A.At(j, k) : A.At(k, j);
				Bij = tB ? B.At(k, i) : B.At(i, k);
				sum += Alpha * Aij * Bij;
			}
			Cij = tC ? C.At(i, j) : C.At(j, i);
			sum += Beta * Cij;
			D.At(i, j) = sum;
		}
	}
}
void TLinAlg::Transpose(const TFltVV& A, TFltVV& B) {
	Assert(B.GetRows() == A.GetCols() && B.GetCols() == A.GetRows());
	for (int i = 0; i < A.GetCols(); i++) {
		for (int j = 0; j < A.GetRows(); j++) {
			B.At(i, j) = A.At(j, i);
		}
	}
}
void TLinAlg::Inverse(const TFltVV& A, TFltVV& B, const TLinAlgInverseType& DecompType) {
	switch (DecompType) {
		case DECOMP_SVD:
			InverseSVD(A, B);
	}
}
void TLinAlg::InverseSVD(const TFltVV& M, TFltVV& B) {

	TFltVV U, V;
	TFltV E;
	TSvd SVD;
	U.Gen(M.GetRows(), M.GetRows());
	V.Gen(M.GetCols(), M.GetCols());

	SVD.Svd(M, U, E, V);

	for (int i = 0; i < E.Len(); i++) {
		E[i] = 1 / E[i];
	}

	for (int i = 0; i < U.GetCols(); i++) {
		for (int j = 0; j < V.GetRows(); j++) {
			double sum = 0;
			for (int k = 0; k < U.GetCols(); k++) {
				sum += E[k] * V.At(i, k) * U.At(j, k);
			}
			B.At(i, j) = sum;
		}
	}	
}
void TLinAlg::GS(TVec<TFltV>& Q) {
    IAssert(Q.Len() > 0);
    int m = Q.Len();
    for (int i = 0; i < m; i++) {
        printf("%d\r",i);
        for (int j = 0; j < i; j++) {
            double r = TLinAlg::DotProduct(Q[i], Q[j]);
            TLinAlg::AddVec(-r,Q[j],Q[i],Q[i]);
        }
        TLinAlg::Normalize(Q[i]);
    }
    printf("\n");
}
void TLinAlg::GS(TFltVV& Q) {
    int m = Q.GetCols(), n = Q.GetRows();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < i; j++) {
            double r = TLinAlg::DotProduct(Q, i, Q, j);
            TLinAlg::AddVec(-r,Q,j,Q,i);
        }
        double nr = TLinAlg::Norm(Q,i);
        for (int k = 0; k < n; k++)
            Q(k,i) = Q(k,i) / nr;
    }
}
void TLinAlg::Rotate(const double& OldX, const double& OldY, const double& Angle, double& NewX, double& NewY) {
    NewX = OldX*cos(Angle) - OldY*sin(Angle);
    NewY = OldX*sin(Angle) + OldY*cos(Angle);
}
void TLinAlg::AssertOrtogonality(const TVec<TFltV>& Vecs, const double& Threshold) {
    int m = Vecs.Len();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < i; j++) {
            double res = DotProduct(Vecs[i], Vecs[j]);
            if (TFlt::Abs(res) > Threshold)
                printf("<%d,%d> = %.5f", i,j,res);
        }
        double norm = Norm2(Vecs[i]);
        if (TFlt::Abs(norm-1) > Threshold)
            printf("||%d|| = %.5f", i, norm);
    }
}
void TLinAlg::AssertOrtogonality(const TFltVV& Vecs, const double& Threshold) {
    int m = Vecs.GetCols();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < i; j++) {
            double res = DotProduct(Vecs, i, Vecs, j);
            if (TFlt::Abs(res) > Threshold)
                printf("<%d,%d> = %.5f", i, j, res);
        }
        double norm = Norm2(Vecs, i);
        if (TFlt::Abs(norm-1) > Threshold)
            printf("||%d|| = %.5f", i, norm);
    }
    printf("\n");
}
double TNumericalStuff::sqr(double a) {
  return a == 0.0 ? 0.0 : a*a;
}
double TNumericalStuff::sign(double a, double b) {
  return b >= 0.0 ? fabs(a) : -fabs(a);
}
void TNumericalStuff::nrerror(const TStr& error_text) {
    printf("NR_ERROR: %s", error_text.CStr());
    throw new TNSException(error_text);
}
double TNumericalStuff::pythag(double a, double b) {
    double absa = fabs(a), absb = fabs(b);
    if (absa > absb)
        return absa*sqrt(1.0+sqr(absb/absa));
    else
        return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+sqr(absa/absb)));
}
void TNumericalStuff::SymetricToTridiag(TFltVV& a, int n, TFltV& d, TFltV& e) {
    int l,k,j,i;
    double scale,hh,h,g,f;
    for (i=n;i>=2;i--) {
        l=i-1;
        h=scale=0.0;
        if (l > 1) {
            for (k=1;k<=l;k++)
                scale += fabs(a(i-1,k-1).Val);
            if (scale == 0.0)
                e[i]=a(i-1,l-1);
            else {
                for (k=1;k<=l;k++) {
                    a(i-1,k-1) /= scale;
                    h += a(i-1,k-1)*a(i-1,k-1);
                }
                f=a(i-1,l-1);
                g=(f >= 0.0 ? -sqrt(h) : sqrt(h));
                IAssertR(_isnan(g) == 0, TFlt::GetStr(h));
                e[i]=scale*g;
                h -= f*g;
                a(i-1,l-1)=f-g;
                f=0.0;
                for (j=1;j<=l;j++) {

                    a(j-1,i-1)=a(i-1,j-1)/h;
                    g=0.0;
                    for (k=1;k<=j;k++)
                        g += a(j-1,k-1)*a(i-1,k-1);
                    for (k=j+1;k<=l;k++)
                        g += a(k-1,j-1)*a(i-1,k-1);
                    e[j]=g/h;
                    f += e[j]*a(i-1,j-1);
                }
                hh=f/(h+h);
                for (j=1;j<=l;j++) {
                    f=a(i-1,j-1);
                    e[j]=g=e[j]-hh*f;
                    for (k=1;k<=j;k++) {
                        a(j-1,k-1) -= (f*e[k]+g*a(i-1,k-1));
                        Assert(_isnan(static_cast<double>(a(j-1,k-1))) == 0);
                    }
                }
            }
        } else
            e[i]=a(i-1,l-1);
        d[i]=h;
    }

    d[1]=0.0;
    e[1]=0.0;


    for (i=1;i<=n;i++) {
        l=i-1;
        if (d[i]) {
            for (j=1;j<=l;j++) {
                g=0.0;
                for (k=1;k<=l;k++)
                    g += a(i-1,k-1)*a(k-1,j-1);
                for (k=1;k<=l;k++) {
                    a(k-1,j-1) -= g*a(k-1,i-1);
                    Assert(_isnan(static_cast<double>(a(k-1,j-1))) == 0);
                }
            }
        }
        d[i]=a(i-1,i-1);
        a(i-1,i-1)=1.0;
        for (j=1;j<=l;j++) a(j-1,i-1)=a(i-1,j-1)=0.0;
    }
}
void TNumericalStuff::EigSymmetricTridiag(TFltV& d, TFltV& e, int n, TFltVV& z) {
    int m,l,iter,i,k;
    double s,r,p,g,f,dd,c,b;

    for (i=2;i<=n;i++) e[i-1]=e[i];
    e[n]=0.0;
    for (l=1;l<=n;l++) {
        iter=0;
        do {

            for (m=l;m<=n-1;m++) {
        dd=TFlt::Abs(d[m])+TFlt::Abs(d[m+1]);
                if ((double)(TFlt::Abs(e[m])+dd) == dd) break;
            }
            if (m != l) {
                if (iter++ == 60) nrerror("Too many iterations in EigSymmetricTridiag");

                g=(d[l+1]-d[l])/(2.0*e[l]);
                r=pythag(g,1.0);

                g=d[m]-d[l]+e[l]/(g+sign(r,g));
                s=c=1.0;
                p=0.0;


                for (i=m-1;i>=l;i--) {
                    f=s*e[i];
                    b=c*e[i];
                    e[i+1]=(r=pythag(f,g));

                    if (r == 0.0) {
                        d[i+1] -= p;
                        e[m]=0.0;
                        break;
                    }
                    s=f/r;
                    c=g/r;
                    g=d[i+1]-p;
                    r=(d[i]-g)*s+2.0*c*b;
                    d[i+1]=g+(p=s*r);
                    g=c*r-b;

                    for (k=0;k<n;k++) {
                        f=z(k,i);
                        z(k,i)=s*z(k,i-1)+c*f;
                        z(k,i-1)=c*z(k,i-1)-s*f;
                    }
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p;
                e[l]=g;
                e[m]=0.0;
            }
        } while (m != l);
    }
}
void TNumericalStuff::CholeskyDecomposition(TFltVV& A, TFltV& p) {
  Assert(A.GetRows() == A.GetCols());
  int n = A.GetRows(); p.Reserve(n,n);
  int i,j,k;
  double sum;
  for (i=1;i<=n;i++) {
    for (j=i;j<=n;j++) {
      for (sum=A(i-1,j-1),k=i-1;k>=1;k--) sum -= A(i-1,k-1)*A(j-1,k-1);
      if (i == j) {
        if (sum <= 0.0)
          nrerror("choldc failed");
        p[i-1]=sqrt(sum);
      } else A(j-1,i-1)=sum/p[i-1];
    }
  }
}
void TNumericalStuff::CholeskySolve(const TFltVV& A, const TFltV& p, const TFltV& b, TFltV& x) {
  IAssert(A.GetRows() == A.GetCols());
  int n = A.GetRows(); x.Reserve(n,n);
  int i,k;
  double sum;

  for (i=1;i<=n;i++) {
    for (sum=b[i-1],k=i-1;k>=1;k--)
      sum -= A(i-1,k-1)*x[k-1];
    x[i-1]=sum/p[i-1];
  }

  for (i=n;i>=1;i--) {
    for (sum=x[i-1],k=i+1;k<=n;k++)
      sum -= A(k-1,i-1)*x[k-1];
    x[i-1]=sum/p[i-1];
  }
}
void TNumericalStuff::SolveSymetricSystem(TFltVV& A, const TFltV& b, TFltV& x) {
  IAssert(A.GetRows() == A.GetCols());
  TFltV p; CholeskyDecomposition(A, p);
  CholeskySolve(A, p, b, x);
}
void TNumericalStuff::InverseSubstitute(TFltVV& A, const TFltV& p) {
  IAssert(A.GetRows() == A.GetCols());
  int n = A.GetRows(); TFltV x(n);
    int i, j, k; double sum;
    for (i = 0; i < n; i++) {


        for (j = 0; j < i; j++) x[j] = 0.0;

        x[i] = 1/p[i];

        for (j = i+1; j < n; j++) {
            for (sum = 0.0, k = i; k < j; k++)
                sum -= A(j,k) * x[k];
            x[j] = sum / p[j];
        }

        for (j = n-1; j >= i; j--) {
            for (sum = x[j], k = j+1; k < n; k++)
                sum -= A(k,j)*x[k];
            x[j] = sum/p[j];
        }
        for (int j = i; j < n; j++) A(i,j) = x[j];
    }
}
void TNumericalStuff::InverseSymetric(TFltVV& A) {
    IAssert(A.GetRows() == A.GetCols());
    TFltV p;

    CholeskyDecomposition(A, p);

    InverseSubstitute(A, p);
}
void TNumericalStuff::InverseTriagonal(TFltVV& A) {
  IAssert(A.GetRows() == A.GetCols());
  int n = A.GetRows(); TFltV x(n), p(n);
    int i, j, k; double sum;

    for (i = 0; i < n; i++) {
        p[i] = A(i,i);
        for (j = i+1; j < n; j++)
            A(j,i) = A(i,j);
    }

    for (i = 0; i < n; i++) {


        for (j = n-1; j > i; j--) x[j] = 0.0;

        x[i] = 1/p[i];

        for (j = i-1; j >= 0; j--) {
            for (sum = 0.0, k = i; k > j; k--)
                sum -= A(k,j) * x[k];
            x[j] = sum / p[j];
        }
        for (int j = 0; j <= i; j++) A(j,i) = x[j];
    }
}
void TNumericalStuff::LUDecomposition(TFltVV& A, TIntV& indx, double& d) {
  Assert(A.GetRows() == A.GetCols());
  int n = A.GetRows(); indx.Reserve(n,n);
    int i=0,imax=0,j=0,k=0;
    double big,dum,sum,temp;
    TFltV vv(n);
    d=1.0;

    for (i=1;i<=n;i++) {
        big=0.0;
        for (j=1;j<=n;j++)
            if ((temp=TFlt::Abs(A(i-1,j-1))) > big) big=temp;
        if (big == 0.0) nrerror("Singular matrix in routine LUDecomposition");
        vv[i-1]=1.0/big;
    }
    for (j=1;j<=n;j++) {
        for (i=1;i<j;i++) {
            sum=A(i-1,j-1);
            for (k=1;k<i;k++) sum -= A(i-1,k-1)*A(k-1,j-1);
            A(i-1,j-1)=sum;
        }
        big=0.0;
        for (i=j;i<=n;i++) {
            sum=A(i-1,j-1);
            for (k=1;k<j;k++)
                sum -= A(i-1,k-1)*A(k-1,j-1);
            A(i-1,j-1)=sum;

            if ((dum=vv[i-1] * TFlt::Abs(sum)) >= big) {
                big=dum;
                imax=i;
            }
        }

        if (j != imax) {

            for (k=1;k<=n;k++) {
                dum=A(imax-1,k-1);
            A(imax-1,k-1)=A(j-1,k-1);
            A(j-1,k-1)=dum;
            }

            d = -d;
            vv[imax-1]=vv[j-1];
        }
        indx[j-1]=imax;



        if (A(j-1,j-1) == 0.0) A(j-1,j-1)=1e-20;

        if (j != n) {
            dum=1.0/(A(j-1,j-1));
            for (i=j+1;i<=n;i++) A(i-1,j-1) *= dum;
        }
    }
}
void TNumericalStuff::LUSolve(const TFltVV& A, const TIntV& indx, TFltV& b) {
  Assert(A.GetRows() == A.GetCols());
  int n = A.GetRows();
    int i,ii=0,ip,j;
    double sum;
    for (i=1;i<=n;i++) {
        ip=indx[i-1];
        sum=b[ip-1];
        b[ip-1]=b[i-1];
        if (ii)
            for (j=ii;j<=i-1;j++) sum -= A(i-1,j-1)*b[j-1];
        else if (sum) ii=i;b[i-1]=sum;
    }
    for (i=n;i>=1;i--) {
        sum=b[i-1];
        for (j=i+1;j<=n;j++) sum -= A(i-1,j-1)*b[j-1];
        b[i-1]=sum/A(i-1,i-1);
    }
}
void TNumericalStuff::SolveLinearSystem(TFltVV& A, const TFltV& b, TFltV& x) {
    TIntV indx; double d;
    LUDecomposition(A, indx, d);
    x = b;
    LUSolve(A, indx, x);
}
void TSparseSVD::MultiplyATA(const TMatrix& Matrix,
        const TFltVV& Vec, int ColId, TFltV& Result) {
    TFltV tmp(Matrix.GetRows());

    Matrix.Multiply(Vec, ColId, tmp);

    Matrix.MultiplyT(tmp, Result);
}
void TSparseSVD::MultiplyATA(const TMatrix& Matrix,
        const TFltV& Vec, TFltV& Result) {
    TFltV tmp(Matrix.GetRows());

    Matrix.Multiply(Vec, tmp);

    Matrix.MultiplyT(tmp, Result);
}
void TSparseSVD::OrtoIterSVD(const TMatrix& Matrix,
        int NumSV, int IterN, TFltV& SgnValV) {
    int i, j, k;
    int N = Matrix.GetCols(), M = NumSV;
    TFltVV Q(N, M);

    TRnd rnd;
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++)
            Q(i,j) = rnd.GetUniDev();
    }
    TFltV tmp(N);
    for (int IterC = 0; IterC < IterN; IterC++) {
        printf("%d..", IterC);

        TLinAlg::GS(Q);

        for (int ColId = 0; ColId < M; ColId++) {
            MultiplyATA(Matrix, Q, ColId, tmp);
            for (k = 0; k < N; k++) Q(k,ColId) = tmp[k];
        }
    }
    SgnValV.Reserve(NumSV,0);
    for (i = 0; i < NumSV; i++)
        SgnValV.Add(sqrt(TLinAlg::Norm(Q,i)));
    TLinAlg::GS(Q);
}
void TSparseSVD::SimpleLanczos(const TMatrix& Matrix,
        const int& NumEig, TFltV& EigValV,
        const bool& DoLocalReortoP, const bool& SvdMatrixProductP) {
    if (SvdMatrixProductP) {

        IAssert(Matrix.GetRows() >= Matrix.GetCols());
    } else {
        IAssert(Matrix.GetRows() == Matrix.GetCols());
    }
    const int N = Matrix.GetCols();
    TFltV r(N), v0(N), v1(N);
    TFltV alpha(NumEig, 0), beta(NumEig, 0);
    printf("Calculating %d eigen-values of %d x %d matrix\n", NumEig, N, N);


    for (int i = 0; i < N; i++) {
        r[i] = 1/sqrt((double)N);
        v0[i] = v1[i] = 0.0;
    }
    beta.Add(TLinAlg::Norm(r));
    for (int j = 0; j < NumEig; j++) {
        printf("%d\r", j+1);

        v0 = v1;

        TLinAlg::MultiplyScalar(1/beta[j], r, v1);

        if (SvdMatrixProductP) {

            MultiplyATA(Matrix, v1, r);
        } else {

            Matrix.Multiply(v1, r);
        }

        TLinAlg::AddVec(-beta[j], v0, r, r);

        alpha.Add(TLinAlg::DotProduct(v1, r));

        TLinAlg::AddVec(-alpha[j], v1, r, r);

        if (DoLocalReortoP) { }

        beta.Add(TLinAlg::Norm(r));


    }
    printf("\n");

    TFltV d(NumEig + 1), e(NumEig + 1);
    d[1] = alpha[0]; d[0] = e[0] = e[1] = 0.0;
    for (int i = 1; i < NumEig; i++) {
        d[i+1] = alpha[i]; e[i+1] = beta[i]; }

    TFltVV S(NumEig+1,NumEig+1);
    TLAMisc::FillIdentity(S);
    TNumericalStuff::EigSymmetricTridiag(d, e, NumEig, S);


    TFltKdV AllEigValV(NumEig, 0);
    for (int i = 1; i <= NumEig; i++) {
        const double ResidualNorm = TFlt::Abs(S(i-1, NumEig-1) * beta.Last());
        if (ResidualNorm < 1e-5)
            AllEigValV.Add(TFltKd(TFlt::Abs(d[i]), d[i]));
    }

    AllEigValV.Sort(false); EigValV.Gen(NumEig, 0);
    for (int i = 0; i < AllEigValV.Len(); i++) {
        if (i == 0 || (TFlt::Abs(AllEigValV[i].Dat/AllEigValV[i-1].Dat) < 0.9999))
            EigValV.Add(AllEigValV[i].Dat);
    }
}
void TSparseSVD::Lanczos(const TMatrix& Matrix, int NumEig,
        int Iters, const TSpSVDReOrtoType& ReOrtoType,
        TFltV& EigValV, TFltVV& EigVecVV, const bool& SvdMatrixProductP) {
  if (SvdMatrixProductP) {

    IAssert(Matrix.GetRows() >= Matrix.GetCols());
  } else {
    IAssert(Matrix.GetRows() == Matrix.GetCols());
  }
  IAssertR(NumEig <= Iters, TStr::Fmt("%d <= %d", NumEig, Iters));

  int i, N = Matrix.GetCols(), K = 0;
  double t = 0.0, eps = 1e-6;

  TFltVV Q(N, Iters);
  double tmp = 1/sqrt((double)N);
  for (i = 0; i < N; i++) {
    Q(i,0) = tmp;
  }

  TVec<TFltV> ConvgQV(Iters);
  TIntV CountConvgV(Iters);
  for (i = 0; i < Iters; i++) CountConvgV[i] = 0;


  TFltV d(Iters+1), e(Iters+1);


  TFltVV V(Iters, Iters);

  TFltV z(N), bb(Iters), aa(Iters), y(N);

  if (SvdMatrixProductP) {

    MultiplyATA(Matrix, Q, 0, z);
  } else {

    Matrix.Multiply(Q, 0, z);
  }
  for (int j = 0; j < (Iters-1); j++) {



    aa[j] = TLinAlg::DotProduct(Q, j, z);

    TLinAlg::AddVec(-aa[j], Q, j, z);
    if (j > 0) {

      TLinAlg::AddVec(-bb[j-1], Q, j-1, z);

      if (ReOrtoType == ssotSelective || ReOrtoType == ssotFull) {
        for (i = 0; i <= j; i++) {

          if ((ReOrtoType == ssotFull) ||
            (bb[j-1] * TFlt::Abs(V(K-1, i)) < eps * t)) {
            ConvgQV[i].Reserve(N,N); CountConvgV[i]++;
            TFltV& vec = ConvgQV[i];

            for (int k = 0; k < N; k++) {
              vec[k] = 0.0;
              for (int l = 0; l < K; l++) {
                vec[k] += Q(k,l) * V(l,i);
              }
            }
            TLinAlg::AddVec(-TLinAlg::DotProduct(ConvgQV[i], z), ConvgQV[i], z ,z);
          }
        }
      }
    }

    bb[j] = TLinAlg::Norm(z);
    if (!(bb[j] > 1e-10)) {
      printf("Rank of matrix is only %d\n", j+2);
      printf("Last singular value is %g\n", bb[j].Val);
      break;
    }
    for (i = 0; i < N; i++) {
      Q(i, j+1) = z[i] / bb[j];
    }

    if (SvdMatrixProductP) {

      MultiplyATA(Matrix, Q, j+1, z);
    } else {

      Matrix.Multiply(Q, j+1, z);
    }

    K = j + 2;

    for (i = 1; i < K; i++) d[i] = aa[i-1];
    d[K] = TLinAlg::DotProduct(Q, K-1, z);

    e[1] = 0.0;
    for (i = 2; i <= K; i++) e[i] = bb[i-2];

    t = TFlt::GetMx(TFlt::Abs(d[1]) + TFlt::Abs(e[2]), TFlt::Abs(e[K]) + TFlt::Abs(d[K]));
    for (i = 2; i < K; i++) {
      t = TFlt::GetMx(t, TFlt::Abs(e[i]) + TFlt::Abs(d[i]) + TFlt::Abs(e[i+1]));
    }


    for (i = 0; i < K; i++) {
      for (int k = 0; k < K; k++) {
        V(i,k) = 0.0;
      }
      V(i,i) = 1.0;
    }

    TNumericalStuff::EigSymmetricTridiag(d, e, K, V);
  }


  TFltIntKdV sv(K);
  for (i = 0; i < K; i++) {
    sv[i].Key = TFlt::Abs(d[i+1]);
    sv[i].Dat = i;
  }
  sv.Sort(false);
  TFltV uu(Matrix.GetRows());
  const int FinalNumEig = TInt::GetMn(NumEig, K);
  EigValV.Reserve(FinalNumEig,0);
  EigVecVV.Gen(Matrix.GetCols(), FinalNumEig);
  for (i = 0; i < FinalNumEig; i++) {

    int ii = sv[i].Dat;
    double sigma = d[ii+1].Val;

    EigValV.Add(sigma);

    TLinAlg::Multiply(Q, V, ii, EigVecVV, i);
  }

}
void TSparseSVD::Lanczos2(const TMatrix& Matrix, int MaxNumEig,
    int MaxSecs, const TSpSVDReOrtoType& ReOrtoType,
    TFltV& EigValV, TFltVV& EigVecVV, const bool& SvdMatrixProductP) {
  if (SvdMatrixProductP) {

    IAssert(Matrix.GetRows() >= Matrix.GetCols());
  } else {
    IAssert(Matrix.GetRows() == Matrix.GetCols());
  }


  int i, N = Matrix.GetCols(), K = 0;
  double t = 0.0, eps = 1e-6;

  TFltVV Q(N, MaxNumEig);
  double tmp = 1/sqrt((double)N);
  for (i = 0; i < N; i++)
      Q(i,0) = tmp;

  TVec<TFltV> ConvgQV(MaxNumEig);
  TIntV CountConvgV(MaxNumEig);
  for (i = 0; i < MaxNumEig; i++) CountConvgV[i] = 0;


  TFltV d(MaxNumEig+1), e(MaxNumEig+1);


  TFltVV V(MaxNumEig, MaxNumEig);

  TFltV z(N), bb(MaxNumEig), aa(MaxNumEig), y(N);

  if (SvdMatrixProductP) {

      MultiplyATA(Matrix, Q, 0, z);
  } else {

      Matrix.Multiply(Q, 0, z);
  }
  TExeTm ExeTm;
  for (int j = 0; j < (MaxNumEig-1); j++) {
    printf("%d [%s]..\r",j+2, ExeTm.GetStr());
    if (ExeTm.GetSecs() > MaxSecs) { break; }


    aa[j] = TLinAlg::DotProduct(Q, j, z);

    TLinAlg::AddVec(-aa[j], Q, j, z);
    if (j > 0) {

        TLinAlg::AddVec(-bb[j-1], Q, j-1, z);

        if (ReOrtoType == ssotSelective || ReOrtoType == ssotFull) {
            for (i = 0; i <= j; i++) {

                if ((ReOrtoType == ssotFull) ||
                    (bb[j-1] * TFlt::Abs(V(K-1, i)) < eps * t)) {
                    ConvgQV[i].Reserve(N,N); CountConvgV[i]++;
                    TFltV& vec = ConvgQV[i];

                    for (int k = 0; k < N; k++) {
                        vec[k] = 0.0;
                        for (int l = 0; l < K; l++)
                            vec[k] += Q(k,l) * V(l,i);
                    }
                    TLinAlg::AddVec(-TLinAlg::DotProduct(ConvgQV[i], z), ConvgQV[i], z ,z);
                }
            }
        }
    }

    bb[j] = TLinAlg::Norm(z);
    if (!(bb[j] > 1e-10)) {
      printf("Rank of matrix is only %d\n", j+2);
      printf("Last singular value is %g\n", bb[j].Val);
      break;
    }
    for (i = 0; i < N; i++)
        Q(i, j+1) = z[i] / bb[j];

    if (SvdMatrixProductP) {

        MultiplyATA(Matrix, Q, j+1, z);
    } else {

        Matrix.Multiply(Q, j+1, z);
    }

    K = j + 2;

    for (i = 1; i < K; i++) d[i] = aa[i-1];
    d[K] = TLinAlg::DotProduct(Q, K-1, z);

    e[1] = 0.0;
    for (i = 2; i <= K; i++) e[i] = bb[i-2];

    t = TFlt::GetMx(TFlt::Abs(d[1]) + TFlt::Abs(e[2]), TFlt::Abs(e[K]) + TFlt::Abs(d[K]));
    for (i = 2; i < K; i++)
        t = TFlt::GetMx(t, TFlt::Abs(e[i]) + TFlt::Abs(d[i]) + TFlt::Abs(e[i+1]));


    for (i = 0; i < K; i++) {
        for (int k = 0; k < K; k++)
            V(i,k) = 0.0;
        V(i,i) = 1.0;
    }

    TNumericalStuff::EigSymmetricTridiag(d, e, K, V);
  }
  printf("... calc %d.", K);

  TFltIntKdV sv(K);
  for (i = 0; i < K; i++) {
    sv[i].Key = TFlt::Abs(d[i+1]);
    sv[i].Dat = i;
  }
  sv.Sort(false);
  TFltV uu(Matrix.GetRows());
  const int FinalNumEig = K;
  EigValV.Reserve(FinalNumEig,0);
  EigVecVV.Gen(Matrix.GetCols(), FinalNumEig);
  for (i = 0; i < FinalNumEig; i++) {

    int ii = sv[i].Dat;
    double sigma = d[ii+1].Val;

    EigValV.Add(sigma);

    TLinAlg::Multiply(Q, V, ii, EigVecVV, i);
  }
  printf("  done\n");
}
void TSparseSVD::SimpleLanczosSVD(const TMatrix& Matrix,
        const int& CalcSV, TFltV& SngValV, const bool& DoLocalReorto) {
    SimpleLanczos(Matrix, CalcSV, SngValV, DoLocalReorto, true);
    for (int SngValN = 0; SngValN < SngValV.Len(); SngValN++) {

      if (SngValV[SngValN] < 0.0) {
        printf("bad sng val: %d %g\n", SngValN, SngValV[SngValN]());
        SngValV[SngValN] = 0;
      }
      SngValV[SngValN] = sqrt(SngValV[SngValN].Val);
    }
}
void TSparseSVD::LanczosSVD(const TMatrix& Matrix, int NumSV,
        int Iters, const TSpSVDReOrtoType& ReOrtoType,
        TFltV& SgnValV, TFltVV& LeftSgnVecVV, TFltVV& RightSgnVecVV) {

    Lanczos(Matrix, NumSV, Iters, ReOrtoType, SgnValV, RightSgnVecVV, true);

    const int FinalNumSV = SgnValV.Len();
    LeftSgnVecVV.Gen(Matrix.GetRows(), FinalNumSV);
    TFltV LeftSgnVecV(Matrix.GetRows());
    for (int i = 0; i < FinalNumSV; i++) {
        if (SgnValV[i].Val < 0.0) { SgnValV[i] = 0.0; }
        const double SgnVal = sqrt(SgnValV[i]);
        SgnValV[i] = SgnVal;

        Matrix.Multiply(RightSgnVecVV, i, LeftSgnVecV);
        for (int j = 0; j < LeftSgnVecV.Len(); j++) {
            LeftSgnVecVV(j,i) = LeftSgnVecV[j] / SgnVal; }
    }

}
void TSparseSVD::Project(const TIntFltKdV& Vec, const TFltVV& U, TFltV& ProjVec) {
    const int m = U.GetCols();
    ProjVec.Gen(m, 0);
    for (int j = 0; j < m; j++) {
        double x = 0.0;
        for (int i = 0; i < Vec.Len(); i++)
            x += U(Vec[i].Key, j) * Vec[i].Dat;
        ProjVec.Add(x);
    }
}
double TSigmoid::EvaluateFit(const TFltIntKdV& data, const double A, const double B)
{
  double J = 0.0;
  for (int i = 0; i < data.Len(); i++)
  {
    double zi = data[i].Key; int yi = data[i].Dat;
    double e = exp(-A * zi + B);
    double denum = 1.0 + e;
    double prob = (yi > 0) ? (1.0 / denum) : (e / denum);
    J -= log(prob < 1e-20 ? 1e-20 : prob);
  }
  return J;
}
void TSigmoid::EvaluateFit(const TFltIntKdV& data, const double A, const double B, double& J, double& JA, double& JB)
{




  J = 0.0; double sum_all_PyNeg = 0.0, sum_all_ziPyNeg = 0.0, sum_yNeg_zi = 0.0, sum_yNeg_1 = 0.0;
  for (int i = 0; i < data.Len(); i++)
  {
    double zi = data[i].Key; int yi = data[i].Dat;
    double e = exp(-A * zi + B);
    double denum = 1.0 + e;
    double prob = (yi > 0) ? (1.0 / denum) : (e / denum);
    J -= log(prob < 1e-20 ? 1e-20 : prob);
    sum_all_PyNeg += e / denum;
    sum_all_ziPyNeg += zi * e / denum;
    if (yi < 0) { sum_yNeg_zi += zi; sum_yNeg_1 += 1; }
  }
  JA = -sum_all_ziPyNeg +     sum_yNeg_zi;
  JB =  sum_all_PyNeg   -     sum_yNeg_1;
}
void TSigmoid::EvaluateFit(const TFltIntKdV& data, const double A, const double B, const double U,
                           const double V, const double lambda, double& J, double& JJ, double& JJJ)
{





  J = 0.0; JJ = 0.0; JJJ = 0.0;
  for (int i = 0; i < data.Len(); i++)
  {
    double zi = data[i].Key; int yi = data[i].Dat;
    double e = exp(-A * zi + B);
    double denum = 1.0 + e;
    double prob = (yi > 0) ? (1.0 / denum) : (e / denum);
    J -= log(prob < 1e-20 ? 1e-20 : prob);
    double VU = V - U * zi;
    JJ += VU * (e / denum); if (yi < 0) JJ -= VU;
    JJJ += VU * VU * e / denum / denum;
  }
}
TSigmoid::TSigmoid(const TFltIntKdV& data) {









  double minProj = data[0].Key, maxProj = data[0].Key;
  {for (int i = 1; i < data.Len(); i++) {
    double zi = data[i].Key; if (zi < minProj) minProj = zi; if (zi > maxProj) maxProj = zi; }}

  A = 1.0; B = 0.5 * (minProj + maxProj);
  double bestJ = 0.0, bestA = 0.0, bestB = 0.0, lambda = 1.0;
  for (int nIter = 0; nIter < 50; nIter++)
  {
    double J, JA, JB; TSigmoid::EvaluateFit(data, A, B, J, JA, JB);
    if (nIter == 0 || J < bestJ) { bestJ = J; bestA = A; bestB = B; }


        double norm = TMath::Sqr(JA) + TMath::Sqr(JB);
    if (norm < 1e-10) break;
    const int cl = -1;
    double Jc = TSigmoid::EvaluateFit(data, A + cl * lambda * JA / norm, B + cl * lambda * JB / norm);

    if (Jc > J) {
      while (lambda > 1e-5) {
        lambda = 0.5 * lambda;
        Jc = TSigmoid::EvaluateFit(data, A + cl * lambda * JA / norm, B + cl * lambda * JB / norm);

      } }
    else if (Jc < J) {
      while (lambda < 1e5) {
        double lambda2 = 2 * lambda;
        double Jc2 = TSigmoid::EvaluateFit(data, A + cl * lambda2 * JA / norm, B + cl * lambda2 * JB / norm);

        if (Jc2 > Jc) break;
        lambda = lambda2; Jc = Jc2; } }
    if (Jc >= J) break;
    A += cl * lambda * JA / norm; B += cl * lambda * JB / norm;

  }
  A = bestA; B = bestB;
}
void TLAMisc::SaveCsvTFltV(const TFltV& Vec, TSOut& SOut) {
    for (int ValN = 0; ValN < Vec.Len(); ValN++) {
        SOut.PutFlt(Vec[ValN]); SOut.PutCh(',');
    }
    SOut.PutLn();
}
void TLAMisc::SaveMatlabTFltIntKdV(const TIntFltKdV& SpV, const int& ColN, TSOut& SOut) {
    const int Len = SpV.Len();
    for (int ValN = 0; ValN < Len; ValN++) {
        SOut.PutStrLn(TStr::Fmt("%d %d %g", SpV[ValN].Key+1, ColN+1, SpV[ValN].Dat()));
    }
}
void TLAMisc::SaveMatlabTFltV(const TFltV& m, const TStr& FName) {
    PSOut out = TFOut::New(FName);
    const int RowN = m.Len();
    for (int RowId = 0; RowId < RowN; RowId++) {
        out->PutStr(TFlt::GetStr(m[RowId], 20, 18));
        out->PutCh('\n');
    }
    out->Flush();
}
void TLAMisc::SaveMatlabTIntV(const TIntV& m, const TStr& FName) {
    PSOut out = TFOut::New(FName);
    const int RowN = m.Len();
    for (int RowId = 0; RowId < RowN; RowId++) {
        out->PutInt(m[RowId]);
        out->PutCh('\n');
    }
    out->Flush();
}
void TLAMisc::SaveMatlabTFltVVCol(const TFltVV& m, int ColId, const TStr& FName) {
    PSOut out = TFOut::New(FName);
    const int RowN = m.GetRows();
    for (int RowId = 0; RowId < RowN; RowId++) {
        out->PutStr(TFlt::GetStr(m(RowId,ColId), 20, 18));
        out->PutCh('\n');
    }
    out->Flush();
}
void TLAMisc::SaveMatlabTFltVV(const TFltVV& m, const TStr& FName) {
    PSOut out = TFOut::New(FName);
    const int RowN = m.GetRows();
    const int ColN = m.GetCols();
    for (int RowId = 0; RowId < RowN; RowId++) {
        for (int ColId = 0; ColId < ColN; ColId++) {
            out->PutStr(TFlt::GetStr(m(RowId,ColId), 20, 18));
            out->PutCh(' ');
        }
        out->PutCh('\n');
    }
    out->Flush();
}
void TLAMisc::SaveMatlabTFltVVMjrSubMtrx(const TFltVV& m,
        int RowN, int ColN, const TStr& FName) {
    PSOut out = TFOut::New(FName);
    for (int RowId = 0; RowId < RowN; RowId++) {
        for (int ColId = 0; ColId < ColN; ColId++) {
            out->PutStr(TFlt::GetStr(m(RowId,ColId), 20, 18)); out->PutCh(' ');
        }
        out->PutCh('\n');
    }
    out->Flush();
}
void TLAMisc::LoadMatlabTFltVV(const TStr& FNm, TVec<TFltV>& ColV) {
    PSIn SIn = TFIn::New(FNm);
    TILx Lx(SIn, TFSet()|iloRetEoln|iloSigNum|iloExcept);
    int Row = 0, Col = 0; ColV.Clr();
    Lx.GetSym(syFlt, syEof, syEoln);

    while (Lx.Sym != syEof) {
        if (Lx.Sym == syFlt) {
            if (ColV.Len() > Col) {
                IAssert(ColV[Col].Len() == Row);
                ColV[Col].Add(Lx.Flt);
            } else {
                IAssert(Row == 0);
                ColV.Add(TFltV::GetV(Lx.Flt));
            }
            Col++;
        } else if (Lx.Sym == syEoln) {
            IAssert(Col == ColV.Len());
            Col = 0; Row++;
            if (Row%100 == 0) {

            }
        } else {
            Fail;
        }
        Lx.GetSym(syFlt, syEof, syEoln);
    }

    IAssert(Col == ColV.Len() || Col == 0);
}
void TLAMisc::LoadMatlabTFltVV(const TStr& FNm, TFltVV& MatrixVV) {
    TVec<TFltV> ColV; LoadMatlabTFltVV(FNm, ColV);
    if (ColV.Empty()) { MatrixVV.Clr(); return; }
    const int Rows = ColV[0].Len(), Cols = ColV.Len();
    MatrixVV.Gen(Rows, Cols);
    for (int RowN = 0; RowN < Rows; RowN++) {
        for (int ColN = 0; ColN < Cols; ColN++) {
            MatrixVV(RowN, ColN) = ColV[ColN][RowN];
        }
    }
}
void TLAMisc::PrintTFltV(const TFltV& Vec, const TStr& VecNm) {
    printf("%s = [", VecNm.CStr());
    for (int i = 0; i < Vec.Len(); i++) {
        printf("%.5f", Vec[i]());
		if (i < Vec.Len() - 1) { printf(", "); }
    }
    printf("]\n");
}
void TLAMisc::PrintTFltVV(const TFltVV& A, const TStr& MatrixNm) {
    printf("%s = [\n", MatrixNm.CStr());
	for (int j = 0; j < A.GetRows(); j++) {
		for (int i = 0; i < A.GetCols(); i++) {
			printf("%f\t", A.At(i, j).Val);
		}
		printf("\n");
	}
    printf("]\n");
}
void TLAMisc::PrintTIntV(const TIntV& Vec, const TStr& VecNm) {
    printf("%s = [", VecNm.CStr());
    for (int i = 0; i < Vec.Len(); i++) {
        printf("%d", Vec[i]());
        if (i < Vec.Len() - 1) printf(", ");
    }
    printf("]\n");
}
void TLAMisc::FillRnd(TFltV& Vec, TRnd& Rnd) {
    int Len = Vec.Len();
    for (int i = 0; i < Len; i++)
        Vec[i] = Rnd.GetNrmDev();
}
void TLAMisc::FillIdentity(TFltVV& M) {
    IAssert(M.GetRows() == M.GetCols());
    int Len = M.GetRows();
    for (int i = 0; i < Len; i++) {
        for (int j = 0; j < Len; j++) M(i,j) = 0.0;
        M(i,i) = 1.0;
    }
}
void TLAMisc::FillIdentity(TFltVV& M, const double& Elt) {
    IAssert(M.GetRows() == M.GetCols());
    int Len = M.GetRows();
    for (int i = 0; i < Len; i++) {
        for (int j = 0; j < Len; j++) M(i,j) = 0.0;
        M(i,i) = Elt;
    }
}
int TLAMisc::SumVec(const TIntV& Vec) {
    const int Len = Vec.Len();
    int res = 0;
    for (int i = 0; i < Len; i++)
        res += Vec[i];
    return res;
}
double TLAMisc::SumVec(const TFltV& Vec) {
    const int Len = Vec.Len();
    double res = 0.0;
    for (int i = 0; i < Len; i++)
        res += Vec[i];
    return res;
}
void TLAMisc::ToSpVec(const TFltV& Vec, TIntFltKdV& SpVec,
        const double& CutSumPrc) {

    IAssert(0.0 <= CutSumPrc && CutSumPrc <= 1.0);
    const int Elts = Vec.Len();
    double EltSum = 0.0;
    for (int EltN = 0; EltN < Elts; EltN++) {
        EltSum += TFlt::Abs(Vec[EltN]); }
    const double MnEltVal = CutSumPrc * EltSum;

    SpVec.Clr();
    for (int EltN = 0; EltN < Elts; EltN++) {
        if (TFlt::Abs(Vec[EltN]) > MnEltVal) {
            SpVec.Add(TIntFltKd(EltN, Vec[EltN]));
        }
    }
    SpVec.Pack();
}
void TLAMisc::ToVec(const TIntFltKdV& SpVec, TFltV& Vec, const int& VecLen) {
    Vec.Gen(VecLen); Vec.PutAll(0.0);
    int Elts = SpVec.Len();
    for (int EltN = 0; EltN < Elts; EltN++) {
        if (SpVec[EltN].Key < VecLen) {
            Vec[SpVec[EltN].Key] = SpVec[EltN].Dat;
        }
    }
}
