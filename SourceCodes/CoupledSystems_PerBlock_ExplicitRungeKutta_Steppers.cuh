#ifndef COUPLEDSYSTEM_PERBLOCK_EXPLICITRUNGEKUTTA_STEPPERS_H
#define COUPLEDSYSTEM_PERBLOCK_EXPLICITRUNGEKUTTA_STEPPERS_H


template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NCp, int CBW, int CCI, int NUP, int NSPp, int NGP, int NiGP, int NUA, int NiUA, int NSAp, int NiSAp, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_OdeFunction_WithBiConjugateGradient(\
			int        GlobalSystemID, \
			int        LocalThreadID, \
			Precision  r_Stage[UD], \
			Precision  r_State[UD], \
			Precision  s_Time, \
			Precision  r_UnitParameters[(NUP==0?1:NUP)], \
			Precision  s_SystemParameters[(NSPp==0?1:NSPp)], \
			Precision* gs_GlobalParameters, \
			int*       gs_IntegerGlobalParameters, \
			Precision  r_UnitAccessories[(NUA==0?1:NUA)], \
			int        r_IntegerUnitAccessories[(NiUA==0?1:NiUA)], \
			Precision  s_SystemAccessories[(NSAp==0?1:NSAp)], \
			int        s_IntegerSystemAccessories[(NiSAp==0?1:NiSAp)], \
			Precision  s_CouplingTerms[UPS][NCp], \
			Precision  r_CouplingFactor[NC], \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingStrength[NCp], \
			int        s_CouplingIndex[NCp], \
			int        nOddThreads, \
			int        nNonP2Warps, \
			int        nP2Warps, \
			int        WarpFactor);


template <int UPS, int NC, int NCp, int CBW, int CCI, class Precision>
__forceinline__ __device__ Precision ComputeCouplingValue_SingleSystem(\
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingTerms[UPS][NCp], \
			int        UnitID, \
			int        CouplingSerialNumber);

template <int UPS, class Precision>
__forceinline__ __device__ Precision ImplicitMatrixVectorMultiplication(\
			Precision* gs_CouplingMatrix, \
			Precision  s_Vector[UPS], \
			Precision  rD, \
			Precision  s_X1sq[UPS], \
			int        UnitID);

template <int UPS, class Precision>
__forceinline__ __device__ void ImplicitNormalTransposeMatrixVectorMultiplication(\
			Precision* gs_CouplingMatrix, \
			Precision  s_p[UPS], \
			Precision  s_pt[UPS], \
			Precision  s_q[UPS], \
			Precision& r_qt, \
			Precision* s_Temp, \
			Precision  rD, \
			Precision  s_X1sq[UPS], \
			int        UnitID, \
			int        nOddThreads, \
			int        nNonP2Warps, \
			int        nP2Warps,
			int        WarpFactor);

template <int UPS, class Precision>
__forceinline__ __device__ void DotProduct(\
			Precision* s_Vector1, \
			Precision* s_Vector2, \
			Precision* s_Temp, \
			int        UnitID, \
			int        nOddThreads, \
			int        nNonP2Warps, \
			int        nP2Warps,
			int        WarpFactor);

template <int UPS, class Precision>
__forceinline__ __device__ void ReductionAdd(\
			Precision* s_Temp, \
			int        UnitID, \
			int        nOddThreads, \
			int        nNonP2Warps, \
			int        nP2Warps,
			int        WarpFactor);

template <int UPS, class Precision>
__forceinline__ __device__ void ReductionMax(\
			Precision* s_Temp, \
			int        UnitID, \
			int        nOddThreads, \
			int        nNonP2Warps, \
			int        nP2Warps,
			int        WarpFactor);


// ----------------------------------------------------------------------------
template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NCp, int CBW, int CCI, int NUP, int NSPp, int NGP, int NiGP, int NUA, int NiUA, int NSAp, int NiSAp, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_Stepper_RKCK45(\
			const int  LocalThreadID, \
			const int  GlobalSystemID, \
			Precision  r_ActualState[UD], \
			Precision  r_NextState[UD], \
			Precision  s_ActualTime, \
			Precision  s_TimeStep, \
			int&       s_IsFinite, \
			Precision  r_Error[UD], \
			Precision  r_UnitParameters[(NUP==0?1:NUP)], \
			Precision  s_SystemParameters[(NSPp==0?1:NSPp)], \
			Precision* gs_GlobalParameters, \
			int*       gs_IntegerGlobalParameters, \
			Precision  r_UnitAccessories[(NUA==0?1:NUA)], \
			int        r_IntegerUnitAccessories[(NiUA==0?1:NiUA)], \
			Precision  s_SystemAccessories[(NSAp==0?1:NSAp)], \
			int        s_IntegerSystemAccessories[(NiSAp==0?1:NiSAp)], \
			Precision  s_CouplingTerms[UPS][NCp], \
			Precision  r_CouplingFactor[NC], \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingStrength[NCp], \
			int        s_CouplingIndex[NCp])
{
	// THREAD MANAGEMENT ------------------------------------------------------
	int nOddThreads = UPS % 32;
	int nFullWarps = UPS / 32;
	int nP2Warps;
	int WarpFactor = 0;
	
	if ( nFullWarps == 0 )
		nP2Warps = 0;
	
	if ( nFullWarps > 0  )
	{
		nP2Warps = 1;
		while ( (2*nP2Warps) <= nFullWarps )
		{
			nP2Warps = 2*nP2Warps;
			WarpFactor++;
		}
	}
	
	int nNonP2Warps = nFullWarps - nP2Warps;
	
	
	// MEMORY MANAGEMENT ------------------------------------------------------
	__shared__ Precision s_Time;
	
	Precision r_State[UD];
	
	Precision r_Stage1[UD];
	Precision r_Stage2[UD];
	Precision r_Stage3[UD];
	Precision r_Stage4[UD];
	Precision r_Stage5[UD];
	Precision r_Stage6[UD];
	
	Precision r_Uncoupled[UD];
	
	// K1 ---------------------------------------------------------------------
	if ( LocalThreadID < UPS )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i];
		
		if ( LocalThreadID == 0 )
			s_Time = s_ActualTime;
	}
	__syncthreads();
	
	if ( LocalThreadID < UPS )
	{
		CoupledSystems_PerBlock_OdeFunction_WithBiConjugateGradient<NS,UPS,UD,TPB,SPB,NC,NCp,CBW,CCI,NUP,NSPp,NGP,NiGP,NUA,NiUA,NSAp,NiSAp,NE,NDO,Precision>(\
			GlobalSystemID, \
			LocalThreadID, \
			r_Stage1, \
			r_State, \
			s_Time, \
			r_UnitParameters, \
			s_SystemParameters, \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			r_UnitAccessories, \
			r_IntegerUnitAccessories, \
			s_SystemAccessories, \
			s_IntegerSystemAccessories, \
			s_CouplingTerms, \
			r_CouplingFactor, \
			gs_CouplingMatrix, \
			s_CouplingStrength, \
			s_CouplingIndex, \
			nOddThreads, \
			nNonP2Warps, \
			nP2Warps, \
			WarpFactor);
	}
	__syncthreads();
	
	
	// K2 ---------------------------------------------------------------------
	if ( LocalThreadID < UPS )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(1.0/5.0) ) * s_TimeStep;
		
		if ( LocalThreadID == 0 )
			s_Time = s_ActualTime + static_cast<Precision>(1.0/5.0)*s_TimeStep;
	}
	__syncthreads();
	
	if ( LocalThreadID < UPS )
	{
		CoupledSystems_PerBlock_OdeFunction_WithBiConjugateGradient<NS,UPS,UD,TPB,SPB,NC,NCp,CBW,CCI,NUP,NSPp,NGP,NiGP,NUA,NiUA,NSAp,NiSAp,NE,NDO,Precision>(\
			GlobalSystemID, \
			LocalThreadID, \
			r_Stage2, \
			r_State, \
			s_Time, \
			r_UnitParameters, \
			s_SystemParameters, \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			r_UnitAccessories, \
			r_IntegerUnitAccessories, \
			s_SystemAccessories, \
			s_IntegerSystemAccessories, \
			s_CouplingTerms, \
			r_CouplingFactor, \
			gs_CouplingMatrix, \
			s_CouplingStrength, \
			s_CouplingIndex, \
			nOddThreads, \
			nNonP2Warps, \
			nP2Warps, \
			WarpFactor);
	}
	__syncthreads();
	
	
	// K3 ---------------------------------------------------------------------
	if ( LocalThreadID < UPS )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(3.0/40.0) + \
                                              r_Stage2[i] * static_cast<Precision>(9.0/40.0) ) * s_TimeStep;
		
		if ( LocalThreadID == 0 )
			s_Time = s_ActualTime + static_cast<Precision>(3.0/10.0)*s_TimeStep;
	}
	__syncthreads();
	
	if ( LocalThreadID < UPS )
	{
		CoupledSystems_PerBlock_OdeFunction_WithBiConjugateGradient<NS,UPS,UD,TPB,SPB,NC,NCp,CBW,CCI,NUP,NSPp,NGP,NiGP,NUA,NiUA,NSAp,NiSAp,NE,NDO,Precision>(\
			GlobalSystemID, \
			LocalThreadID, \
			r_Stage3, \
			r_State, \
			s_Time, \
			r_UnitParameters, \
			s_SystemParameters, \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			r_UnitAccessories, \
			r_IntegerUnitAccessories, \
			s_SystemAccessories, \
			s_IntegerSystemAccessories, \
			s_CouplingTerms, \
			r_CouplingFactor, \
			gs_CouplingMatrix, \
			s_CouplingStrength, \
			s_CouplingIndex, \
			nOddThreads, \
			nNonP2Warps, \
			nP2Warps, \
			WarpFactor);
	}
	__syncthreads();
	
	
	// K4 ---------------------------------------------------------------------
	if ( LocalThreadID < UPS )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(3.0/10.0) + \
                                              r_Stage2[i] * static_cast<Precision>(-9.0/10.0) + \
											  r_Stage3[i] * static_cast<Precision>(6.0/5.0) ) * s_TimeStep;
		
		if ( LocalThreadID == 0 )
			s_Time = s_ActualTime + static_cast<Precision>(3.0/5.0)*s_TimeStep;
	}
	__syncthreads();
	
	if ( LocalThreadID < UPS )
	{
		CoupledSystems_PerBlock_OdeFunction_WithBiConjugateGradient<NS,UPS,UD,TPB,SPB,NC,NCp,CBW,CCI,NUP,NSPp,NGP,NiGP,NUA,NiUA,NSAp,NiSAp,NE,NDO,Precision>(\
			GlobalSystemID, \
			LocalThreadID, \
			r_Stage4, \
			r_State, \
			s_Time, \
			r_UnitParameters, \
			s_SystemParameters, \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			r_UnitAccessories, \
			r_IntegerUnitAccessories, \
			s_SystemAccessories, \
			s_IntegerSystemAccessories, \
			s_CouplingTerms, \
			r_CouplingFactor, \
			gs_CouplingMatrix, \
			s_CouplingStrength, \
			s_CouplingIndex, \
			nOddThreads, \
			nNonP2Warps, \
			nP2Warps, \
			WarpFactor);
	}
	__syncthreads();
	
	
	// K5 ---------------------------------------------------------------------
	if ( LocalThreadID < UPS )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(-11.0/54.0) + \
                                              r_Stage2[i] * static_cast<Precision>(5.0/2.0) + \
											  r_Stage3[i] * static_cast<Precision>(-70.0/27.0) + \
											  r_Stage4[i] * static_cast<Precision>(35.0/27.0) ) * s_TimeStep;
		
		if ( LocalThreadID == 0 )
			s_Time = s_ActualTime + s_TimeStep;
	}
	__syncthreads();
	
	if ( LocalThreadID < UPS )
	{
		CoupledSystems_PerBlock_OdeFunction_WithBiConjugateGradient<NS,UPS,UD,TPB,SPB,NC,NCp,CBW,CCI,NUP,NSPp,NGP,NiGP,NUA,NiUA,NSAp,NiSAp,NE,NDO,Precision>(\
			GlobalSystemID, \
			LocalThreadID, \
			r_Stage5, \
			r_State, \
			s_Time, \
			r_UnitParameters, \
			s_SystemParameters, \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			r_UnitAccessories, \
			r_IntegerUnitAccessories, \
			s_SystemAccessories, \
			s_IntegerSystemAccessories, \
			s_CouplingTerms, \
			r_CouplingFactor, \
			gs_CouplingMatrix, \
			s_CouplingStrength, \
			s_CouplingIndex, \
			nOddThreads, \
			nNonP2Warps, \
			nP2Warps, \
			WarpFactor);
	}
	__syncthreads();
	
	
	// K6 ---------------------------------------------------------------------
	if ( LocalThreadID < UPS )
	{
		for (int i=0; i<UD; i++)
			r_State[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(1631.0/55296.0) + \
                                              r_Stage2[i] * static_cast<Precision>(175.0/512.0) + \
											  r_Stage3[i] * static_cast<Precision>(575.0/13824.0) + \
											  r_Stage4[i] * static_cast<Precision>(44275.0/110592.0) + \
											  r_Stage5[i] * static_cast<Precision>(253.0/4096.0) ) * s_TimeStep;
		
		if ( LocalThreadID == 0 )
			s_Time = s_ActualTime + static_cast<Precision>(7.0/8.0)*s_TimeStep;
	}
	__syncthreads();
	
	if ( LocalThreadID < UPS )
	{
		CoupledSystems_PerBlock_OdeFunction_WithBiConjugateGradient<NS,UPS,UD,TPB,SPB,NC,NCp,CBW,CCI,NUP,NSPp,NGP,NiGP,NUA,NiUA,NSAp,NiSAp,NE,NDO,Precision>(\
			GlobalSystemID, \
			LocalThreadID, \
			r_Stage6, \
			r_State, \
			s_Time, \
			r_UnitParameters, \
			s_SystemParameters, \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			r_UnitAccessories, \
			r_IntegerUnitAccessories, \
			s_SystemAccessories, \
			s_IntegerSystemAccessories, \
			s_CouplingTerms, \
			r_CouplingFactor, \
			gs_CouplingMatrix, \
			s_CouplingStrength, \
			s_CouplingIndex, \
			nOddThreads, \
			nNonP2Warps, \
			nP2Warps, \
			WarpFactor);
	}
	__syncthreads();
	
	
	// NEW STATE AND ERROR ----------------------------------------------------
	if ( LocalThreadID < UPS )
	{
		for (int i=0; i<UD; i++)
		{
			r_NextState[i] = r_ActualState[i] + ( r_Stage1[i] * static_cast<Precision>(37.0/378.0) + \
                                                  r_Stage3[i] * static_cast<Precision>(250.0/621.0) + \
											      r_Stage4[i] * static_cast<Precision>(125.0/594.0) + \
											      r_Stage6[i] * static_cast<Precision>(512.0/1771.0) ) * s_TimeStep;
			
			r_Error[i] = r_Stage1[i] * ( static_cast<Precision>(  37.0/378.0  -  2825.0/27648.0 ) ) + \
                         r_Stage3[i] * ( static_cast<Precision>( 250.0/621.0  - 18575.0/48384.0 ) ) + \
						 r_Stage4[i] * ( static_cast<Precision>( 125.0/594.0  - 13525.0/55296.0 ) ) + \
						 r_Stage5[i] * ( static_cast<Precision>(   0.0        -   277.0/14336.0 ) ) + \
						 r_Stage6[i] * ( static_cast<Precision>( 512.0/1771.0 -     1.0/4.0 ) );
			r_Error[i] = s_TimeStep*abs( r_Error[i] ) + static_cast<Precision>(1e-18);
			
			if ( ( isfinite( r_NextState[i] ) == 0 ) || ( isfinite( r_Error[i] ) == 0 ) )
				s_IsFinite = 0;
		}
	}
	__syncthreads();
}


// --- AUXILIARY FUNCTIONS ---


template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int NCp, int CBW, int CCI, int NUP, int NSPp, int NGP, int NiGP, int NUA, int NiUA, int NSAp, int NiSAp, int NE, int NDO, class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_OdeFunction_WithBiConjugateGradient(\
			int        GlobalSystemID, \
			int        LocalThreadID, \
			Precision  r_Stage[UD], \
			Precision  r_State[UD], \
			Precision  s_Time, \
			Precision  r_UnitParameters[(NUP==0?1:NUP)], \
			Precision  s_SystemParameters[(NSPp==0?1:NSPp)], \
			Precision* gs_GlobalParameters, \
			int*       gs_IntegerGlobalParameters, \
			Precision  r_UnitAccessories[(NUA==0?1:NUA)], \
			int        r_IntegerUnitAccessories[(NiUA==0?1:NiUA)], \
			Precision  s_SystemAccessories[(NSAp==0?1:NSAp)], \
			int        s_IntegerSystemAccessories[(NiSAp==0?1:NiSAp)], \
			Precision  s_CouplingTerms[UPS][NCp], \
			Precision  r_CouplingFactor[NC], \
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingStrength[NCp], \
			int        s_CouplingIndex[NCp], \
			int        nOddThreads, \
			int        nNonP2Warps, \
			int        nP2Warps, \
			int        WarpFactor)
{
	// MEMORY MANAGEMENT
	__shared__ Precision s_X1sq[UPS]; // Second power of X[1], cannot be temporary storage
	__shared__ Precision s_p[UPS];    // Can be temporary storage only during initialisation
	__shared__ Precision s_pt[UPS];   // Can be temporary storage only during initialisation
	__shared__ Precision s_r[UPS];    // Cannot be temporary storage
	__shared__ Precision s_rt[UPS];   // Cannot be temporary storage
	__shared__ Precision s_q[UPS];    // Can be temporary storage
	__shared__ Precision s_rho1;
	__shared__ Precision s_rho2;
	__shared__ Precision s_beta;
	__shared__ Precision s_alpha;
	
	Precision r_b;
	Precision r_qt;
	
	// UNCOUPLED RIGHT-HAND SIDE EVALUATION
	CoupledSystems_PerBlock_OdeFunction<Precision>(\
			GlobalSystemID, \
			LocalThreadID, \
			&r_Stage[0], \
			&r_State[0], \
			s_Time, \
			&r_UnitParameters[0], \
			&s_SystemParameters[0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[0], \
			&s_IntegerSystemAccessories[0], \
			&s_CouplingTerms[LocalThreadID][0], \
			&r_CouplingFactor[0]);
	__syncthreads();
	
	// BI-CONJUGATE GRADIENT METHOD
	s_X1sq[LocalThreadID] = r_State[0]*r_State[0]; // For implicit matrix-vector multiplication
	s_p[LocalThreadID] = r_Stage[1];               // In generaly F[1] = x = r_Stage[1], initially F[1] = x0 = xu (uncoupled) stored temporarily in s_p; 
	__syncthreads();
	
	// Initialisation
	// Computation of the right-hand side of the linear equation system(r_b)
	r_b = ComputeCouplingValue_SingleSystem<UPS,NC,NCp,CBW,CCI,Precision>(gs_CouplingMatrix, s_CouplingTerms, LocalThreadID, 0);
	r_b = r_CouplingFactor[0]*r_b + s_p[LocalThreadID];
	__syncthreads();
	
	// Computation of the residuals
	s_r[LocalThreadID] = - ImplicitMatrixVectorMultiplication<UPS,Precision>(gs_CouplingMatrix, s_p, r_CouplingFactor[0], s_X1sq, LocalThreadID);
	s_r[LocalThreadID] = s_r[LocalThreadID] + r_b;
	s_rt[LocalThreadID] = s_r[LocalThreadID];
	__syncthreads();
	
	// Start iteration
	for (int iter=1; iter<20; iter++)
	{
		// s_CouplingTerms is the temporary vector for dot product
		DotProduct<UPS,Precision>(&s_r[0], &s_rt[0], &s_CouplingTerms[0][0], LocalThreadID, nOddThreads, nNonP2Warps, nP2Warps, WarpFactor);
		if ( LocalThreadID == 0 )
			s_rho1 = s_CouplingTerms[0][0];
		__syncthreads();
		
		// Linear combination for s_p and s_pt
		if ( iter==1 )
		{
			s_p[LocalThreadID]  = s_r[LocalThreadID];
			s_pt[LocalThreadID] = s_rt[LocalThreadID];
			
			if ( MPGOS::FABS(s_rho1) < 1e-6 )
				break;
		} else
		{
			if ( LocalThreadID == 0 )
				s_beta = s_rho1/s_rho2;
			__syncthreads();
			
			s_p[LocalThreadID]  = s_r[LocalThreadID]  + s_beta*s_p[LocalThreadID];
			s_pt[LocalThreadID] = s_rt[LocalThreadID] + s_beta*s_pt[LocalThreadID];
		}
		__syncthreads();
		
		// Matrix-vector and transposed-matrix-Vector multiplication, s_CouplingTerms is the temporary vector for reduction operation
		ImplicitNormalTransposeMatrixVectorMultiplication<UPS,Precision>(gs_CouplingMatrix, s_p, s_pt, s_q, r_qt, &s_CouplingTerms[0][0], r_CouplingFactor[0], s_X1sq, LocalThreadID, nOddThreads, nNonP2Warps, nP2Warps, WarpFactor);
		
		// s_CouplingTerms is the temporary vector for dot product
		DotProduct<UPS,Precision>(&s_pt[0], &s_q[0], &s_CouplingTerms[0][0], LocalThreadID, nOddThreads, nNonP2Warps, nP2Warps, WarpFactor);
		if ( LocalThreadID == 0 )
			s_alpha = s_rho1 / s_CouplingTerms[0][0];
		__syncthreads();
		
		// Linear combination for x, s_r and s_rt
		r_Stage[1] = r_Stage[1] + s_alpha*s_p[LocalThreadID];
		s_r[LocalThreadID] = s_r[LocalThreadID] - s_alpha*s_q[LocalThreadID];
		s_rt[LocalThreadID] = s_rt[LocalThreadID] - s_alpha*r_qt;
		s_CouplingTerms[LocalThreadID][0] = MPGOS::FABS( s_r[LocalThreadID] ); // For error control
		__syncthreads();
		
		if ( LocalThreadID == 0 )
			s_rho2 = s_rho1;
		__syncthreads();
		
		// Compute norm
		ReductionMax<UPS,Precision>(&s_CouplingTerms[0][0], LocalThreadID, nOddThreads, nNonP2Warps, nP2Warps, WarpFactor);
		
		if ( s_CouplingTerms[0][0] < 1e-6 )
			break;
		__syncthreads();
	}
}


template <int UPS, int NC, int NCp, int CBW, int CCI, class Precision>
__forceinline__ __device__ Precision ComputeCouplingValue_SingleSystem(\
			Precision* gs_CouplingMatrix, \
			Precision  s_CouplingTerms[UPS][NCp], \
			int        UnitID, \
			int        CouplingSerialNumber)
{
	Precision CouplingValue = 0;
	int idx;
	
	// FULL IRREGULAR
	if ( ( CCI == 0 ) && ( CBW == 0 ) )
	{
		int MemoryShift = CouplingSerialNumber * UPS*UPS;
		for (int Col=0; Col<UPS; Col++)
		{
			idx = UnitID + Col*UPS + MemoryShift;
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[Col][CouplingSerialNumber];
		}
	}
	
	// DIAGONAL (including the circular extensions)
	if ( ( CCI == 0 ) && ( CBW > 0  ) )
	{
		int MemoryShift = CouplingSerialNumber * UPS*(2*CBW+1);
		int ShiftedUnitID;
		for (int Diag=0; Diag<(2*CBW+1); Diag++)
		{
			idx = UnitID + Diag*UPS + MemoryShift;
			
			ShiftedUnitID = UnitID + Diag - CBW;
			if ( ShiftedUnitID >= UPS )
				ShiftedUnitID = ShiftedUnitID - UPS;
			if ( ShiftedUnitID < 0 )
				ShiftedUnitID = ShiftedUnitID + UPS;
			
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[ShiftedUnitID][CouplingSerialNumber];
		}
	}
	
	// CIRCULARLY DIAGONAL
	if ( ( CCI == 1 ) && ( CBW > 0  ) )
	{
		int MemoryShift = CouplingSerialNumber * (2*CBW+1);
		int ShiftedUnitID;
		for (int Diag=0; Diag<(2*CBW+1); Diag++)
		{
			idx = Diag + MemoryShift;
			
			ShiftedUnitID = UnitID + Diag - CBW;
			if ( ShiftedUnitID >= UPS )
				ShiftedUnitID = ShiftedUnitID - UPS;
			if ( ShiftedUnitID < 0 )
				ShiftedUnitID = ShiftedUnitID + UPS;
			
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[ShiftedUnitID][CouplingSerialNumber];
		}
	}
	
	// FULL CIRCULAR
	if ( ( CCI == 1 ) && ( CBW == 0 ) )
	{
		int MemoryShift = CouplingSerialNumber * UPS;
		int ShiftedUnitID;
		for (int Diag=0; Diag<UPS; Diag++)
		{
			idx = Diag + MemoryShift;
		
			ShiftedUnitID = UnitID + Diag - CBW;
			if ( ShiftedUnitID >= UPS )
				ShiftedUnitID = ShiftedUnitID - UPS;
			if ( ShiftedUnitID < 0 )
				ShiftedUnitID = ShiftedUnitID + UPS;
			
			CouplingValue += gs_CouplingMatrix[idx]*s_CouplingTerms[ShiftedUnitID][CouplingSerialNumber];
		}
	}
	
	return CouplingValue;
}


template <int UPS, class Precision>
__forceinline__ __device__ Precision ImplicitMatrixVectorMultiplication<UPS,Precision>(\
			Precision* gs_CouplingMatrix, \
			Precision  s_Vector[UPS], \
			Precision  rD, \
			Precision  s_X1sq[UPS], \
			int        UnitID)
{
	Precision Value = 0;
	int idx;
	
	for (int Col=0; Col<UPS; Col++)
	{
		idx = UnitID + Col*UPS;
		Value += ( gs_CouplingMatrix[idx]*s_X1sq[Col] ) * s_Vector[Col];
	}
	Value = s_Vector[UnitID] + rD*Value; 
	
	return Value;
}


template <int UPS, class Precision>
__forceinline__ __device__ void ImplicitNormalTransposeMatrixVectorMultiplication<UPS,Precision>(\
			Precision* gs_CouplingMatrix, \
			Precision  s_p[UPS], \
			Precision  s_pt[UPS], \
			Precision  s_q[UPS], \
			Precision& r_qt, \
			Precision* s_Temp, \
			Precision  rD, \
			Precision  s_X1sq[UPS], \
			int        UnitID, \
			int        nOddThreads, \
			int        nNonP2Warps, \
			int        nP2Warps, \
			int        WarpFactor)
{
	s_q[UnitID] = s_p[UnitID];
	r_qt = s_pt[UnitID];
	Precision MatrixValue;
	int idx;
	
	for (int Col=0; Col<UPS; Col++)
	{
		idx = UnitID + Col*UPS;
		MatrixValue = rD*gs_CouplingMatrix[idx]*s_X1sq[Col];
		s_q[UnitID] += MatrixValue * s_p[Col];
		//MPGOS::atomicAdd(&(s_qt[Col]), MatrixValue * s_pt[UnitID]); // Kill the performance -> temporary shared + reduction
		
		s_Temp[UnitID] = MatrixValue * s_pt[UnitID];
		__syncthreads();
		
		ReductionAdd<UPS,Precision>(s_Temp, UnitID, nOddThreads, nNonP2Warps, nP2Warps, WarpFactor);
		
		if ( UnitID==Col )
			r_qt += s_Temp[0];
		__syncthreads();
	}
}


template <int UPS, class Precision>
__forceinline__ __device__ void DotProduct(\
			Precision* s_Vector1, \
			Precision* s_Vector2, \
			Precision* s_Temp, \
			int        UnitID, \
			int        nOddThreads, \
			int        nNonP2Warps, \
			int        nP2Warps, \
			int        WarpFactor)
{
	s_Temp[UnitID] = s_Vector1[UnitID] * s_Vector2[UnitID];
	__syncthreads;
	
	// Treating OddThreads
	if ( UnitID < nOddThreads )
		s_Temp[UnitID] = s_Temp[UnitID] + s_Temp[UnitID+32*(nNonP2Warps+nP2Warps)];
	__syncthreads();
	
	// Treating NonP2Warps
	if ( UnitID < 32*nNonP2Warps )
		s_Temp[UnitID] = s_Temp[UnitID] + s_Temp[UnitID+32*nP2Warps];
	__syncthreads();
	
	// Reduction of P2Warps together with the last warp
	int UnitLimit = 32*nP2Warps;
	for (int i=0; i<(WarpFactor+5); i++)
	{
		UnitLimit = UnitLimit/2;
		if ( UnitID < UnitLimit )
			s_Temp[UnitID] = s_Temp[UnitID] + s_Temp[UnitID+UnitLimit];
		
		__syncthreads();
	}
}

template <int UPS, class Precision>
__forceinline__ __device__ void ReductionAdd(\
			Precision* s_Temp, \
			int        UnitID, \
			int        nOddThreads, \
			int        nNonP2Warps, \
			int        nP2Warps, \
			int        WarpFactor)
{
	// Treating OddThreads
	if ( UnitID < nOddThreads )
		s_Temp[UnitID] = s_Temp[UnitID] + s_Temp[UnitID+32*(nNonP2Warps+nP2Warps)];
	__syncthreads();
	
	// Treating NonP2Warps
	if ( UnitID < 32*nNonP2Warps )
		s_Temp[UnitID] = s_Temp[UnitID] + s_Temp[UnitID+32*nP2Warps];
	__syncthreads();
	
	// Reduction of P2Warps together with the last warp
	int UnitLimit = 32*nP2Warps;
	for (int i=0; i<(WarpFactor+5); i++)
	{
		UnitLimit = UnitLimit/2;
		if ( UnitID < UnitLimit )
			s_Temp[UnitID] = s_Temp[UnitID] + s_Temp[UnitID+UnitLimit];
		
		__syncthreads();
	}
}

template <int UPS, class Precision>
__forceinline__ __device__ void ReductionMax(\
			Precision* s_Temp, \
			int        UnitID, \
			int        nOddThreads, \
			int        nNonP2Warps, \
			int        nP2Warps, \
			int        WarpFactor)
{
	// Treating OddThreads
	if ( UnitID < nOddThreads )
		s_Temp[UnitID] = MPGOS::FMAX(s_Temp[UnitID], s_Temp[UnitID+32*(nNonP2Warps+nP2Warps)]);
	__syncthreads();
	
	// Treating NonP2Warps
	if ( UnitID < 32*nNonP2Warps )
		s_Temp[UnitID] = MPGOS::FMAX(s_Temp[UnitID], s_Temp[UnitID+32*nP2Warps]);
	__syncthreads();
	
	// Reduction of P2Warps together with the last warp
	int UnitLimit = 32*nP2Warps;
	for (int i=0; i<(WarpFactor+5); i++)
	{
		UnitLimit = UnitLimit/2;
		if ( UnitID < UnitLimit )
			s_Temp[UnitID] = MPGOS::FMAX(s_Temp[UnitID], s_Temp[UnitID+UnitLimit]);
		
		__syncthreads();
	}
}

#endif