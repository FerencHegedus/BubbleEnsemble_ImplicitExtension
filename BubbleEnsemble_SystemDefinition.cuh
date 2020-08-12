#ifndef PERBLOCKCOUPLING_SYSTEMDEFINITION_H
#define PERBLOCKCOUPLING_SYSTEMDEFINITION_H

#define PI 3.14159265358979323846

// SYSTEM
template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_OdeFunction(\
			int sid, int uid, \
			Precision*    F, Precision*     X, Precision     T,             \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR, int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC, int* isACC, \
			Precision*  CPT, Precision* CPF)
{
	Precision rx1 = 1.0/X[0];
	Precision p   = pow(rx1, uPAR[10]);
	
	Precision s1;
	Precision c1;
	sincospi(2.0*T, &s1, &c1);
	
	Precision s2 = sin(2.0*uPAR[11]*PI*T+uPAR[12]);
	Precision c2 = cos(2.0*uPAR[11]*PI*T+uPAR[12]);
	
	Precision N;
	Precision D;
	Precision rD;
	
	N = (uPAR[0]+uPAR[1]*X[1])*p - uPAR[2]*(1.0+uPAR[9]*X[1]) - uPAR[3]*rx1 - uPAR[4]*X[1]*rx1 - 1.5*(1.0-uPAR[9]*X[1]/3.0)*X[1]*X[1] - ( uPAR[5]*s1 + uPAR[6]*s2 ) * (1.0+uPAR[9]*X[1]) - X[0]*( uPAR[7]*c1 + uPAR[8]*c2 );
	D = X[0] - uPAR[9]*X[0]*X[1] + uPAR[4]*uPAR[9];
	rD = 1.0/D;
	
	// Uncoupled part, F[1]= x for solving ddR
	F[0] = X[1];
	F[1] = N*rD;
	
	// Explicit part of the coupling stored in Shared Memory
	// CPT is the temporary shared vector
	CPT[0] = -2*X[0]*X[1]*X[1];
	
	// Coupling term for fixed point iteration
	//CPT[0] = -(2*X[0]*X[1]*X[1] + X[0]*X[0]*F[1]);
	
	// For the composition of the Implicit Coupling Matrix, stored in Registers
	CPF[0] = rD;
	
	//if ( ( sid == 0 ) && ( uid == 0 ) )
	//	printf("CPT ODE: %+6.10e, X[0]: %6.10e, X[1]: %6.10e, T: %6.10e \n", CPT[0], X[0], X[1], T);
}

template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_UpdateCouplingTerm(\
			int sid, int uid, \
			Precision*    F, Precision*     X, Precision     T,             \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR, int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC, int* isACC, \
			Precision*  CPT, Precision* CPF)
{
	// Update coupling term for fixed-point iteration
	CPT[0] = -(2*X[0]*X[1]*X[1] + X[0]*X[0]*F[1]);
	
	//if ( ( sid == 0 ) && ( uid == 0 ) )
	//	printf("CPT UPD: %+6.10e, X[0]: %6.10e, X[1]: %6.10e, T: %6.10e \n", CPT[0], X[0], X[1], T);
}


// EVENTS
template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_EventFunction(\
			int sid, int uid, Precision* EF, \
			Precision     T, Precision     dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{	
	EF[0] = X[1];
}

template <class Precision> // ActionAfterSuccessfulTimeStep called first!
__forceinline__ __device__ void CoupledSystems_PerBlock_ActionAfterEventDetection(\
			int sid, int uid, int IDX, int& UDT, \
			Precision&    T, Precision&    dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{	
	iuACC[0]++;
	
	if ( iuACC[0] < NCOLL )
	{
		uACC[ iuACC[0]    ] = X[0]; // Store local maxima
		uACC[ iuACC[0]+NCOLL ] = X[0]; // Initiate its local minima
	}
}


// ACCESSORIES
template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_ActionAfterSuccessfulTimeStep(\
			int sid, int uid, int& UDT, \
			Precision&    T, Precision&    dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{
	if  ( ( iuACC[0] < NCOLL ) && ( X[0] < uACC[ iuACC[0]+NCOLL ] ) )
		uACC[ iuACC[0]+NCOLL ] = X[0]; // Update local minima
}

template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_Initialization(\
			int sid, int uid, int& DOIDX, \
			Precision&    T, Precision&    dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{
	if ( uid == 0 )
		T = TD[0];
}

template <class Precision>
__forceinline__ __device__ void CoupledSystems_PerBlock_Finalization(\
			int sid, int uid, int& DOIDX, \
			Precision&    T, Precision&    dT, Precision*   TD, Precision*     X, \
			Precision* uPAR, Precision*  sPAR, Precision* gPAR,       int* igPAR, \
			Precision* uACC,       int* iuACC, Precision* sACC,       int* isACC)
{
	
}

#endif