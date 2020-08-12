#ifndef COUPLEDSYSTEMS_PERBLOCK_SOLVER_H
#define COUPLEDSYSTEMS_PERBLOCK_SOLVER_H

#include "MPGOS_Overloaded_MathFunction.cuh"
#include "CoupledSystems_PerBlock_ExplicitRungeKutta_Steppers.cuh"
#include "CoupledSystems_PerBlock_ExplicitRungeKutta_ErrorControllers.cuh"
#include "CoupledSystems_PerBlock_EventHandling.cuh"
#include "CoupledSystems_PerBlock_DenseOutput.cuh"


template <int NS, int UPS, int UD, int TPB, int SPB, int NC, int CBW, int CCI, int NUP, int NSP, int NGP, int NiGP, int NUA, int NiUA, int NSA, int NiSA, int NE, int NDO, Algorithms Algorithm, class Precision>
__global__ void CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch(Struct_ThreadConfiguration ThreadConfiguration, Struct_GlobalVariables<Precision> GlobalVariables, Struct_SharedMemoryUsage SharedMemoryUsage, Struct_SolverOptions<Precision> SolverOptions)
{
	// THREAD MANAGEMENT ------------------------------------------------------
	const int LogicalThreadsPerBlock      = SPB * UPS;
	const int NumberOfBlockLaunches       = LogicalThreadsPerBlock / TPB + (LogicalThreadsPerBlock % TPB == 0 ? 0 : 1);
	const int ThreadPaddingPerBlock       = NumberOfBlockLaunches * TPB - LogicalThreadsPerBlock;
	const int TotalLogicalThreads         = (LogicalThreadsPerBlock + ThreadPaddingPerBlock) * gridDim.x;
	const int TotalLogicalThreadsPerBlock = (LogicalThreadsPerBlock + ThreadPaddingPerBlock);
	const int LocalThreadID               = threadIdx.x;
	const int GlobalThreadID              = threadIdx.x + blockIdx.x*blockDim.x;
	const int GlobalSystemID              = blockIdx.x;
	
	int GlobalMemoryID;
	
	
	// SHARED MEMORY MANAGEMENT -----------------------------------------------
	//    DUE TO REQUIRED MEMORY ALIGMENT: PRECISONS FIRST, INTS NEXT IN DYNAMICALLY ALLOCATED SHARED MEMORY
	//    SYSTEMS ARE PADDED IN SHARED MEMORY (NO GLOBAL BOUND CHECK IS NECESSARY AFTER LOADING)
	//    MINIMUM ALLOCABLE MEMORY IS 1
	extern __shared__ int DynamicSharedMemory[];
	int MemoryShift;
	
	Precision* gs_GlobalParameters = (Precision*)&DynamicSharedMemory;
		MemoryShift = (SharedMemoryUsage.GlobalVariables  == 1 ? NGP : 0);
	
	Precision* gs_CouplingMatrix = (Precision*)&gs_GlobalParameters[MemoryShift];
		MemoryShift = (SharedMemoryUsage.CouplingMatrices == 1 ? NC*SharedMemoryUsage.SingleCouplingMatrixSize : 0);
	
	int* gs_IntegerGlobalParameters = (int*)&gs_CouplingMatrix[MemoryShift];
	
	const bool IsAdaptive  = ( Algorithm==RK4 ? 0 : 1 );
	
	// Bank conflict if NCmod = 0, 2, 4, 8 or 16
	const int NCp   = ( NC==0 ? NC+1 : ( NC==2 ? NC+1 : ( NC==4 ? NC+1 : ( NC==8 ? NC+1 : ( NC==16 ? NC+1 : NC ) ) ) ) );
	// Bank conflicts if NSP, NSA and NiSA = 4, 8 or 16
	const int NSPp  = (  NSP==4 ?  NSP+1 : (  NSP==8 ?  NSP+1 : (  NSP==16 ?  NSP+1 : NSP  ) ) );
	const int NSAp  = (  NSA==4 ?  NSA+1 : (  NSA==8 ?  NSA+1 : (  NSA==16 ?  NSA+1 : NSA  ) ) );
	const int NiSAp = ( NiSA==4 ? NiSA+1 : ( NiSA==8 ? NiSA+1 : ( NiSA==16 ? NiSA+1 : NiSA ) ) );
	
	__shared__ Precision s_CouplingTerms[UPS][NCp];                       // Need access by user
	__shared__ Precision s_CouplingStrength[NCp];                         // Internal
	__shared__ Precision s_TimeDomain[2];                                 // Need access by user
	__shared__ Precision s_ActualTime;                                    // Need access by user
	__shared__ Precision s_TimeStep;                                      // Need access by user
	__shared__ Precision s_NewTimeStep;                                   // Need access by user
	__shared__ Precision s_SystemParameters[ (NSPp==0 ? 1 : NSPp) ];      // Need access by user
	__shared__ Precision s_SystemAccessories[ (NSAp==0 ? 1 : NSAp) ];     // Need access by user
	__shared__ Precision s_RelativeTolerance[ (IsAdaptive==0 ? 1 : UD) ]; // Internal
	__shared__ Precision s_AbsoluteTolerance[ (IsAdaptive==0 ? 1 : UD) ]; // Internal
	__shared__ Precision s_EventTolerance[ (NE==0 ? 1 : NE) ];            // Internal
	__shared__ Precision s_DenseOutputActualTime;                         // Internal
	__shared__ int s_CouplingIndex[NC];                                   // Internal
	__shared__ int s_DenseOutputIndex;                                    // Internal
	__shared__ int s_UpdateDenseOutput;                                   // Internal
	__shared__ int s_NumberOfSkippedStores;                               // Internal
	__shared__ int s_IntegerSystemAccessories[ (NiSAp==0 ? 1 : NiSAp) ];  // Need access by user
	__shared__ int s_EventDirection[ (NE==0 ? 1 : NE) ];                  // Need access by user
	__shared__ int s_TerminatedSystemsPerBlock;                           // Internal
	__shared__ int s_IsFinite;                                            // Internal
	__shared__ int s_TerminateSystemScope;                                // Internal
	__shared__ int s_UserDefinedTermination;                              // Internal
	__shared__ int s_UpdateStep;                                          // Internal
	__shared__ int s_EndTimeDomainReached;                                // Internal
	
	// Initialise block scope variables
	if ( threadIdx.x == 0 )
		s_TerminatedSystemsPerBlock = 0;
	__syncthreads();
	
	// Initialise system scope variables
	if ( ( threadIdx.x == 0 ) && ( GlobalSystemID < SolverOptions.ActiveSystems ) && ( GlobalSystemID < NS ) )
	{
		for (int i=0; i<2; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_TimeDomain[i] = GlobalVariables.d_TimeDomain[GlobalMemoryID];
			
			if ( i==0 )
			{
				s_ActualTime  = GlobalVariables.d_ActualTime[GlobalSystemID];
				s_TimeStep    = SolverOptions.InitialTimeStep;
				s_NewTimeStep = SolverOptions.InitialTimeStep;
			}
		}
		
		for (int i=0; i<NSP; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_SystemParameters[i] = GlobalVariables.d_SystemParameters[GlobalMemoryID];
		}
		
		for (int i=0; i<NSA; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_SystemAccessories[i] = GlobalVariables.d_SystemAccessories[GlobalMemoryID];
		}
		
		for (int i=0; i<NiSA; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_IntegerSystemAccessories[i] = GlobalVariables.d_IntegerSystemAccessories[GlobalMemoryID];
		}
		
		for (int i=0; i<NC; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_CouplingStrength[i] = GlobalVariables.d_CouplingStrength[GlobalMemoryID];
		}
		
		for (int i=0; i<1; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			s_DenseOutputIndex       = GlobalVariables.d_DenseOutputIndex[GlobalMemoryID];
			s_UpdateDenseOutput      = 1;
			s_NumberOfSkippedStores  = 0;
			s_TerminateSystemScope   = 0;
			s_UserDefinedTermination = 0;
		}
	}
	
	if ( ( threadIdx.x == 0 ) && ( ( GlobalSystemID >= SolverOptions.ActiveSystems ) || ( GlobalSystemID >= NS ) ) )
	{
		atomicAdd(&s_TerminatedSystemsPerBlock, 1);
		s_TerminateSystemScope = 1;
		s_UpdateStep           = 0;
		s_UpdateDenseOutput    = 0;
	}
	
	// Initialise solver tolerances
	if ( IsAdaptive  == 1 )
	{
		int Launches = UD / blockDim.x + (UD % blockDim.x == 0 ? 0 : 1);
		
		for (int i=0; i<Launches; i++)
		{
			int idx = threadIdx.x + i*blockDim.x;
			
			if ( idx < UD )
			{
				s_RelativeTolerance[idx] = GlobalVariables.d_RelativeTolerance[idx];
				s_AbsoluteTolerance[idx] = GlobalVariables.d_AbsoluteTolerance[idx];
			}
		}
	}
	
	// Initialise event options
	int Launches = NE / blockDim.x + (NE % blockDim.x == 0 ? 0 : 1);
	for (int i=0; i<Launches; i++)
	{
		int idx = threadIdx.x + i*blockDim.x;
		
		if ( idx < NE )
		{
			s_EventTolerance[idx]   = GlobalVariables.d_EventTolerance[idx];
			s_EventDirection[idx]   = GlobalVariables.d_EventDirection[idx];
		}
	}
	
	// Initialise global scope variables if ON
	if ( SharedMemoryUsage.GlobalVariables  == 0 )
	{
		gs_GlobalParameters        = GlobalVariables.d_GlobalParameters;
		gs_IntegerGlobalParameters = GlobalVariables.d_IntegerGlobalParameters;
	} else
	{
		int MaxElementNumber = max( NGP, NiGP );
		Launches = MaxElementNumber / blockDim.x + (MaxElementNumber % blockDim.x == 0 ? 0 : 1);
		
		for (int i=0; i<Launches; i++)
		{
			int idx = threadIdx.x + i*blockDim.x;
			
			if ( idx < NGP )
				gs_GlobalParameters[idx] = GlobalVariables.d_GlobalParameters[idx];
			
			if ( idx < NiGP )
				gs_IntegerGlobalParameters[idx] = GlobalVariables.d_IntegerGlobalParameters[idx];
		}
	}
	
	// Initialise coupling matrices if ON
	if ( SharedMemoryUsage.CouplingMatrices  == 0 )
	{
		gs_CouplingMatrix = GlobalVariables.d_CouplingMatrix;
	} else
	{
		int MaxElementNumber = NC*SharedMemoryUsage.SingleCouplingMatrixSize;
		Launches = MaxElementNumber / blockDim.x + (MaxElementNumber % blockDim.x == 0 ? 0 : 1);
		
		for (int i=0; i<Launches; i++)
		{
			int idx = threadIdx.x + i*blockDim.x;
			
			if ( idx < MaxElementNumber )
				gs_CouplingMatrix[idx] = GlobalVariables.d_CouplingMatrix[idx];
		}
	}
	
	// Initialise coupling index
	Launches = NC / blockDim.x + (NC % blockDim.x == 0 ? 0 : 1);	
	for (int i=0; i<Launches; i++)
	{
		int idx = threadIdx.x + i*blockDim.x;
		
		if ( idx < NC )
			s_CouplingIndex[idx] = GlobalVariables.d_CouplingIndex[idx];
	}
	
	
	// REGISTER MEMORY MANAGEMENT ---------------------------------------------
	//    THREADS ARE PADDED (NO GLOBAL BOUND CHECK IS NECESSARY AFTER LOADING)
	//    MINIMUM ALLOCABLE MEMORY IS 1
	Precision r_CouplingFactor[NC];
	Precision r_ActualState[UD];
	Precision r_NextState[UD];
	Precision r_ActualEventValue[ (NE==0 ? 1 : NE) ];
	Precision r_NextEventValue[ (NE==0 ? 1 : NE) ];
	Precision r_Error[UD];
	Precision r_UnitParameters[ (NUP==0 ? 1 : NUP) ];
	Precision r_UnitAccessories[ (NUA==0 ? 1 : NUA) ];
	int       r_IntegerUnitAccessories[ (NiUA==0 ? 1 : NiUA) ];
	
	// Initialise unit scope variables
	for (int i=0; i<UD; i++)
	{
		GlobalMemoryID = GlobalThreadID + i*TotalLogicalThreads;
		r_ActualState[i] = GlobalVariables.d_ActualState[GlobalMemoryID];
	}
	
	for (int i=0; i<NUP; i++)
	{
		GlobalMemoryID = GlobalThreadID + i*TotalLogicalThreads;
		r_UnitParameters[i] = GlobalVariables.d_UnitParameters[GlobalMemoryID];
	}
	
	for (int i=0; i<NUA; i++)
	{
		GlobalMemoryID = GlobalThreadID + i*TotalLogicalThreads;
		r_UnitAccessories[i] = GlobalVariables.d_UnitAccessories[GlobalMemoryID];
	}
	
	for (int i=0; i<NiUA; i++)
	{
		GlobalMemoryID = GlobalThreadID + i*TotalLogicalThreads;
		r_IntegerUnitAccessories[i] = GlobalVariables.d_IntegerUnitAccessories[GlobalMemoryID];
	}
	__syncthreads();
	
	
	// INITIALISATION
	if ( ( LocalThreadID < UPS ) && ( s_TerminateSystemScope == 0 ) )
	{
		CoupledSystems_PerBlock_Initialization<Precision>(\
			GlobalSystemID, \
			LocalThreadID, \
			s_DenseOutputIndex, \
			s_ActualTime, \
			s_TimeStep, \
			&s_TimeDomain[0], \
			&r_ActualState[0], \
			&r_UnitParameters[0], \
			&s_SystemParameters[0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[0], \
			&s_IntegerSystemAccessories[0]);
	}
	__syncthreads();
	
	if ( NE > 0 )
	{
		if ( ( LocalThreadID < UPS ) && ( s_TerminateSystemScope == 0 ) )
		{
			CoupledSystems_PerBlock_EventFunction<Precision>(\
				GlobalSystemID, \
				LocalThreadID, \
				&r_ActualEventValue[0], \
				s_ActualTime, \
				s_TimeStep, \
				&s_TimeDomain[0], \
				&r_ActualState[0], \
				&r_UnitParameters[0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[0], \
				&r_IntegerUnitAccessories[0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0]);
		}
		__syncthreads();
	}
	
	if ( NDO > 0 )
	{
		CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_StoreDenseOutput<NS, UPS, UD, Precision>(\
			LocalThreadID, \
			GlobalThreadID, \
			GlobalSystemID, \
			s_UpdateStep, \
			s_UpdateDenseOutput, \
			s_DenseOutputIndex, \
			s_NumberOfSkippedStores, \
			GlobalVariables.d_DenseOutputTimeInstances, \
			GlobalVariables.d_DenseOutputStates, \
			s_DenseOutputActualTime, \
			s_ActualTime, \
			r_ActualState, \
			s_TimeDomain, \
			ThreadConfiguration, \
			SolverOptions);
	}
	
	
	// SOLVER MANAGEMENT ------------------------------------------------------
	while ( s_TerminatedSystemsPerBlock < SPB )
	{
		// INITIALISE TIME STEPPING -------------------------------------------
		if ( ( threadIdx.x == 0 ) && ( s_TerminateSystemScope == 0 ) )
		{
			s_UpdateStep           = 1;
			s_IsFinite             = 1;
			s_EndTimeDomainReached = 0;
			
			s_TimeStep = s_NewTimeStep;
			
			if ( s_TimeStep > ( s_TimeDomain[1] - s_ActualTime ) )
			{
				s_TimeStep = s_TimeDomain[1] - s_ActualTime;
				s_EndTimeDomainReached = 1;
			}
		}
		__syncthreads();
		
		
		if ( Algorithm == RKCK45 )
		{
			CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_Stepper_RKCK45<NS,UPS,UD,TPB,SPB,NC,NCp,CBW,CCI,NUP,NSPp,NGP,NiGP,NUA,NiUA,NSAp,NiSAp,NE,NDO,Precision>(\
				LocalThreadID, \
				GlobalSystemID, \
				r_ActualState, \
				r_NextState, \
				s_ActualTime, \
				s_TimeStep, \
				s_IsFinite, \
				r_Error, \
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
				s_CouplingIndex);
			
			CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_ErrorController_RKCK45<UPS, UD, Precision>(\
				LocalThreadID, \
				GlobalSystemID, \
				s_IsFinite, \
				s_TimeStep, \
				s_NewTimeStep, \
				r_ActualState, \
				r_NextState, \
				r_Error, \
				s_RelativeTolerance, \
				s_AbsoluteTolerance, \
				s_TerminateSystemScope, \
				s_TerminatedSystemsPerBlock, \
				s_UpdateStep, \
				SolverOptions);
		}
		
		
		// NEW EVENT VALUE AND TIME STEP CONTROL-------------------------------
		if ( NE > 0 )
		{
			if ( ( LocalThreadID < UPS ) && ( s_TerminateSystemScope == 0 ) )
			{
				CoupledSystems_PerBlock_EventFunction<Precision>(\
					GlobalSystemID, \
					LocalThreadID, \
					&r_NextEventValue[0], \
					s_ActualTime+s_TimeStep, \
					s_NewTimeStep, \
					&s_TimeDomain[0], \
					&r_NextState[0], \
					&r_UnitParameters[0], \
					&s_SystemParameters[0], \
					gs_GlobalParameters, \
					gs_IntegerGlobalParameters, \
					&r_UnitAccessories[0], \
					&r_IntegerUnitAccessories[0], \
					&s_SystemAccessories[0], \
					&s_IntegerSystemAccessories[0]);
			}
			__syncthreads();
			
			CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_EventTimeStepControl<UPS, NE, Precision>(\
				GlobalSystemID, \
				LocalThreadID, \
				s_TimeStep, \
				s_NewTimeStep, \
				s_TerminateSystemScope, \
				s_UpdateStep, \
				r_ActualEventValue, \
				r_NextEventValue, \
				s_EventTolerance, \
				s_EventDirection, \
				SolverOptions.MinimumTimeStep);
		}
		
		
		// UPDATE PROCESS -----------------------------------------------------
		// Update actual state and time for an accepted step
		if ( ( LocalThreadID < UPS ) && ( s_UpdateStep == 1 ) )
		{
			if ( LocalThreadID == 0 )
				s_ActualTime += s_TimeStep;
			
			for (int i=0; i<UD; i++)
				r_ActualState[i] = r_NextState[i];
		}
		__syncthreads();
		
		// Call user defined ActionAfterSuccessfulTimeStep
		if ( ( LocalThreadID < UPS ) && ( s_UpdateStep == 1 ) )
		{
			CoupledSystems_PerBlock_ActionAfterSuccessfulTimeStep<Precision>(\
				GlobalSystemID, \
				LocalThreadID, \
				s_UserDefinedTermination, \
				s_ActualTime, \
				s_TimeStep, \
				&s_TimeDomain[0], \
				&r_ActualState[0], \
				&r_UnitParameters[0], \
				&s_SystemParameters[0], \
				gs_GlobalParameters, \
				gs_IntegerGlobalParameters, \
				&r_UnitAccessories[0], \
				&r_IntegerUnitAccessories[0], \
				&s_SystemAccessories[0], \
				&s_IntegerSystemAccessories[0]);
		}
		__syncthreads();
		
		if ( NE > 0 )
		{
			// Call user defined ActionAfterEventDetection if event is detected
			if ( ( LocalThreadID < UPS ) && ( s_UpdateStep == 1 ) )
			{	
				for (int i=0; i<NE; i++)
				{
					if ( ( ( r_ActualEventValue[i] >  s_EventTolerance[i] ) && ( abs(r_NextEventValue[i]) < s_EventTolerance[i] ) && ( s_EventDirection[i] <= 0 ) ) || \
						 ( ( r_ActualEventValue[i] < -s_EventTolerance[i] ) && ( abs(r_NextEventValue[i]) < s_EventTolerance[i] ) && ( s_EventDirection[i] >= 0 ) ) )
					{
						CoupledSystems_PerBlock_ActionAfterEventDetection<Precision>(\
							GlobalSystemID, \
							LocalThreadID, \
							i, \
							s_UserDefinedTermination, \
							s_ActualTime, \
							s_TimeStep, \
							&s_TimeDomain[0], \
							&r_ActualState[0], \
							&r_UnitParameters[0], \
							&s_SystemParameters[0], \
							gs_GlobalParameters, \
							gs_IntegerGlobalParameters, \
							&r_UnitAccessories[0], \
							&r_IntegerUnitAccessories[0], \
							&s_SystemAccessories[0], \
							&s_IntegerSystemAccessories[0]);
					}
				}
			}
			__syncthreads();
			
			// Update next event value (modification of state variables is possuble by the user)
			if ( ( LocalThreadID < UPS ) && ( s_UpdateStep == 1 ) )
			{
				CoupledSystems_PerBlock_EventFunction<Precision>(\
					GlobalSystemID, \
					LocalThreadID, \
					&r_NextEventValue[0], \
					s_ActualTime, \
					s_NewTimeStep, \
					&s_TimeDomain[0], \
					&r_ActualState[0], \
					&r_UnitParameters[0], \
					&s_SystemParameters[0], \
					gs_GlobalParameters, \
					gs_IntegerGlobalParameters, \
					&r_UnitAccessories[0], \
					&r_IntegerUnitAccessories[0], \
					&s_SystemAccessories[0], \
					&s_IntegerSystemAccessories[0]);
			}
			__syncthreads();	
				
			// Update event values
			if ( ( LocalThreadID < UPS ) && ( s_UpdateStep == 1 ) )
			{
				for (int i=0; i<NE; i++)
					r_ActualEventValue[i] = r_NextEventValue[i];
			}
			__syncthreads();
		}
		
		if ( NDO > 0)
		{
			CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_DenseOutputStorageCondition<NDO, Precision>(\
				s_EndTimeDomainReached, \
				s_UserDefinedTermination, \
				s_UpdateStep, \
				s_UpdateDenseOutput, \
				s_DenseOutputIndex, \
				s_NumberOfSkippedStores, \
				s_DenseOutputActualTime, \
				s_ActualTime, \
				SolverOptions);
			
			CoupledSystems_PerBlock_SingleSystem_SingleBlockLaunch_StoreDenseOutput<NS, UPS, UD, Precision>(\
				LocalThreadID, \
				GlobalThreadID, \
				GlobalSystemID, \
				s_UpdateStep, \
				s_UpdateDenseOutput, \
				s_DenseOutputIndex, \
				s_NumberOfSkippedStores, \
				GlobalVariables.d_DenseOutputTimeInstances, \
				GlobalVariables.d_DenseOutputStates, \
				s_DenseOutputActualTime, \
				s_ActualTime, \
				r_ActualState, \
				s_TimeDomain, \
				ThreadConfiguration, \
				SolverOptions);
		}
		
		// Chect termination
		if ( ( threadIdx.x == 0 ) && ( s_UpdateStep == 1 ) )
		{
			if ( ( s_EndTimeDomainReached == 1 ) || ( s_UserDefinedTermination == 1 ) )
			{
				s_TerminateSystemScope = 1;
				atomicAdd(&s_TerminatedSystemsPerBlock, 1);
				
				s_UpdateStep = 0;
			}
		}
		__syncthreads();
		
	}
	__syncthreads();
	
	
	// FINALISATION
	if ( LocalThreadID < UPS )
	{
		CoupledSystems_PerBlock_Finalization<Precision>(\
			GlobalSystemID, \
			LocalThreadID, \
			s_DenseOutputIndex, \
			s_ActualTime, \
			s_TimeStep, \
			&s_TimeDomain[0], \
			&r_ActualState[0], \
			&r_UnitParameters[0], \
			&s_SystemParameters[0], \
			gs_GlobalParameters, \
			gs_IntegerGlobalParameters, \
			&r_UnitAccessories[0], \
			&r_IntegerUnitAccessories[0], \
			&s_SystemAccessories[0], \
			&s_IntegerSystemAccessories[0]);
	}
	__syncthreads();
	
	
	// WRITE DATA BACK TO GLOBAL MEMORY ---------------------------------------
	// Unit scope variables (register variables)
	for (int i=0; i<UD; i++)
	{
		GlobalMemoryID = GlobalThreadID + i*TotalLogicalThreads;
		GlobalVariables.d_ActualState[GlobalMemoryID] = r_ActualState[i];
	}
	
	for (int i=0; i<NUA; i++)
	{
		GlobalMemoryID = GlobalThreadID + i*TotalLogicalThreads;
		GlobalVariables.d_UnitAccessories[GlobalMemoryID] = r_UnitAccessories[i];
	}
	
	for (int i=0; i<NiUA; i++)
	{
		GlobalMemoryID = GlobalThreadID + i*TotalLogicalThreads;
		GlobalVariables.d_IntegerUnitAccessories[GlobalMemoryID] = r_IntegerUnitAccessories[i];
	}
	__syncthreads();
	
	// System scope variables (shared variables)
	if ( ( threadIdx.x == 0 ) && ( GlobalSystemID < SolverOptions.ActiveSystems ) && ( GlobalSystemID < NS ) )
	{
		GlobalVariables.d_ActualTime[GlobalSystemID] = s_ActualTime;
		
		for (int i=0; i<2; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			GlobalVariables.d_TimeDomain[GlobalMemoryID] = s_TimeDomain[i];
		}
		
		for (int i=0; i<NSA; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			GlobalVariables.d_SystemAccessories[GlobalMemoryID] = s_SystemAccessories[i];
		}
		
		for (int i=0; i<NiSA; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			GlobalVariables.d_IntegerSystemAccessories[GlobalMemoryID] = s_IntegerSystemAccessories[i];
		}
		
		for (int i=0; i<1; i++)
		{
			GlobalMemoryID = GlobalSystemID + i*NS;
			GlobalVariables.d_DenseOutputIndex[GlobalMemoryID] = s_DenseOutputIndex;
		}
	}
	
	__syncthreads();
}

#endif