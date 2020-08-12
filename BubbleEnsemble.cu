#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <random>
#include <math.h>

#include "BubbleEnsemble_SystemDefinition.cuh"
#include "CoupledSystems_PerBlock_Interface.cuh"

#define PI 3.14159265358979323846

using namespace std;

// Physical control parameters
const int NumberOfFrequency = 1;
const int NumberOfAmplitude = 480;
const int NumberOfInitialConditions = 1;
const int NumberOfUnitsPerSystem = 32;

const int AnalysisType = 0; // 0 bifurcation diagram of Poincare sections
                            // 1 period numbers
							// 2 amplifications

// Solver Configuration
const int TransientIterations = 5;
const int PoincareSections = 32;
#define ICTYPE "LIN"
#define SOLVER RKCK45      // RK4, RKCK45
#define PRECISION double   // float, double
const int NS   = NumberOfFrequency * NumberOfAmplitude*NumberOfInitialConditions; // NumberOfSystems
const int UPS  = NumberOfUnitsPerSystem; // UnitsPerSystem
const int UD   = 2;     // UnitDimension
const int TPB  = 32;    // ThreadsPerBlock (integer multiple of the warp size that is 32)
const int SPB  = 1;     // SystemsPerBlock
const int NC   = 1;     // NumberOfCouplings
const int CBW  = 0;     // CouplingBandwidthRadius (0: full coupling matrix)
const int CCI  = 0;     // CouplingCircularity (0: non-circular matrix, 1: circular matrix)

const int NUP  = 21;    // NumberOfUnitParameters (different form system to system, different from unit to unit)
const int NSP  = 0;     // NumberOfSystemParameters (different from system to system, shared by all units)
const int NGP  = 0;     // NumberOfGlobalParameters (shared by all systems, share by all units)
const int NiGP = 0;     // NumberOfIntegerGlobalParameters (shared by all systems, shared by all units)

const int NUA  = 1;     // NumberOfUnitAccessories (different form system to system, different from unit to unit)
const int NiUA = 0;     // NumberOfIntegerUnitAccessories (different form system to system, different from unit to unit)
const int NSA  = 0;     // NumberOfSystemAccessories (different from system to system, shared by all units)
const int NiSA = 0;     // NumberOfIntegerSystemAccessories (different from system to system, shared by all units)

const int NE   = 0;     // NumberOfEvents (per units)
const int NDO  = 0;   // NumberOfPointsOfDenseOutput (per units)

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Logspace(vector<PRECISION>&, PRECISION, PRECISION, int);

void LoadBubbleSizes(vector<PRECISION>&, string, int);
void LoadCouplingMatrix(vector<vector<PRECISION>>&, string, int);

void FillSolverObject(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&);
void FillCouplingMatrix(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, const vector<vector<PRECISION>>&, int);

void SavePoincare(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, ofstream&, int, int, int);
void SavePeriod(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, vector<vector<int>>&, ofstream&, int, int, int);
void SaveRelativeExpansion(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, vector<vector<int>>&, ofstream&, int, int, int);

// ------------------------------------------------------------------------------------------------

int main()
{
	// ALLOCATE PARAMETERS ----------------------
	vector<PRECISION> Frequency(NumberOfFrequency,0);
	vector<PRECISION> Amplitude(NumberOfAmplitude,0);
	vector<PRECISION> BubbleSize(UPS,0);
	vector<PRECISION> InitialRadii(NumberOfInitialConditions,0);
	vector<vector<PRECISION>> CouplingMatrixData(UPS, vector<PRECISION>(UPS));
	
	Logspace(Frequency, 1000.0, 2000.0, NumberOfFrequency); // kHz
	Linspace(Amplitude, 0.0, 5.0, NumberOfAmplitude); // bar
	Linspace(InitialRadii, 1.6, 2.2, NumberOfInitialConditions); // bar
	
	// SELECT CUDA DEVICE
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	// DEFINE PROBLEM SOLVER AND ITS OPTIONS ----
	ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION> ScanSystem(SelectedDevice);
	
	ScanSystem.SolverOption(SharedCouplingMatrices, 0);
	ScanSystem.SolverOption(InitialTimeStep, 1e-3);
	ScanSystem.SolverOption(RelativeTolerance, 0, 1e-10);
	ScanSystem.SolverOption(AbsoluteTolerance, 0, 1e-10);
	ScanSystem.SolverOption(MinimumTimeStep, 1e-13);
	
	// FILL SOLVER OBJECT WITH DATA -------------
	LoadBubbleSizes(BubbleSize, "BubbleSizes_N32_D1_d150_RE2-8.txt", UPS);
	LoadCouplingMatrix(CouplingMatrixData, "CouplingMatrix_N32_D1_d150_RE2-8.txt", UPS);
	
	FillSolverObject(ScanSystem, Frequency, Amplitude, BubbleSize, InitialRadii);
	FillCouplingMatrix(ScanSystem, CouplingMatrixData, UPS);
	
	ScanSystem.Print(ActualState);
	
	// SIMULATION -------------------------------
	
	ofstream DataFile;
	string Filename;
	stringstream StreamFilename;
	StreamFilename.str("");
	StreamFilename.precision(0);
	StreamFilename.setf(ios::fixed);
	
	if ( AnalysisType == 0 )
	{
		//StreamFilename << "PoincareSection_N32_f1000_PA" << Amplitude[0] << "-" << Amplitude.back() \
		               << "_" << ICTYPE << NumberOfInitialConditions << "_FIT16_TRN" << TransientIterations \
		               << "_PSEC" << PoincareSections << "_RES" << NumberOfAmplitude << ".txt";
		
		StreamFilename << "PoincareSection_N32_f1000_PA" << Amplitude[0] << "-" << Amplitude.back() \
		               << "_" << ICTYPE << NumberOfInitialConditions << "_BiCG_TRN" << TransientIterations \
		               << "_PSEC" << PoincareSections << "_RES" << NumberOfAmplitude << ".txt";
	}
	
	if ( AnalysisType == 1 )
	{
		StreamFilename << "PeriodNumber_N32_f200_PA" << Amplitude[0] << "-" << Amplitude.back() \
		               << "_" << ICTYPE << NumberOfInitialConditions << "_FIT16_TRN" << TransientIterations \
		               << "_PSEC" << PoincareSections << "_RES" << NumberOfAmplitude << ".txt";
	}
	
	if ( AnalysisType == 2 )
	{
		StreamFilename << "RelativeExpansion_N32_f20-2000_PA2_RES101_WithCoupling.txt";
	}
	
	Filename = StreamFilename.str();
	
	DataFile.open ( Filename.c_str() );
	
	clock_t SimulationStart = clock();
	
	ScanSystem.SynchroniseFromHostToDevice(All);
	
	vector<vector<vector<PRECISION>>> LastPointOfTransient(NS, vector<vector<PRECISION>>(UPS, vector<PRECISION>(UD)));
	vector<vector<int>> PeriodNumber(NS, vector<int>(UPS));
	vector<vector<int>> PeriodFound(NS, vector<int>(UPS));
	vector<vector<int>> RelativeExpansion(NS, vector<int>(UPS));
	
	for (int i=0; i<TransientIterations; i++)
	{
		ScanSystem.Solve();
		ScanSystem.InsertSynchronisationPoint();
		ScanSystem.SynchroniseSolver();
		
		cout << "Transient finished: " << i << endl;
	}
	
	ScanSystem.SynchroniseFromDeviceToHost(All);
	ScanSystem.InsertSynchronisationPoint();
	ScanSystem.SynchroniseSolver();
	
	clock_t SimulationEnd = clock();
	cout << "Transient simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
	// Store the last point of the transient
	cout << "Start transient storing!" << endl;
	for (int sid=0; sid<NS; sid++)
	{
		for (int uid=0; uid<UPS; uid++)
		{
			for (int cmp=0; cmp<UD; cmp++)
			{
				LastPointOfTransient[sid][uid][cmp] = ScanSystem.GetHost<PRECISION>(sid, uid, ActualState, cmp);
			}
			PeriodFound[sid][uid] = 0;
			RelativeExpansion[sid][uid] = ScanSystem.GetHost<PRECISION>(sid, uid, UnitAccessories, 0);
		}
	}
	
	// Save Poincare section
	cout << "Start Poincare section!" << endl;
	for (int i=0; i<PoincareSections; i++)
	{
		ScanSystem.Solve();
		ScanSystem.SynchroniseFromDeviceToHost(All);
		ScanSystem.InsertSynchronisationPoint();
		ScanSystem.SynchroniseSolver();
		
		// Compare actual state with the last point of the trasient
		for (int sid=0; sid<NS; sid++)
		{
			for (int uid=0; uid<UPS; uid++)
			{
				PRECISION y1t = LastPointOfTransient[sid][uid][0];
				PRECISION y2t = LastPointOfTransient[sid][uid][1];
				PRECISION y1a = ScanSystem.GetHost<PRECISION>(sid, uid, ActualState, 0);
				PRECISION y2a = ScanSystem.GetHost<PRECISION>(sid, uid, ActualState, 1);
				
				PRECISION Distance = sqrt( (y1t-y1a)*(y1t-y1a) + (y2t-y2a)*(y2t-y2a) );
				
				if ( ( Distance < 1e-6 ) && ( PeriodFound[sid][uid] == 0 ) )
				{
					PeriodNumber[sid][uid] = i+1;
					PeriodFound[sid][uid] = 1;
				}
				
				if ( ScanSystem.GetHost<PRECISION>(sid, uid, UnitAccessories, 0) > RelativeExpansion[sid][uid] )
					RelativeExpansion[sid][uid] = ScanSystem.GetHost<PRECISION>(sid, uid, UnitAccessories, 0);
			}
		}
		
		if ( AnalysisType == 0 )
			SavePoincare(ScanSystem, DataFile, NS, UPS, UD);
		
		cout << "Poincare finished: " << i << endl;
	}
	
	//if ( AnalysisType == 1 )
	//		SavePeriod(ScanSystem, PeriodNumber, DataFile, NS, UPS, UD);
	
	if ( AnalysisType == 2 )
		SaveRelativeExpansion(ScanSystem, RelativeExpansion, DataFile, NS, UPS, UD);
	
	SimulationEnd = clock();
	cout << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
	DataFile.close();
	
	cout << "SIMULATION COMPLETED!" << endl;
}

// ------------------------------------------------------------------------------------------------

void Linspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
    PRECISION Increment;
	
	x[0]   = B;
	
	if ( N>1 )
	{
		x[N-1] = E;
		Increment = (E-B)/(N-1);
		
		for (int i=1; i<N-1; i++)
		{
			x[i] = B + i*Increment;
		}
	}
}

void Logspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
    x[0] = B; 
	
	if ( N>1 )
	{
		x[N-1] = E;
		PRECISION ExpB = log10(B);
		PRECISION ExpE = log10(E);
		PRECISION ExpIncr = (ExpE-ExpB)/(N-1);
		for (int i=1; i<N-1; i++)
		{
			x[i] = pow(10,ExpB + i*ExpIncr);
		}
	}
}

// ------------------------------------------------------------------------------------------------

void LoadBubbleSizes(vector<PRECISION>& BubbleSize, string FileName, int UPS)
{
	ifstream DataFile;
	
	DataFile.open ( FileName.c_str() );
	
	for ( int i=0; i<UPS; i++ )
	{
		DataFile >> BubbleSize[i];
		//cout << scientific << setw(20) << setprecision(13) << BubbleSize[i] << '\n';
	}
	
	DataFile.close();
}

void LoadCouplingMatrix(vector<vector<PRECISION>>& CouplingMatrixData, string FileName, int UPS)
{
	ifstream DataFile;
	
	DataFile.open ( FileName.c_str() );
	
	for (int i=0; i<UPS; i++)
	{
		for (int j=0; j<UPS; j++)
		{
			DataFile >> CouplingMatrixData[i][j];
			//cout << scientific << setw(20) << setprecision(13) << CouplingMatrixData[i][j] << " ";
		}
		//cout << '\n';
	}
	
	DataFile.close();
}

void FillSolverObject(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& Frequency, const vector<PRECISION>& Amplitude, const vector<PRECISION>& BubbleSize, const vector<PRECISION>& InitialRadii)
{
	srand(time(NULL));
	//PRECISION LowerLimit = InitialRadii[0];
	//PRECISION UpperLimit = InitialRadii.back();
	PRECISION LowerLimit = 0.9;
	PRECISION UpperLimit = 2.2;
	int Random;
	int Resolution = 100001;
	
	PRECISION P1;   // pressure amplitude1 [bar] (Shared among units, different among blocks; the first control parameter)
	PRECISION P2;   // relative frequency1 [kHz] (Shared among units, different among blocks; the second control parameter)
	PRECISION P3=0; // pressure amplitude2 [bar] (Zero)
	PRECISION P4=0; // relative frequency2 [kHz] (Zero)
	PRECISION P5=0; // phase shift         [-]   (Zero)
	PRECISION P6;   // equilibrium radius  [mum] (Shared among blocks, different among units; NOT a control parameter, describes the bubble size distribution)
	
	PRECISION P7 = 1.0; // ambient pressure     [bar]
	PRECISION P9 = 1.4; // polytrophic exponent [-]
	
	PRECISION Pv  = 3.166775638952003e+03;
    PRECISION Rho = 9.970639504998557e+02;
    PRECISION ST  = 0.071977583160056;
    PRECISION Vis = 8.902125058209557e-04;
    PRECISION CL  = 1.497251785455527e+03;
	
	PRECISION Pinf;
	PRECISION PA1;
	PRECISION PA2;
	PRECISION RE;
	PRECISION f1;
	PRECISION f2;
	
	// Loop over the systems (blocks): control parameters
	int SystemNumber = 0;
	for (auto const& CP2: Frequency) // Frequency [kHz]
	{
		for (auto const& CP1: Amplitude) // Amplitude [bar]
		{
			for (auto const& IC: InitialRadii) // Initila radii [-]
			{
				// SYSTEM SCOPE
				Solver.SetHost(SystemNumber, TimeDomain, 0, 0.0);
				Solver.SetHost(SystemNumber, TimeDomain, 1, 1.0);
				
				for (int i=0; i<NC; i++)
					Solver.SetHost(SystemNumber, CouplingStrength, i, 1.0);
				
				// UNIT SCOPE
				int UnitNumber = 0;
				for (auto const& CP0: BubbleSize) // equilibrium radius  [mum]
				{
					P1 = CP1; // pressure amplitude1 [bar]
					P2 = CP2; // relative frequency1 [kHz]
					P6 = CP0; // equilibrium radius  [mum]
					
					Random = (rand() % Resolution);
					//Solver.SetHost(SystemNumber, UnitNumber, ActualState, 0, LowerLimit + Random*(UpperLimit-LowerLimit)/Resolution );
					Solver.SetHost(SystemNumber, UnitNumber, ActualState, 0, IC);
					Solver.SetHost(SystemNumber, UnitNumber, ActualState, 1, 0.0);
					
					// Dimensional physical parameters
					Pinf = P7 * 1e5;
					PA1  = P1 * 1e5;
					PA2  = P3 * 1e5;
					RE   = P6 / 1e6;
					
					f1   = 2.0*PI*(P2*1000);
					f2   = 2.0*PI*(P4*1000);
					
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  0, (2.0*ST/RE + Pinf - Pv) * pow(2.0*PI/RE/f1, 2.0) / Rho );
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  1, (1.0-3.0*P9) * (2*ST/RE + Pinf - Pv) * (2.0*PI/RE/f1) / CL/Rho );
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  2, (Pinf - Pv) * pow(2.0*PI/RE/f1, 2.0) / Rho );
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  3, (2.0*ST/RE/Rho) * pow(2.0*PI/RE/f1, 2.0) );
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  4, (4.0*Vis/Rho/pow(RE,2.0)) * (2.0*PI/f1) );
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  5, PA1 * pow(2.0*PI/RE/f1, 2.0) / Rho );
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  6, PA2 * pow(2.0*PI/RE/f1, 2.0) / Rho );          // ZERO: Shared
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  7, (RE*f1*PA1/Rho/CL) * pow(2.0*PI/RE/f1, 2.0) );
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  8, (RE*f2*PA2/Rho/CL) * pow(2.0*PI/RE/f1, 2.0) ); // ZERO: Shared
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters,  9, RE*f1/(2.0*PI)/CL );
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 10, 3.0*P9 );
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 11, P4/P2 ); // ZERO: Shared
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 12, P5 );    // ZERO: Shared
					
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 13, 2.0*PI/f1 ); // tref
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 14, RE );        // Rref
					
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 15, P1 );
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 16, P2 );
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 17, P3 ); // ZERO
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 18, P4 ); // ZERO
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 19, P5 ); // ZERO
					Solver.SetHost(SystemNumber, UnitNumber, UnitParameters, 20, P6 );
					
					UnitNumber++;
				}
				
				SystemNumber++;
			}
		}
	}
	
	// GLOBAL SCOPE
	for (int i=0; i<NC; i++)
		Solver.SetHost(CouplingIndex, i, 1);
}

void FillCouplingMatrix(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, const vector<vector<PRECISION>>& CouplingMatrixData, int UPS)
{
	for (int Row=0; Row<UPS; Row++)
	{
		for (int Col=0; Col<UPS; Col++)
		{
			// Full Matrix Storage
			Solver.SetHost(0, CouplingMatrix, Row, Col, CouplingMatrixData[Row][Col]);
		}
	}
}


// ------------------------------------------------------------------------------------------------

void SavePoincare(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, ofstream& DataFile, int NS, int UPS, int UD)
{
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	for (int sid=0; sid<NS; sid++)
	{
		DataFile.width(4); DataFile << sid << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(sid, 0, UnitParameters, 15) << ',';
		for (int uid=0; uid<UPS; uid++)
		{
			for (int cmp=0; cmp<1; cmp++)
			{
				//cout << "sid: " << sid << " uid: " << uid << " cmp: " << cmp << " Value: " << Solver.GetHost<PRECISION>(sid, uid, ActualState, cmp) << endl;
				DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(sid, uid, ActualState, 0) << ',';
			}
		}
		DataFile << '\n';
	}
}

void SavePeriod(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, vector<vector<int>>& Periods, ofstream& DataFile, int NS, int UPS, int UD)
{
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	for (int sid=0; sid<NS; sid++)
	{
		DataFile.width(4); DataFile << sid << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(sid, 0, UnitParameters, 15) << ',';
		for (int uid=0; uid<UPS; uid++)
		{
			DataFile.width(4); DataFile << Periods[sid][uid] << ',';
		}
		DataFile << '\n';
	}
}

void SaveRelativeExpansion(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, vector<vector<int>>& RelativeExpansion, ofstream& DataFile, int NS, int UPS, int UD)
{
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	for (int sid=0; sid<NS; sid++)
	{
		DataFile.width(4); DataFile << sid << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(sid, 0, UnitParameters, 16) << ',';
		for (int uid=0; uid<UPS; uid++)
		{
			DataFile.width(4); DataFile << RelativeExpansion[sid][uid] << ',';
		}
		DataFile << '\n';
	}
}