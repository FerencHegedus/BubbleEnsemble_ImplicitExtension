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

#define PI 3.14159265358979323846

// Physical control parameters and solver configuration
#define PAMIN 0.0
#define PAMAX 2.0
#define PARES 101
#define FRMIN 20.0
#define FRMAX 2000.0
#define FRRES 101
#define CSTR 1.0
#define NTRN 2
#define NCOLL 3
#define DEVICE 1

#include "BubbleEnsemble_SystemDefinition.cuh"
#include "CoupledSystems_PerBlock_Interface.cuh"

using namespace std;

#define SOLVER RKCK45      // RK4, RKCK45
#define PRECISION double   // float, double
const int NS   = PARES*FRRES; // NumberOfSystems
const int UPS  = 32;    // UnitsPerSystem
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

const int NUA  = 2*NCOLL; // NumberOfUnitAccessories (different form system to system, different from unit to unit)
const int NiUA = 1;     // NumberOfIntegerUnitAccessories (different form system to system, different from unit to unit)
const int NSA  = 0;     // NumberOfSystemAccessories (different from system to system, shared by all units)
const int NiSA = 0;     // NumberOfIntegerSystemAccessories (different from system to system, shared by all units)

const int NE   = 1;     // NumberOfEvents (per units)
const int NDO  = 0;     // NumberOfPointsOfDenseOutput (per units)

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Logspace(vector<PRECISION>&, PRECISION, PRECISION, int);

void LoadBubbleSizes(vector<PRECISION>&, string, int);
void LoadCouplingMatrix(vector<vector<PRECISION>>&, string, int);

void FillSolverObject(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&);
void ResetAccessories(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&, const vector<PRECISION>&);
void FillCouplingMatrix(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>&, const vector<vector<PRECISION>>&, int);

// ------------------------------------------------------------------------------------------------

int main()
{
	// ALLOCATE PARAMETERS ----------------------
	vector<PRECISION> Frequency(FRRES,0);
	vector<PRECISION> Amplitude(PARES,0);
	vector<PRECISION> BubbleSize(UPS,0);
	vector<vector<PRECISION>> CouplingMatrixData(UPS, vector<PRECISION>(UPS));
	
	Logspace(Frequency, FRMIN, FRMAX, FRRES); // kHz
	Linspace(Amplitude, PAMIN, PAMAX, PARES); // bar
	
	// SELECT CUDA DEVICE
	ListCUDADevices();
	PrintPropertiesOfSpecificDevice(DEVICE);
	
	// DEFINE PROBLEM SOLVER AND ITS OPTIONS ----
	ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION> ScanSystem(DEVICE);
	
	ScanSystem.SolverOption(SharedCouplingMatrices, 0);
	ScanSystem.SolverOption(InitialTimeStep, 1e-3);
	ScanSystem.SolverOption(RelativeTolerance, 0, 1e-9);
	ScanSystem.SolverOption(AbsoluteTolerance, 0, 1e-9);
	ScanSystem.SolverOption(MinimumTimeStep, 1e-13);
	ScanSystem.SolverOption(EventDirection, 0, -1);
	ScanSystem.SolverOption(EventTolerance, 0, 1e-4);
	
	// FILL SOLVER OBJECT WITH DATA -------------
	LoadBubbleSizes(BubbleSize, "BubbleSizes_N32_D1_d150_RE2-8.txt", UPS);
	LoadCouplingMatrix(CouplingMatrixData, "CouplingMatrix_N32_D1_d150_RE2-8.txt", UPS);
	
	FillSolverObject(ScanSystem, Frequency, Amplitude, BubbleSize);
	FillCouplingMatrix(ScanSystem, CouplingMatrixData, UPS);
	
	//int sid = 302;
	//cout << "PA: " << ScanSystem.GetHost<PRECISION>(sid, 0, UnitParameters, 15) << endl;
	//cout << "f: " << ScanSystem.GetHost<PRECISION>(sid, 0, UnitParameters, 16) << endl;
	
	// DATA FILENAME GENERATION -----------------
	ofstream DataFile;
	string Filename;
	stringstream StreamFilename;
	StreamFilename.str("");
	StreamFilename.setf(ios::fixed);
	
	StreamFilename.precision(1);
	StreamFilename << "CollapseStrength_N32_CSTR" << CSTR;
	StreamFilename << "_PA" << PAMIN << "_" << PAMAX << "_" << PARES;
	StreamFilename.precision(0);
	StreamFilename << "_F" << FRMIN << "_" << FRMAX << "_" << FRRES << ".txt";
	
	Filename = StreamFilename.str();
	
	// TRANSIENTS -------------------------------
	clock_t SimulationStart = clock();
	ScanSystem.SynchroniseFromHostToDevice(All);
	for (int i=0; i<NTRN; i++)
	{
		ScanSystem.Solve();
		ScanSystem.InsertSynchronisationPoint();
		ScanSystem.SynchroniseSolver();
		
		cout << "Transient iteration finished: " << i << endl;
	}
	clock_t SimulationEnd = clock();
	cout << "Transient simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
	// RESET ACCESSORIES ------------------------
	ResetAccessories(ScanSystem, Frequency, Amplitude, BubbleSize);
	ScanSystem.SynchroniseFromHostToDevice(UnitAccessories);
	ScanSystem.SynchroniseFromHostToDevice(IntegerUnitAccessories);
	
	// REGISTER COLLAPSES -----------------------
	for (int i=0; i<NCOLL; i++)
	{
		ScanSystem.Solve();
		ScanSystem.InsertSynchronisationPoint();
		ScanSystem.SynchroniseSolver();
		
		cout << "Converged iteration finished: " << i << endl;
	}
	
	ScanSystem.SynchroniseFromDeviceToHost(All);
	ScanSystem.InsertSynchronisationPoint();
	ScanSystem.SynchroniseSolver();
	
	// QUICK CHECK ------------------------------
	/*int sid = 57;
	int uid = 6;
	
	cout.precision(10);
	cout.flags(ios::scientific);
	cout << "Pressure amplitude: " << ScanSystem.GetHost<PRECISION>(sid, uid, UnitParameters, 15) << endl;
	cout << "Frequency:          " << ScanSystem.GetHost<PRECISION>(sid, uid, UnitParameters, 16) << endl;
	
	cout << "End state: " << endl;
	cout << ", X0: " << ScanSystem.GetHost<PRECISION>(sid, uid, ActualState, 0) << ", X1: " << ScanSystem.GetHost<PRECISION>(sid, uid, ActualState, 1) << endl;
	
	cout << "Local maxima and their corresponding minima: " << endl;
	for (int i=0; i<NCOLL; i++)
	{
		cout << "N: " << i;
		cout << ", MAX: " << ScanSystem.GetHost<PRECISION>(sid, uid, UnitAccessories, i) << ", MIN: " << ScanSystem.GetHost<PRECISION>(sid, uid, UnitAccessories, i+NCOLL);
		cout << endl;
	}*/
	
	// SAVE COLLAPSE DATA -----------------------
	DataFile.open ( Filename.c_str() );
	DataFile.flags(ios::scientific);
	
	for (int sid=0; sid<NS; sid++)
	{
		for (int cnum=0; cnum<NCOLL; cnum++)
		{
			DataFile.width(4); DataFile << sid << ',';
			DataFile.width(18); DataFile.precision(10); DataFile << ScanSystem.GetHost<PRECISION>(sid, 0, UnitParameters, 15) << ',';
			DataFile.width(18); DataFile.precision(10); DataFile << ScanSystem.GetHost<PRECISION>(sid, 0, UnitParameters, 16) << ',';
			
			for (int uid=0; uid<UPS; uid++)
			{
				DataFile.width(11); DataFile.precision(3); DataFile << ScanSystem.GetHost<PRECISION>(sid, uid, UnitAccessories, cnum) << ',';
			}
			
			for (int uid=0; uid<UPS; uid++)
			{
				DataFile.width(11); DataFile.precision(3); DataFile << ScanSystem.GetHost<PRECISION>(sid, uid, UnitAccessories, cnum+NCOLL) << ',';
			}
			
			DataFile << '\n';
		}
	}
	DataFile.close();
	
	// FINALISATION -----------------------------
	SimulationEnd = clock();
	cout << "Total simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
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

void FillSolverObject(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& Frequency, const vector<PRECISION>& Amplitude, const vector<PRECISION>& BubbleSize)
{
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
				
				Solver.SetHost(SystemNumber, UnitNumber, ActualState, 0, 1.0);
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
				
				Solver.SetHost(SystemNumber, UnitNumber, IntegerUnitAccessories, 0, -1 );
				
				UnitNumber++;
			}
			
			SystemNumber++;
		}
	}
	
	// GLOBAL SCOPE
	for (int i=0; i<NC; i++)
		Solver.SetHost(CouplingIndex, i, 1);
}

void ResetAccessories(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& Frequency, const vector<PRECISION>& Amplitude, const vector<PRECISION>& BubbleSize)
{
	// Loop over the systems (blocks): control parameters
	int SystemNumber = 0;
	for (auto const& CP2: Frequency) // Frequency [kHz]
	{
		for (auto const& CP1: Amplitude) // Amplitude [bar]
		{
			// UNIT SCOPE
			int UnitNumber = 0;
			for (auto const& CP0: BubbleSize) // equilibrium radius  [mum]
			{
				for (int cmp=0; cmp<2*NCOLL; cmp++)
					Solver.SetHost(SystemNumber, UnitNumber, UnitAccessories, cmp, 0.0 );
				
				Solver.SetHost(SystemNumber, UnitNumber, IntegerUnitAccessories, 0, -1 );
				
				UnitNumber++;
			}
			
			SystemNumber++;
		}
	}
}

void FillCouplingMatrix(ProblemSolver<NS,UPS,UD,TPB,SPB,NC,CBW,CCI,NUP,NSP,NGP,NiGP,NUA,NiUA,NSA,NiSA,NE,NDO,SOLVER,PRECISION>& Solver, const vector<vector<PRECISION>>& CouplingMatrixData, int UPS)
{
	for (int Row=0; Row<UPS; Row++)
	{
		for (int Col=0; Col<UPS; Col++)
		{
			// Full Matrix Storage extended with coupling strength
			Solver.SetHost(0, CouplingMatrix, Row, Col, CSTR*CouplingMatrixData[Row][Col]);
		}
	}
}