#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

JOB1_DEVICE=0
JOB1_CSTR=0.0

JOB2_DEVICE=1
JOB2_CSTR=0.1

JOB3_DEVICE=0
JOB3_CSTR=0.2

JOB4_DEVICE=1
JOB4_CSTR=0.5

JOB5_DEVICE=0
JOB5_CSTR=1.0

JOB6_DEVICE=1
JOB6_CSTR=2.0

JOB7_DEVICE=0
JOB7_CSTR=3.0

JOB8_DEVICE=1
JOB8_CSTR=4.0

JOB9_DEVICE=0
JOB9_CSTR=6.0

JOB10_DEVICE=1
JOB10_CSTR=8.0

echo "Start job generation"

# Job1 ------------------------------------------
sed -i '24 s/#define CSTR.*/#define CSTR '$JOB1_CSTR'/' CollapseStrength.cu
sed -i '25 s/#define DEVICE.*/#define DEVICE '$JOB1_DEVICE'/' CollapseStrength.cu

rm -f Job1.exe
nvcc -o Job1.exe CollapseStrength.cu -I$HOME/BubbleEnsemble_ImplicitExtension/SourceCodes -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

qsub -cwd -o Job1.out -e Job1.err -q taranis-gpu0.q -N Job1 Job1.sh

# Job2 ------------------------------------------
sed -i '24 s/#define CSTR.*/#define CSTR '$JOB2_CSTR'/' CollapseStrength.cu
sed -i '25 s/#define DEVICE.*/#define DEVICE '$JOB2_DEVICE'/' CollapseStrength.cu

rm -f Job2.exe
nvcc -o Job2.exe CollapseStrength.cu -I$HOME/BubbleEnsemble_ImplicitExtension/SourceCodes -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

qsub -cwd -o Job2.out -e Job2.err -q taranis-gpu1.q -N Job2 Job2.sh

# Job3 ------------------------------------------
sed -i '24 s/#define CSTR.*/#define CSTR '$JOB3_CSTR'/' CollapseStrength.cu
sed -i '25 s/#define DEVICE.*/#define DEVICE '$JOB3_DEVICE'/' CollapseStrength.cu

rm -f Job3.exe
nvcc -o Job3.exe CollapseStrength.cu -I$HOME/BubbleEnsemble_ImplicitExtension/SourceCodes -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

qsub -cwd -o Job3.out -e Job3.err -q taranis-gpu0.q -N Job3 Job3.sh

# Job4 ------------------------------------------
sed -i '24 s/#define CSTR.*/#define CSTR '$JOB4_CSTR'/' CollapseStrength.cu
sed -i '25 s/#define DEVICE.*/#define DEVICE '$JOB4_DEVICE'/' CollapseStrength.cu

rm -f Job4.exe
nvcc -o Job4.exe CollapseStrength.cu -I$HOME/BubbleEnsemble_ImplicitExtension/SourceCodes -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

qsub -cwd -o Job4.out -e Job4.err -q taranis-gpu1.q -N Job4 Job4.sh

# Job5 ------------------------------------------
sed -i '24 s/#define CSTR.*/#define CSTR '$JOB5_CSTR'/' CollapseStrength.cu
sed -i '25 s/#define DEVICE.*/#define DEVICE '$JOB5_DEVICE'/' CollapseStrength.cu

rm -f Job5.exe
nvcc -o Job5.exe CollapseStrength.cu -I$HOME/BubbleEnsemble_ImplicitExtension/SourceCodes -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

qsub -cwd -o Job5.out -e Job5.err -q taranis-gpu0.q -N Job5 Job5.sh

# Job6 ------------------------------------------
sed -i '24 s/#define CSTR.*/#define CSTR '$JOB6_CSTR'/' CollapseStrength.cu
sed -i '25 s/#define DEVICE.*/#define DEVICE '$JOB6_DEVICE'/' CollapseStrength.cu

rm -f Job6.exe
nvcc -o Job6.exe CollapseStrength.cu -I$HOME/BubbleEnsemble_ImplicitExtension/SourceCodes -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

qsub -cwd -o Job6.out -e Job6.err -q taranis-gpu1.q -N Job6 Job6.sh

# Job7 ------------------------------------------
sed -i '24 s/#define CSTR.*/#define CSTR '$JOB7_CSTR'/' CollapseStrength.cu
sed -i '25 s/#define DEVICE.*/#define DEVICE '$JOB7_DEVICE'/' CollapseStrength.cu

rm -f Job7.exe
nvcc -o Job7.exe CollapseStrength.cu -I$HOME/BubbleEnsemble_ImplicitExtension/SourceCodes -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

qsub -cwd -o Job7.out -e Job7.err -q taranis-gpu0.q -N Job7 Job7.sh

# Job8 ------------------------------------------
sed -i '24 s/#define CSTR.*/#define CSTR '$JOB8_CSTR'/' CollapseStrength.cu
sed -i '25 s/#define DEVICE.*/#define DEVICE '$JOB8_DEVICE'/' CollapseStrength.cu

rm -f Job8.exe
nvcc -o Job8.exe CollapseStrength.cu -I$HOME/BubbleEnsemble_ImplicitExtension/SourceCodes -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

qsub -cwd -o Job8.out -e Job8.err -q taranis-gpu1.q -N Job8 Job8.sh

# Job9 ------------------------------------------
sed -i '24 s/#define CSTR.*/#define CSTR '$JOB9_CSTR'/' CollapseStrength.cu
sed -i '25 s/#define DEVICE.*/#define DEVICE '$JOB9_DEVICE'/' CollapseStrength.cu

rm -f Job9.exe
nvcc -o Job9.exe CollapseStrength.cu -I$HOME/BubbleEnsemble_ImplicitExtension/SourceCodes -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

qsub -cwd -o Job9.out -e Job9.err -q taranis-gpu0.q -N Job9 Job9.sh

# Job10 ------------------------------------------
sed -i '24 s/#define CSTR.*/#define CSTR '$JOB10_CSTR'/' CollapseStrength.cu
sed -i '25 s/#define DEVICE.*/#define DEVICE '$JOB10_DEVICE'/' CollapseStrength.cu

rm -f Job10.exe
nvcc -o Job10.exe CollapseStrength.cu -I$HOME/BubbleEnsemble_ImplicitExtension/SourceCodes -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

qsub -cwd -o Job10.out -e Job10.err -q taranis-gpu1.q -N Job10 Job10.sh