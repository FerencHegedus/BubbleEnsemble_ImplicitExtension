INCL_DIR = -I$(HOME)/BubbleEnsemble_ImplicitExtension/SourceCodes
CMPL_OPT = -O3 --std=c++11 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage

all: CollapseStrength.exe

CollapseStrength.exe: CollapseStrength.cu
	nvcc -o	CollapseStrength.exe CollapseStrength.cu $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f CollapseStrength.exe