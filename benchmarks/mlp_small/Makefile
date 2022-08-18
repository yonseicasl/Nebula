#######################################################
# Nebula: Lightweight Neural Network Benchmarks       #
# Developed by the Intelligent Computing Systems Lab  #
# School of Electrical Engineering, Yonsei University #
# Version: 1.4                                        #
#######################################################

HDR=$(wildcard *.h)
CCSRC=$(wildcard *.cc)
CCOBJ=$(CCSRC:.cc=.o)

ifeq ($(GPU_ENABLED), 1)
CUSRC=$(wildcard *.cu)
CUOBJ=$(CUSRC:.cu=.ko)
endif

default: $(EXE)

$(EXE): $(CCOBJ) $(CUOBJ) $(LIB)
	$(LC) $(LDFLAG) -o $@ $(CCOBJ) $(CUOBJ) $(LIBFLAG)

%.o: %.cc $(HDR)
	$(CC) $(CCFLAG) $(STD) -o $@ -c $<

%.ko: %.cu $(HDR)
	$(CU) $(CUARCH) --compiler-options "$(CCFLAG)" $(STD) -o $@ -c $<

clean:
	rm -f $(CCOBJ) $(CUOBJ) $(EXE)

