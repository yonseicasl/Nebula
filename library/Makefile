#######################################################
# Nebula: Lightweight Neural Network Benchmarks       #
# Developed by the Intelligent Computing Systems Lab  #
# School of Electrical Engineering, Yonsei University #
# Version: 1.4                                        #
#######################################################

HDR=$(wildcard $(DIR)/common/*.h)
HDR+=$(wildcard $(DIR)/models/networks/*.h)
HDR+=$(wildcard $(DIR)/models/layers/*.h)
CCSRC=$(wildcard $(DIR)/common/*.cc)
CCSRC+=$(wildcard $(DIR)/models/networks/*.cc)
CCSRC+=$(wildcard $(DIR)/models/layers/*.cc)
CCOBJ=$(CCSRC:.cc=.o)

ifeq ($(GPU_ENABLED), 1)
CUSRC=$(wildcard $(DIR)/common/*.cu)
CUSRC+=$(wildcard $(DIR)/models/networks/*.cu)
CUSRC+=$(wildcard $(DIR)/models/layers/*.cu)
CUOBJ=$(CUSRC:.cu=.ko)
endif

default: $(LIB)

$(LIB): $(CCOBJ) $(CUOBJ)
	ar rcs $@ $(CCOBJ) $(CUOBJ)

%.o: %.cc $(HDR)
	$(CC) $(CCFLAG) $(STD) -o $@ -c $<

%.ko: %.cu $(HDR)
	$(CU) $(CUARCH) --compiler-options "$(CCFLAG)" $(STD) -o $@ -c $<

clean:
	rm -f $(CCOBJ) $(CUOBJ) $(LIB)

