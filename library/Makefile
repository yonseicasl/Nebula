############################################
# Developed by                             #
# Intelligent Computing Systems Lab (ICSL) #
# Yonsei University, Seoul, Korea          #
############################################

HDR=$(wildcard $(DIR)/common/*.h)
HDR+=$(wildcard $(DIR)/models/networks/*.h)
HDR+=$(wildcard $(DIR)/models/layers/*.h)
CCSRC=$(wildcard $(DIR)/common/*.cc)
CCSRC+=$(wildcard $(DIR)/models/networks/*.cc)
CCSRC+=$(wildcard $(DIR)/models/layers/*.cc)
CCOBJ=$(CCSRC:.cc=.o)

default: $(LIB)

$(LIB): $(CCOBJ) $(CUOBJ)
	$(CC) -g -shared -Wl,-soname,$@ -o $@ $(CCOBJ) $(CUOBJ)
	

%.o: %.cc $(HDR)
	$(CC) $(CCFLAG) $(STD) -o $@ -c $<

clean:
	rm -f $(CCOBJ) $(CUOBJ) $(LIB)

