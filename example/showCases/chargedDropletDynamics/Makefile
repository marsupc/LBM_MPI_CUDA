
SOBJS = \
main.cu

 






all: lbm

lbm: $(SOBJS)
	nvcc $(SOBJS) -I$(CUDA_INC) -I$(OMPI_INC) -L$(OMPI_LIB) -lmpi -L$(CUDA_LIB64) -lcuda -lcudart -o entrance

clean:
	rm -rf *.o all
