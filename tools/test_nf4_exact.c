test_nf4: tools/test_nf4.c src/wubu_nf4.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_nf4_exact: tools/test_nf4_exact.c src/wubu_nf4.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_spec_decode: tools/test_spec_decode.c src/wubu_spec_decode.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@