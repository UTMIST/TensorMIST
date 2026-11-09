CC = gcc

.PHONY: test
test: src/tests/ops_test.c src/backend/ops.c include/ops.h
	$(CC) -Wall -Iinclude src/tests/ops_test.c src/backend/ops.c -o ops_test -lm
