#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>

typedef struct Mat {
    size_t rows;
    size_t cols;
    float *es;
    struct Mat* grad;
    struct Mat* src0;
    struct Mat* src1;
    const char* ops;
} Mat;

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.es = (float*)malloc(rows * cols * sizeof(*m.es));
    memset(m.es, 0, rows * cols * sizeof(*m.es));

    m.grad = (Mat*)malloc(sizeof(Mat));
    m.grad->rows = rows;
    m.grad->cols = cols;
    m.grad->es = (float*)malloc(rows * cols * sizeof(*m.grad->es));
    memset(m.grad->es, 0, rows * cols * sizeof(*m.grad->es));
    
    m.src0 = NULL;
    m.src1 = NULL;
    m.ops = "None";

    return m;
}


#define MAT_AT(m,i,j) (m).es[(i)*(m).cols + (j)]

#define MAT_PRINT(m) mat_print(m,#m);

void mat_print(Mat m, const char* name) {
    printf("%s = \n", name);
    for (size_t i = 0; i < m.rows; i++) {
        printf("[");
        for (size_t j = 0; j < m.cols; j++) {
            printf("%f", MAT_AT(m, i, j));
            printf(" ");
        }
        printf("]");
        printf("\n");
    }
}

Mat mat_add(Mat a, Mat b) {
    Mat c = mat_alloc(a.rows, a.cols);
    c.src0 = &a;
    c.src1 = &b;
    c.ops = "add";

    for(size_t i=0;i<a.rows;i++){
    	for(size_t j=0;j<a.cols;j++){
    		MAT_AT(c,i,j) = MAT_AT(a,i,j) + MAT_AT(b,i,j); 
    	}
    }

    return c;
}

Mat mat_mul(Mat* a , Mat* b){
	Mat c = mat_alloc(a->rows,b->cols);

	c.src0 = a;
	c.src1 = b;
	c.ops = "mul";

	for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            float sum = 0.0;
            for (size_t k = 0; k < a->cols; k++) {
                sum += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
            MAT_AT(c, i, j) = sum;
        }
    }

    return c;
}

void mat_copy(Mat* dest, Mat* src) {
	
    assert(dest->rows == src->rows && dest->cols == src->cols); 
    size_t num_elements = src->rows * src->cols;
    memcpy(dest->es, src->es, num_elements * sizeof(float));
}

Mat mat_transpose(Mat *mat) {
    Mat result = mat_alloc(mat->cols, mat->rows);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            result.es[j * result.cols + i] = mat->es[i * mat->cols + j];
        }
    }
    return result;
}



void backprop(Mat* c) {
    if (strcmp(c->ops, "add") == 0) {
        if (c->src0 != NULL) {
            mat_copy(c->src0->grad, c->grad);
        }

        if (c->src1 != NULL) {
            assert(c->src1->rows == c->grad->rows && c->src1->cols == c->grad->cols);
            mat_copy(c->src1->grad, c->grad);
        }
    }

    // if (strcmp(c->ops, "mul") == 0) {
    //     if (c->src0 != NULL) {
    //         Mat out = mat_mul(*c->grad, mat_transpose(c->src1));
    //         MAT_PRINT(out);
	// 		printf("%zu %zu %zu %zu", c->src0->rows, out.rows, c->src0->cols, out.cols);            assert(c->src0->rows == out.rows && c->src0->cols == out.cols);
    //         mat_copy(c->src0->grad, &out);
    //         //mat_free(&out);
    //     }

    //     if (c->src1 != NULL) {
    //         Mat out = mat_mul(mat_transpose(c->src0), *c->grad);
    //         assert(c->src1->rows == out.rows && c->src1->cols == out.cols);
    //         mat_copy(c->src1->grad, &out);
    //         //mat_free(&out);
    //     }
    // }
}


int main() {
    float data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    
    float grad_s[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    Mat x = mat_alloc(2,3);
    x.es = data;
    //MAT_PRINT(x);
    Mat y = mat_alloc(3,2);
    y.es = data;
    //MAT_PRINT(y);
    printf("test:%f".y.grad->es);
    
    Mat z = mat_mul(x,y);
    z.grad->es = data;
    MAT_PRINT(z);
    //backprop(&z);
    // MAT_PRINT(*y.grad);
    // Mat yt = mat_transpose(&y);
    // MAT_PRINT(yt);
    return 0;
}
