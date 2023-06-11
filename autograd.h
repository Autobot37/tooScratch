#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//_________________________________
// Val Abstraction
// It fully WORKS backward compatibility with scalar engine.
typedef struct Val {
    float data;
    float grad;
    struct Val* src0;
    struct Val* src1;
    const char* ops;
} Val;

Val* val_alloc(float val) {
    Val* m = (Val*)malloc(sizeof(Val));
    m->data = val;
    m->grad = 0.0;
    m->src0 = NULL;
    m->src1 = NULL;
    return m;
}

Val* val_add(Val* x, Val* y) {
    Val* z = val_alloc(x->data + y->data);

    z->grad = 0.0;
    z->src0 = x;
    z->src1 = y;
    z->ops = "add";

    return z;
}

Val* val_mul(Val* x, Val* y) {
    Val* z = val_alloc(x->data * y->data);
    z->ops = "mul";

    z->grad = 0.0;
    z->src0 = x;
    z->src1 = y;

    return z;
}

Val* val_tanh(Val* x) {
    Val* z = val_alloc(tanhf(x->data));
    z->ops = "tanh";

    z->grad = 0.0;
    z->src0 = x;
    z->src1 = NULL;

    return z;
}

Val* val_exp(Val* x) {
    Val* z = val_alloc(expf(x->data));
    z->ops = "exp";

    z->grad = 0.0;
    z->src0 = x;
    z->src1 = NULL;

    return z;
}

Val* val_pow(Val* x, float power) {
    Val* z = val_alloc(powf(x->data, power));
    z->ops = "pow";

    z->grad = 0.0;
    z->src0 = x;

    Val *pow_w = val_alloc(power);
    z->src1 = pow_w;

    return z;
}

Val* val_negative(Val* x) {
    Val* z = val_alloc(-x->data);
    z->ops = "negative";

    z->grad = 0.0;
    z->src0 = x;
    z->src1 = NULL;

    return z;
}

Val* val_div(Val* x, Val* y) {
    Val* z = val_alloc(x->data / y->data);
    z->ops = "div";

    z->grad = 0.0;
    z->src0 = x;
    z->src1 = y;

    return z;
}

void backprop(Val* x) {
    if(x->src0 == NULL && x->src1 == NULL){
        return;
    }
    if (strcmp(x->ops, "add") == 0) {
        if (x->src0 != NULL) {
            x->src0->grad += x->grad;
        }
        if (x->src1 != NULL) {
            x->src1->grad += x->grad;
        }
    }else if (strcmp(x->ops, "mul") == 0) {
        if (x->src0 != NULL) {
            x->src0->grad += x->src1->data * x->grad;
        }
        if (x->src1 != NULL) {
            x->src1->grad += x->src0->data * x->grad;
        }
    }
    else if(strcmp(x->ops,"pow") == 0){
        if(x->src0 != NULL){    
            x->src0->grad += x->src1->data * powf(x->src0->data, x->src1->data - 1) * x->grad;        
        }
    }
    else if(strcmp(x->ops,"negative") == 0){
        if (x->src0 != NULL) {
            x->src0->grad -= x->grad;
        }
        if (x->src1 != NULL) {
            x->src1->grad -= x->grad;
        }
    }
    else if (strcmp(x->ops, "div") == 0) {
        if (x->src0 != NULL) {
            x->src0->grad += x->grad / x->src1->data;
        }
        if (x->src1 != NULL) {
            x->src1->grad -= (x->grad * x->src0->data) / (x->src1->data * x->src1->data);
        }
    }
    else if (strcmp(x->ops, "tanh") == 0) {
        if (x->src0 != NULL) {
            x->src0->grad += (1 - tanhf(x->src0->data) * tanhf(x->src0->data)) * x->grad;
        }
    }

    if(x->src0!=NULL){
        backprop(x->src0);
    }
    if(x->src1!=NULL){
        backprop(x->src1);
    }
}

//_______________________________________________
//_____________________________________________
// MATRIX ABSTRACTION CONSISTING OF MATRIX of VALUES.
// BUT IT DOESNT WORK IN MAT_DOT[BAD ABSTRACTION]
// BUT DUE TO JAX BEING DOING SOMETHING LIKE THIS MAYBE IT COULD WORK.

typedef struct Mat{
    int rows;
    int cols;
    Val* data;
} Mat;

Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = (Val*) malloc(sizeof(*m.data)*rows*cols);
    //assert(m.data != NULL);
    return m;
}

#define MAT_AT(m,i,j) m.data[(i)*(m).cols + (j)]



Mat mat_dot(Mat a, Mat b) {
    Mat dst = {
        .rows = a.rows,
        .cols = b.cols,
        .data = (Val*)malloc(sizeof(Val) * dst.rows * dst.cols)
    };

    // Assert that the number of columns in 'a' is equal to the number of rows in 'b'
    // assert(a.cols == b.rows);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            dst.data[i * dst.cols + j] = *val_alloc(0.0);
            dst.data[i * dst.cols + j].src0 = NULL;
            dst.data[i * dst.cols + j].src1 = NULL;
            dst.data[i * dst.cols + j].ops = "add";

            for (size_t k = 0; k < a.cols; k++) {
                Val val1 = a.data[i * a.cols + k];
                Val val2 = b.data[k * b.cols + j];

                if (k == 0) {
                    dst.data[i * dst.cols + j].src0 = val_mul(&val1, &val2);
                    dst.data[i * dst.cols + j].ops = "mul";
                    dst.data[i * dst.cols + j].src1 = &val2;
                } else {
                    Val mul_res = *val_mul(&val1, &val2);
                    mul_res.ops = "mul";
                    mul_res.src0 = &val1;
                    mul_res.src1 = &val2;
                    
                    Val* add_res = val_add(&dst.data[i * dst.cols + j], &mul_res);
                    add_res->ops = "add";
                    add_res->src0 = &dst.data[i * dst.cols + j];
                    add_res->src1 = &mul_res;
                    dst.data[i * dst.cols + j] = *add_res;
                }
            }
        }
    }

    return dst;
}



#define MAT_PRINT(m) mat_print(m,#m);


void mat_print(Mat m,const char* name){
    printf("%s = \n",name);
    for(size_t i = 0;i<m.rows;i++){
        printf("[");
        for(size_t j=0;j<m.cols;j++){
            printf("%f",MAT_AT(m,i,j).data);
            printf(" ");
        }
        {printf("]");}
        printf("\n");
    }
}

void mat_backward(Mat m){
    for(int i = 0;i<m.rows;i++){
        for(int j=0;j<m.cols;j++){
            backprop(&MAT_AT(m, i, j));
        }
    }
}

#define MAT_PRINT_GRAD(m) mat_print_grad(m,#m);


void mat_print_grad(Mat m,const char* name){
    printf("%s = \n",name);
    for(size_t i = 0;i<m.rows;i++){
        printf("[");
        for(size_t j=0;j<m.cols;j++){
            printf("%f",MAT_AT(m,i,j).grad);
            printf(" ");
        }
        {printf("]");}
        printf("\n");
    }
}
//___________________________________