#include "b.h"

int main() {
    Val* p = val_alloc(0.1);
    Val* q = val_alloc(0.2);

    Val*h = val_add(p,q);
    Val* e = val_mul(h, p);
    Val* x = val_tanh(e);
    Val*y = val_pow(x,2);
    
    //printf("ended:%f",y->data);
    // y->grad = 1.0;
    // backprop(y);
    // //
    // printf("zsrc0grad: %f\n", p->grad);

    // Remember to free the allocated memory

    Mat mat1 = mat_alloc(1,2);
    mat1.data[0] = *p;
    mat1.data[1] = *q;
    Mat mat2 = mat_alloc(2,1);
    mat2.data[0] = *p;
    mat2.data[1] = *q;

    Mat pp = mat_dot(mat2,mat1); 
    MAT_PRINT(pp);
    //mat_backward(pp);

    //MAT_PRINT_GRAD(pp);
    free(p);
    free(q);
    free(h);
    free(e);
    free(x);
    free(y);

    //MAT_PRINT(pp);

    //printf("mat_data[]0:%f",mat.data[0].data);
    return 0;
}
