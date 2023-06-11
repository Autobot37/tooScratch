# ML from BareBone level.

It is a C/C++ library for dealing with ML contralibility and performance issues..

## Usage

```c
    Val* p = val_alloc(0.1);
    Val* q = val_alloc(0.2);

    Val*h = val_add(p,q);
    Val* e = val_mul(h, p);
    Val* x = val_tanh(e);
    Val*y = val_pow(x,2);
    
    printf("ended:%f",y->data);
    y->grad = 1.0;
    backprop(y);
    // //
    printf("zsrc0grad: %f\n", p->grad);

    // Remember to free the allocated memory

```

## Goals with that
to understand what happens at low level just above boundary of hardware interface.
understanding processor architecture currently CPU and nvidia GPU.

## Notes
cuda functions doesnt run without NVCC + VISUAL STUDIO
