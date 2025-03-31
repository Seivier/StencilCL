kernel void vectorAdd(global float *a, global float *b, global float *c,
                      float k, int n) {
  int gindex = get_global_id(0);
  if (gindex < n) {
    c[gindex] = a[gindex] * k + b[gindex];
  }
}
