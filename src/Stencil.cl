#define N 16
#define BLOCK_SIZE 8
#define RADIUS 3

kernel void stencil_1d(global int *in, global int *out) {
  local int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = get_global_id(0);
  int lindex = get_local_id(0) + RADIUS;
  temp[lindex] = in[gindex];
  if (get_local_id(0) < RADIUS) {
    temp[lindex - RADIUS] = (gindex >= RADIUS) ? in[gindex - RADIUS] : 0;
    temp[lindex + BLOCK_SIZE] =
        ((gindex + BLOCK_SIZE) < N) ? in[gindex + BLOCK_SIZE] : 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++)
    result += temp[lindex + offset];
  out[gindex] = result;
}
