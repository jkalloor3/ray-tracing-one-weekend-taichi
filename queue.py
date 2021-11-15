import taichi as ti
from taichi_glsl.vector import vec2


@ti.func
def at(origin, direction, t):
    return origin + direction * t


@ti.data_oriented
class Queue:
    ''' An queue of "in flight" rays'''

    def __init__(self, total_size):
        self.origin = ti.Vector.field(3, dtype=ti.f32)
        self.direction = ti.Vector.field(3, dtype=ti.f32)
        self.depth = ti.field(ti.i32)
        self.attenuation = ti.Vector.field(3, dtype=ti.f32)
        self.screen_coord = ti.Vector.field(2, dtype=ti.i32)
        ti.root.dense(ti.i, (total_size)).place(self.screen_coord, self.origin, self.direction,
                                                self.depth, self.attenuation)
        self.read_idx = ti.field(dtype=ti.i32, shape=())
        self.write_idx = ti.field(dtype=ti.i32, shape=())
        self.read_idx[()] = 0
        self.write_idx[()] = 0

    ''' Pushing an entry to the queue, returns the index '''
    @ti.func
    def push(self) -> ti.i32:
        ind = ti.atomic_add(self.write_idx[()], 1)
        return ind

    ''' Pop an entry from the queue, returns the index '''
    @ti.func
    def pop(self) -> ti.i32:
        ind = ti.atomic_add(self.read_idx[()], 1)
        return ind

    @ti.pyfunc
    def is_empty(self) -> bool:
        return self.write_idx[()] == self.read_idx[()]

    @ti.func
    def set(self, queue_ind, x, y, ray_org, ray_dir, depth, attenuation):
        self.screen_coord[queue_ind] = vec2(x, y)
        self.origin[queue_ind] = ray_org
        self.direction[queue_ind] = ray_dir
        self.depth[queue_ind] = depth
        self.attenuation[queue_ind] = attenuation

    @ti.func
    def get(self, queue_ind):
        return self.screen_coord[queue_ind], self.origin[queue_ind], self.direction[queue_ind], self.depth[queue_ind], self.attenuation[queue_ind]

    @ti.func
    def get_od(self, queue_ind):
        return self.origin[queue_ind], self.direction[queue_ind]

    @ti.func
    def get_depth(self, queue_ind):
        return self.depth[queue_ind]

    @ti.func
    def set_depth(self, queue_ind, d):
        self.depth[queue_ind] = d
