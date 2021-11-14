import taichi as ti


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
        ti.root.dense(ti.i, (total_size)).place(self.origin, self.direction,
                                           self.depth, self.attenuation)

    @ti.func
    def set(self, queue_ind, ray_org, ray_dir, depth, attenuation):
        self.origin[queue_ind] = ray_org
        self.direction[queue_ind] = ray_dir
        self.depth[queue_ind] = depth
        self.attenuation[queue_ind] = attenuation

    @ti.func
    def get(self, queue_ind):
        return self.origin[queue_ind], self.direction[queue_ind], self.depth[queue_ind], self.attenuation[queue_ind]

    @ti.func
    def get_od(self, queue_ind):
        return self.origin[queue_ind], self.direction[queue_ind]

    @ti.func
    def get_depth(self, queue_ind):
        return self.depth[queue_ind]

    @ti.func
    def set_depth(self, queue_ind, d):
        self.depth[queue_ind] = d