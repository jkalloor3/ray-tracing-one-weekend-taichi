import numpy as np
from taichi_glsl.experimental_array import dtype
import taichi as ti
from vector import *
import ray
from time import time
from hittable import World, Sphere
from camera import Camera
from material import *
import math
import random


# switch to cpu if needed
ti.init(arch=ti.gpu)

@ti.data_oriented
class Queue:
    def __init__(self, struct, size) -> None:
        self.struct_field = struct.field()
        self.struct_field_lock = ti.field(dtype=ti.i32)
        ti.root.dense(ti.i, (size)).place(self.struct_field)
        ti.root.dense(ti.i, (size)).place(self.struct_field_lock)
        self.read_idx = ti.field(dtype=ti.i32, shape=())
        self.write_idx = ti.field(dtype=ti.i32, shape=())
        self.in_flight_count = ti.field(dtype=ti.i32, shape=())
        self.size = size
        self.fault = ti.field(dtype=ti.i32, shape=())
        
    @ti.func
    def push(self, struct : ti.template()):
        idx = ti.atomic_add(self.write_idx[()], 1)
        self.struct_field[idx % self.size] = struct
        if self.struct_field_lock[idx % self.size] > 0:
            self.fault[()] = 1
        self.struct_field_lock[idx % self.size] = 1
        return idx

    @ti.kernel
    def clear_struct_field(self):
        for i in self.struct_field_lock:
            self.struct_field_lock[i] = 0

    def clear(self):
        self.read_idx[()] = 0
        self.write_idx[()] = 0
        self.in_flight_count[()] = 0
        self.clear_struct_field()

    @ti.func
    def reserve(self):
        idx = ti.atomic_add(self.read_idx[()], 1)
        return idx

    @ti.func
    def dequeue(self, idx : ti.i32) -> ti.template():
        s = self.struct_field[idx % self.size]
        self.struct_field_lock[idx % self.size] = 0
        return s

    @ti.func
    def get_read_idx(self) -> ti.i32:
        return ti.atomic_add(self.read_idx[()], 0)

    @ti.func
    def get_write_idx(self) -> ti.i32:
        return ti.atomic_add(self.write_idx[()], 0)

    @ti.func
    def has_data_arrived(self, idx):
        return self.struct_field_lock[idx % self.size] > 0

    @ti.func
    def increment_in_flight(self):
        self.in_flight_count[()] += 1

    @ti.func
    def decrement_in_flight(self):
        self.in_flight_count[()] -= 1

@ti.func
def get_background(dir):
    ''' Returns the background color for a given direction vector '''
    unit_direction = dir.normalized()
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * WHITE + t * BLUE


if __name__ == '__main__':
    random.seed(0)

    # image data
    aspect_ratio = 4.0 / 2.0
    image_width = 2048
    samples_per_pixel = 8
    max_depth = 32
    image_height = int(image_width / aspect_ratio)
    rays = ray.Rays(image_width, image_height)
    pixels = ti.Vector.field(3, dtype=float)
    sample_count = ti.field(dtype=ti.i32)
    ti.root.dense(ti.ij, (image_width, image_height)).place(sample_count)
    ti.root.dense(ti.ij, (image_width, image_height)).place(pixels)

    # materials
    mat_ground = Lambert([0.5, 0.5, 0.5])
    mat2 = Lambert([0.4, 0.2, 0.2])
    mat1 = Dielectric(1.5)
    mat3 = Metal([0.7, 0.6, 0.5], 0.0)

    # world
    R = math.cos(math.pi / 4.0)
    world = World()
    world.add(Sphere([0.0, -1000, 0], 1000.0, mat_ground))

    static_point = Point(4.0, 0.2, 0.0)
    for a in range(-15, 15):
        for b in range(-15, 15):
            choose_mat = random.random()
            center = Point(a + 0.9 * random.random(), 0.2,
                           b + 0.9 * random.random())

            if (center - static_point).norm() > 0.9:
                if choose_mat < 0.8:
                    # diffuse
                    mat = Lambert(
                        Color(random.random(), random.random(),
                              random.random())**2)
                elif choose_mat < 0.95:
                    # metal
                    mat = Metal(
                        Color(random.random(), random.random(),
                              random.random()) * 0.5 + 0.5,
                        random.random() * 0.5)
                else:
                    mat = Dielectric(1.5)

            world.add(Sphere(center, 0.2, mat))

    world.add(Sphere([0.0, 1.0, 0.0], 1.0, mat1))
    world.add(Sphere([-4.0, 1.0, 0.0], 1.0, mat2))
    world.add(Sphere([4.0, 1.0, 0.0], 1.0, mat3))
    world.commit()

    # camera
    vfrom = Point(13.0, 2.0, 3.0)
    at = Point(0.0, 0.0, 0.0)
    up = Vector(0.0, 1.0, 0.0)
    focus_dist = 10.0
    aperture = 0.1
    cam = Camera(vfrom, at, up, 20.0, aspect_ratio, aperture, focus_dist)

    start_attenuation = Vector(1.0, 1.0, 1.0)
    initial = True

    num_completed = ti.field(dtype=ti.i32, shape=())
    num_pixels = image_width * image_height

    ray_sample = ti.types.struct(
        org = ti.types.vector(3, ti.f32),
        dir = ti.types.vector(3, ti.f32),
        pdf = ti.types.vector(3, ti.f32),
        depth = ti.i32,
        x = ti.i32,
        y = ti.i32
    )

    ray_queue = Queue(ray_sample, image_width * image_height * 2)

    @ti.kernel
    def wavefront_initial() -> ti.i32:
        for x, y in pixels:
            # gen sample
            d = max_depth
            pdf = start_attenuation

            u = (x + ti.random()) / (image_width - 1)
            v = (y + ti.random()) / (image_height - 1)
            ray_org, ray_dir = cam.get_ray(u, v)
            # rays.set(x, y, ray_org, ray_dir, depth, pdf)
            next_ray = ray_sample(org = ray_org, dir = ray_dir, depth = d, x = x, y = y, pdf = pdf)
            ray_queue.push(next_ray)
            sample_count[x, y] = 0
            pixels[x, y] = Vector(0.0, 0.0, 0.0)

    @ti.kernel
    def finish():
        for x, y in pixels:
            pixels[x, y] = ti.sqrt(pixels[x, y] / samples_per_pixel)

    @ti.kernel
    def wavefront_queue():
        ''' Loops over pixels
            for each pixel:
                generate ray if needed
                intersect scene with ray
                if miss or last bounce sample backgound
            return pixels that hit max samples
        '''
        ti.block_dim(128)
        for i in range(128 * 12 * 32):
            slot = -1
            while True:
                # Reserve a slot on the queue
                if slot < 0:
                    slot = ray_queue.reserve()

                if num_completed[()] >= num_pixels:
                    break

                if not ray_queue.has_data_arrived(slot):
                    continue

                ray = ray_queue.dequeue(slot)
                slot = -1

                ray_queue.increment_in_flight()

                # gen sample
                ray_org, ray_dir, depth, pdf, x, y = ray.org, ray.dir, ray.depth, ray.pdf, ray.x, ray.y
                depth -= 1

                # intersect
                hit, p, n, front_facing, index = world.hit_all(
                    ray_org, ray_dir)

                if hit:
                    reflected, out_origin, out_direction, attenuation = world.materials.scatter(
                        index, ray_dir, p, n, front_facing)
                    next_ray = ray_sample(org = out_origin, dir = out_direction, depth = depth, x = x, y = y, pdf = pdf * attenuation)
                    ray_queue.push(next_ray)
                    ray_dir = out_direction

                if not hit or depth == 0:
                    pixels[x, y] += pdf * get_background(ray_dir)
                    sample_count[x, y] += 1
                    # num_completed[()] += 1
                    if sample_count[x, y] < samples_per_pixel:
                        u = (x + ti.random()) / (image_width - 1)
                        v = (y + ti.random()) / (image_height - 1)
                        ray_org, ray_dir = cam.get_ray(u, v)
                        next_ray = ray_sample(org = ray_org, dir = ray_dir, depth = max_depth, x = x, y = y, pdf = start_attenuation)
                        ray_queue.push(next_ray)
                    else:
                        num_completed[()] += 1

                ray_queue.decrement_in_flight()

    num_pixels = image_width * image_height

    num_to_do = wavefront_initial()
    wavefront_queue()
    finish()
    ridx = ray_queue.read_idx[()]
    widx = ray_queue.write_idx[()]
    print(ridx, widx, ray_queue.in_flight_count[()], ray_queue.fault[()])
    ray_queue.clear()
    num_completed[()] = 0
    ti.imwrite(pixels.to_numpy(), 'out_lockless_q.png')
    ti.sync()

    res = (512, 512)
    window = ti.ui.Window("Taichi MLS-MPM-128", res=res, vsync=False)

    while True:
        ray_queue.clear()
        num_completed[()] = 0
        print('starting big wavefront')
        t = time()
        num_to_do = wavefront_initial()
        wavefront_queue()
        ridx = ray_queue.read_idx[()]
        widx = ray_queue.write_idx[()]
        print(ridx, widx, ray_queue.in_flight_count[()], ray_queue.fault[()])
        finish()
        ti.sync()
        print(time() - t)
        window.show()
