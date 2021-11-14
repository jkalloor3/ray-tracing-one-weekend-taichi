import numpy as np
import taichi as ti
from vector import *
import ray
from queue import Queue
from time import time
from hittable import World, Sphere
from camera import Camera
from material import *
import math
import random


# switch to cpu if needed
ti.init(arch=ti.gpu)

@ti.func
def get_background(dir):
    ''' Returns the background color for a given direction vector '''
    unit_direction = dir.normalized()
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * WHITE + t * BLUE


if __name__ == '__main__':
    # image data
    aspect_ratio = 3.0 / 2.0
    image_width = 600
    samples_per_pixel = 50
    max_depth = 16
    image_height = int(image_width / aspect_ratio)
    rays = ray.Rays(image_width, image_height)
    sample_counts = ti.field(dtype=ti.i32)
    pixels = ti.Vector.field(3, dtype=float)
    # Queue will track x,y and sample field
    # inner_queue_1 = ti.Vector.field(2, dtype=ti.i32)
    inner_queue_1 = ti.field(ti.i32)
    # inner_queue_2 = ti.field(ti.i32)
    # Double buffer the queue
    ti.root.dynamic(ti.i, image_width*image_height*samples_per_pixel*max_depth).place(inner_queue_1)
    # ti.root.dynamic(ti.i, image_width * image_height).place(inner_queue_2)
    ti.root.dense(ti.ij,
                  (image_width, image_height)).place(pixels, sample_counts)

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
    for a in range(-11, 11):
        for b in range(-11, 11):
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

    num_completed = 0
    num_pixels = image_width * image_height

    @ti.kernel
    def finish():
        for x, y in pixels:
            pixels[x, y] = ti.sqrt(pixels[x, y] / samples_per_pixel)

    @ti.kernel
    def wavefront_initial() -> ti.i32:
        for x, y in pixels:
            # gen sample
            depth = max_depth
            pdf = start_attenuation

            u = (x + ti.random()) / (image_width - 1)
            v = (y + ti.random()) / (image_height - 1)
            ray_org, ray_dir = cam.get_ray(u, v)
            rays.set(x, y, ray_org, ray_dir, depth, pdf)
            sample_counts[x,y] = 0
            inner_queue_1[x*image_height + y] = x*image_height + y

        return image_height * image_width

    @ti.kernel
    def wavefront_queue_1(start: ti.i32, to_do: ti.i32) -> ti.i32:
        ''' Loops over pixels
            for each pixel:
                generate ray if needed
                intersect scene with ray
                if miss or last bounce sample backgound
            return pixels that hit max samples
        '''
        num_added = 0
        for ind in range(start, start + to_do):
            # print(ind)
            x = inner_queue_1[ind] // image_height
            y = inner_queue_1[ind] % image_height
            # print(x,y)
            sample_count = sample_counts[x,y]

            # gen sample
            ray_org, ray_dir, depth, pdf = rays.get(x, y)

            # intersect
            hit, p, n, front_facing, index = world.hit_all(ray_org, ray_dir)
            depth -= 1
            rays.depth[x, y] = depth
            if hit:
                reflected, out_origin, out_direction, attenuation = world.materials.scatter(
                    index, ray_dir, p, n, front_facing)
                rays.set(x, y, out_origin, out_direction, depth,
                         pdf * attenuation)
                ray_dir = out_direction

            if not hit or depth == 0:
                pixels[x, y] += pdf * get_background(ray_dir)
                u = (x + ti.random()) / (image_width - 1)
                v = (y + ti.random()) / (image_height - 1)
                depth = max_depth
                pdf = start_attenuation
                ray_org, ray_dir = cam.get_ray(u, v)
                rays.set(x, y, ray_org, ray_dir, depth, pdf)
                sample_count += 1

            sample_counts[x,y] = sample_count

            if sample_count < samples_per_pixel:
                g = x * image_height + y
                ti.append(inner_queue_1.parent(), [], g)
                num_added += 1

        return num_added

    num_pixels = image_width * image_height

    t = time()
    print('starting big wavefront')
    queue_start = 0
    next_queue_start = num_pixels
    loop_count = 0
    num_to_do = wavefront_initial()
    start = 0
    while num_to_do > 0:
        inn = inner_queue_1.to_numpy()
        old_end = start + num_to_do
        num_to_do = wavefront_queue_1(start, num_to_do)
        start = old_end

    finish()
    print(time() - t)
    ti.imwrite(pixels.to_numpy(), 'out_micro.png')
