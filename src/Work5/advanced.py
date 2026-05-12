import taichi as ti # type: ignore

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
samples_per_pixel = ti.field(ti.i32, shape=())

MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

IOR_AIR = 1.0
IOR_GLASS = 1.5

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    
    if delta > 0:
        sqrt_delta = ti.sqrt(delta)
        t1 = (-b - sqrt_delta) / 2.0
        t2 = (-b + sqrt_delta) / 2.0
        
        if t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        
        if t > 0:
            p = ro + rd * t
            normal = normalize(p - center)
    
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    t, n = intersect_sphere(ro, rd, ti.Vector([-1.5, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.95, 0.95, 0.95])
        hit_mat = MAT_GLASS

    t, n = intersect_sphere(ro, rd, ti.Vector([1.5, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])
    spp = samples_per_pixel[None]

    for i, j in pixels:
        final_color = ti.Vector([0.0, 0.0, 0.0])
        
        for sample in range(spp):
            u_offset = (ti.random() - 0.5) / res_y * 2.0
            v_offset = (ti.random() - 0.5) / res_y * 2.0
            
            u = (i - res_x / 2.0) / res_y * 2.0 + u_offset
            v = (j - res_y / 2.0) / res_y * 2.0 + v_offset
            
            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            pixel_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            inside_glass = 0
            
            for bounce in range(max_bounces[None]):
                t, N, obj_color, mat_id = scene_intersect(ro, rd)
                
                if t > 1e9:
                    pixel_color += throughput * bg_color
                    break
                
                p = ro + rd * t
                
                if mat_id == MAT_MIRROR:
                    ro = p + N * 1e-4
                    rd = normalize(reflect(rd, N))
                    throughput *= 0.85
                
                elif mat_id == MAT_DIFFUSE:
                    L = normalize(light_pos - p)
                    shadow_ray_orig = p + N * 1e-4
                    shadow_result = scene_intersect(shadow_ray_orig, L)
                    shadow_t = shadow_result[0]
                    
                    dist_to_light = (light_pos - p).norm()
                    in_shadow = 0.0
                    if shadow_t < dist_to_light:
                        in_shadow = 1.0
                    
                    ambient = 0.2 * obj_color
                    direct_light = ambient
                    
                    if in_shadow == 0.0:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.8 * diff * obj_color
                        direct_light += diffuse
                    
                    pixel_color += throughput * direct_light
                    break
                
                elif mat_id == MAT_GLASS:
                    cos_theta = (-rd).dot(N)
                    eta_ratio = IOR_AIR / IOR_GLASS
                    
                    if inside_glass == 1:
                        eta_ratio = IOR_GLASS / IOR_AIR
                    
                    sin_theta_t_sq = eta_ratio * eta_ratio * (1.0 - cos_theta * cos_theta)
                    
                    if sin_theta_t_sq <= 1.0:
                        cos_theta_t = ti.sqrt(1.0 - sin_theta_t_sq)
                        rd = eta_ratio * (-rd) + (eta_ratio * cos_theta - cos_theta_t) * N
                        rd = normalize(rd)
                        inside_glass = 1 - inside_glass
                    else:
                        rd = normalize(reflect(rd, N))
                    
                    ro = p + N * 1e-4
            
            final_color += pixel_color
        
        pixels[i, j] = ti.math.clamp(final_color / spp, 0.0, 1.0)

def main():
    window = ti.ui.Window("Advanced Ray Tracing", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 3
    samples_per_pixel[None] = 4

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.35):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)
            samples_per_pixel[None] = gui.slider_int('MSAA Samples', samples_per_pixel[None], 1, 16)

        window.show()

if __name__ == '__main__':
    main()
