import numpy as np
import tkinter as tk


def create_sphere(segments):
    vertices = []
    indices = []

    for i in range(segments + 1):
        latitude = np.pi * i / segments
        for j in range(segments + 1):
            longitude = 2 * np.pi * j / segments
            vertices.append([
                np.sin(latitude) * np.cos(longitude),
                np.cos(latitude),
                np.sin(latitude) * np.sin(longitude),
            ])

    for i in range(segments):
        for j in range(segments):
            x = i * (segments + 1) + j
            y = x + segments + 1
            indices.extend([
                [x, y, x + 1],
                [y, y + 1, x + 1]
            ])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_cube():
    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [0.5, -0.5,  0.5],
        [0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ], dtype=np.float32)

    indices = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 2, 6], [1, 6, 5],
        [3, 2, 6], [3, 6, 7],
        [0, 1, 5], [0, 5, 4],
    ], dtype=np.uint32)

    return vertices, indices


def create_pyramid():
    vertices = np.array([
        [-0.5, -0.5, -0.5],  
        [0.5, -0.5, -0.5],   
        [0.0, -0.5, 0.5],  
        [0.0, 0.5, 0.0],     
    ], dtype=np.float32)

    indices = np.array([
        [0, 1, 2],
        [0, 1, 3], 
        [1, 2, 3],   
        [2, 0, 3],  
    ], dtype=np.uint32)

    return vertices, indices


def create_pyramid_with_square_base():
    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],   
        [0.5, -0.5, 0.5],    
        [-0.5, -0.5, 0.5], 
        [0.0, 0.5, 0.0],    
    ], dtype=np.float32)

    indices = np.array([
        [0, 1, 2], 
        [0, 2, 3], 
        [0, 1, 4],  
        [1, 2, 4],   
        [2, 3, 4],  
        [3, 0, 4], 
    ], dtype=np.uint32)

    return vertices, indices


def create_cone(slices):
    top_vertex = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    
    base_vertices = []
    for i in range(slices):
        angle = 2 * np.pi * i / slices
        x = 0.5 * np.cos(angle)
        z = 0.5 * np.sin(angle)
        base_vertices.append([x, -0.5, z])
    
    base_center = np.array([0.0, -0.5, 0.0], dtype=np.float32)
    
    vertices = np.array([top_vertex] + base_vertices + [base_center], dtype=np.float32)
    
    indices = []
    for i in range(slices):
        next_i = (i + 1) % slices
        indices.append([0, i + 1, next_i + 1])
    
    base_start_idx = slices + 1
    for i in range(slices):
        next_i = (i + 1) % slices
        indices.append([base_start_idx, i + 1, next_i + 1])
    
    return vertices, np.array(indices, dtype=np.uint32)



def get_translation_matrix(x, y, z):
    matrix = np.identity(4, dtype=np.float32)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix


def get_rotation_matrix(x, y, z):
    matrix_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(x), -np.sin(x), 0],
        [0, np.sin(x),  np.cos(x), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    matrix_y = np.array([
        [np.cos(y), 0, np.sin(y), 0],
        [0, 1, 0, 0],
        [-np.sin(y), 0, np.cos(y), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    matrix_z = np.array([
        [np.cos(z), -np.sin(z), 0, 0],
        [np.sin(z),  np.cos(z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    return matrix_z @ matrix_y @ matrix_x


def get_scaling_matrix(x, y, z):
    matrix = np.identity(4, dtype=np.float32)
    matrix[0, 0] = x
    matrix[1, 1] = y
    matrix[2, 2] = z
    return matrix


def get_projection_matrix(fov, aspect, near, far):
    focal = 1 / np.tan(np.radians(fov) / 2)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = focal / aspect
    matrix[1, 1] = focal
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2 * far * near) / (near - far)
    matrix[3, 2] = -1
    return matrix


def get_front(yaw, pitch):
    front = np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw)
    ], dtype=np.float32)
    return front / np.linalg.norm(front)


def get_view_matrix(eye, yaw, pitch):
    front = get_front(yaw, pitch)
    right = np.cross([0, 1, 0], front)
    right /= np.linalg.norm(right)
    up = np.cross(front, right)

    matrix = np.identity(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = up
    matrix[2, :3] = -front  # OpenGL standard...
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def add_to_scene(vertices, indices, model_matrix=np.identity(4, dtype=np.float32)):
    scene.append((vertices, indices, model_matrix))


WIDTH, HEIGHT = 1600, 900
FPS = 60
FRAME_TIME = int(1000 / FPS)
PROJECTION_MATRIX = get_projection_matrix(90, WIDTH / HEIGHT, 0.1, 100)
SENSETIVITY = 0.01
SPEED = 0.05

position = np.array([0, 0, 3], dtype=np.float32)
yaw, pitch = np.pi, 0.0
last_x, last_y = None, None
keys = {
    'w': False,
    's': False,
    'a': False,
    'd': False,
}
scene = []

add_to_scene(
    *create_sphere(16),
    get_translation_matrix(2.0, -3.0, 0.0) 
    @ get_rotation_matrix(np.pi/4, np.pi/2, np.pi/3)
    @ get_scaling_matrix(0.7, 0.5, 1.2)
)
add_to_scene(
    *create_cube(),
    get_translation_matrix(-5.0, 2.0, 0.0) 
    @ get_rotation_matrix(np.pi/6, np.pi/3, np.pi/4) 
    @ get_scaling_matrix(1.5, 0.5, 1.0)
)
add_to_scene(
    *create_pyramid_with_square_base(),
    get_translation_matrix(4.0, 3.0, 0.0) 
    @ get_rotation_matrix(np.pi/3, np.pi/2, np.pi/5) 
    @ get_scaling_matrix(0.8, 1.0, 0.5)
)
add_to_scene(
    *create_pyramid(),
    get_translation_matrix(-2.0, -1.0, 0.0) 
    @ get_rotation_matrix(np.pi/4, np.pi/6, np.pi/2) 
    @ get_scaling_matrix(1.0, 1.3, 0.6)
)
add_to_scene(
    *create_cone(30),
    get_translation_matrix(1.0, 5.0, 0.0) 
    @ get_rotation_matrix(np.pi/2, np.pi/4, np.pi/6) 
    @ get_scaling_matrix(1.0, 0.8, 1.5)
)


root = tk.Tk()
root.title("TkEnginer old demo")
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
canvas.pack()


def on_mouse_down(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y


def on_mouse_move(event):
    global yaw, pitch, last_x, last_y

    if last_x is None or last_y is None:
        last_x, last_y = event.x, event.y
        return

    dx = event.x - last_x
    dy = event.y - last_y

    yaw += dx * SENSETIVITY
    pitch -= dy * SENSETIVITY

    max_pitch = np.pi / 2 - 0.01
    pitch = max(-max_pitch, min(max_pitch, pitch))

    last_x, last_y = event.x, event.y


def on_key_press(event):
    if event.keysym == 'w':
        keys['w'] = True
    elif event.keysym == 's':
        keys['s'] = True
    elif event.keysym == 'a':
        keys['a'] = True
    elif event.keysym == 'd':
        keys['d'] = True


def on_key_release(event):
    if event.keysym == 'w':
        keys['w'] = False
    elif event.keysym == 's':
        keys['s'] = False
    elif event.keysym == 'a':
        keys['a'] = False
    elif event.keysym == 'd':
        keys['d'] = False


def on_update():
    global position

    canvas.delete("all")

    front = get_front(yaw, pitch)
    right = np.cross([0, 1, 0], front)
    right /= np.linalg.norm(right)
    if keys['w']:
        position += front * SPEED
    if keys['s']:
        position -= front * SPEED

    if keys['a']:
        position -= right * SPEED
    if keys['d']:
        position += right * SPEED
    view_matrix = get_view_matrix(position, yaw, pitch)

    for vertices, indices, model_matrix in scene:
        mvp_matrix = PROJECTION_MATRIX @ view_matrix @ model_matrix

        vertices_homogeneous = np.hstack(
            [vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)])

        vertices_clip = vertices_homogeneous @ mvp_matrix.T

        w_coords = vertices_clip[:, 3:4]

        vertices_ndc = vertices_clip[:, :3] / w_coords

        screen_coords = np.empty((len(vertices_ndc), 2), dtype=np.int32)
        screen_coords[:, 0] = (
            (vertices_ndc[:, 0] + 1) * 0.5 * WIDTH).astype(np.int32)
        screen_coords[:, 1] = (
            (1 - (vertices_ndc[:, 1] + 1) * 0.5) * HEIGHT).astype(np.int32)

        for triangle in indices:
            w0, w1, w2 = w_coords[triangle[0],
                                  0], w_coords[triangle[1], 0], w_coords[triangle[2], 0]
            if w0 <= 0 or w1 <= 0 or w2 <= 0:
                continue

            p0, p1, p2 = screen_coords[triangle[0]
                                       ], screen_coords[triangle[1]], screen_coords[triangle[2]]

            points = [
                p0[0], p0[1],
                p1[0], p1[1],
                p2[0], p2[1],
            ]
            canvas.create_polygon(points, outline="white", fill="")

    root.after(FRAME_TIME, on_update)


root.bind("<ButtonPress-1>", on_mouse_down)
root.bind("<B1-Motion>", on_mouse_move)
root.bind("<KeyPress>", on_key_press)
root.bind("<KeyRelease>", on_key_release)
on_update()

root.mainloop()
