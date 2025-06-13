import tkinter as tk
import cProfile


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c0, c1, t):
    return tuple(int(lerp(c0[i], c1[i], t)) for i in range(3))


def barycentric_interp_color(c0, c1, c2, u, v):
    w = 1 - u - v
    return tuple(int(c0[i]*w + c1[i]*u + c2[i]*v) for i in range(3))


def rgb_to_hex(c):
    return f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}'


def subdivide_triangle(p0, p1, p2, c0, c1, c2, steps, canvas):
    prof = cProfile.Profile()
    prof.enable()

    for i in range(steps):
        for j in range(steps - i):
            u0, v0 = i / steps, j / steps
            u1, v1 = (i + 1) / steps, j / steps
            u2, v2 = i / steps, (j + 1) / steps

            pA = (
                p0[0] * (1 - u0 - v0) + p1[0] * u0 + p2[0] * v0,
                p0[1] * (1 - u0 - v0) + p1[1] * u0 + p2[1] * v0,
            )
            pB = (
                p0[0] * (1 - u1 - v1) + p1[0] * u1 + p2[0] * v1,
                p0[1] * (1 - u1 - v1) + p1[1] * u1 + p2[1] * v1,
            )
            pC = (
                p0[0] * (1 - u2 - v2) + p1[0] * u2 + p2[0] * v2,
                p0[1] * (1 - u2 - v2) + p1[1] * u2 + p2[1] * v2,
            )
            cA = barycentric_interp_color(c0, c1, c2, u0, v0)
            cB = barycentric_interp_color(c0, c1, c2, u1, v1)
            cC = barycentric_interp_color(c0, c1, c2, u2, v2)

            canvas.create_polygon(
                pA[0], pA[1], pB[0], pB[1], pC[0], pC[1],
                fill=rgb_to_hex((
                    (cA[0] + cB[0] + cC[0]) // 3,
                    (cA[1] + cB[1] + cC[1]) // 3,
                    (cA[2] + cB[2] + cC[2]) // 3
                )),
                outline=""
            )

            if i + j + 1 < steps:
                u3, v3 = (i + 1) / steps, (j + 1) / steps

                pD = (
                    p0[0] * (1 - u3 - v3) + p1[0] * u3 + p2[0] * v3,
                    p0[1] * (1 - u3 - v3) + p1[1] * u3 + p2[1] * v3,
                )
                cD = barycentric_interp_color(c0, c1, c2, u3, v3)

                canvas.create_polygon(
                    pB[0], pB[1], pD[0], pD[1], pC[0], pC[1],
                    fill=rgb_to_hex((
                        (cB[0] + cC[0] + cD[0]) // 3,
                        (cB[1] + cC[1] + cD[1]) // 3,
                        (cB[2] + cC[2] + cD[2]) // 3
                    )),
                    outline=""
                )
    prof.disable()
    prof.print_stats("cumtime")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x600")

    c = tk.Canvas(root, width=600, height=600)
    c.pack()

    p0 = (100, 500)
    p1 = (500, 500)
    p2 = (300, 100)

    c0 = (255, 0, 0)
    c1 = (0, 255, 0)
    c2 = (0, 0, 255)
    subdivide_triangle(p0, p1, p2, c0, c1, c2, steps=5, canvas=c)

    root.mainloop()
