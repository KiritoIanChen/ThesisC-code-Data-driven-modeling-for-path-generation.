import numpy as np


def gjk(shape1, shape2, iterations=10):
    # 初始化6个搜索方向
    directions = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    for v in directions:
        a, b = pick_line(v, shape2, shape1)
        a, b, c, flag = pick_triangle(a, b, shape2, shape1, iterations)
        if flag:
            break

    if flag:
        a, b, c, d, flag = pick_tetrahedron(a, b, c, shape2, shape1, iterations)

    return flag


def pick_line(v, shape1, shape2):
    b = support(shape2, shape1, v)
    a = support(shape2, shape1, -v)
    return a, b


def pick_triangle(a, b, shape1, shape2, max_iter):
    flag = False
    ab = b - a
    ao = -a
    v = np.cross(np.cross(ab, ao), ab)

    c = b
    b = a
    a = support(shape2, shape1, v)

    for _ in range(max_iter):
        ab = b - a
        ao = -a
        ac = c - a
        abc = np.cross(ab, ac)
        abp = np.cross(ab, abc)
        acp = np.cross(abc, ac)

        if np.dot(abp, ao) > 0:
            c = b
            b = a
            v = abp
        elif np.dot(acp, ao) > 0:
            b = a
            v = acp
        else:
            flag = True
            break
        a = support(shape2, shape1, v)

    return a, b, c, flag


def pick_tetrahedron(a, b, c, shape1, shape2, max_iter):
    flag = False
    ab = b - a
    ac = c - a
    abc = np.cross(ab, ac)
    ao = -a

    if np.dot(abc, ao) > 0:
        d = c
        c = b
        b = a
        v = abc
    else:
        d = b
        b = a
        v = -abc

    a = support(shape2, shape1, v)

    for _ in range(max_iter):
        ab = b - a
        ac = c - a
        ad = d - a
        ao = -a

        abc = np.cross(ab, ac)

        if np.dot(abc, ao) > 0:
            pass
        else:
            acd = np.cross(ac, ad)
            if np.dot(acd, ao) > 0:
                b, c = c, d
                ab, ac = ac, ad
                abc = acd
            elif np.dot(np.cross(ad, ab), ao) > 0:
                c, b = b, d
                ac, ab = ab, ad
                abc = np.cross(ad, ab)
            else:
                flag = True
                break

        if np.dot(abc, ao) > 0:
            d, c, b = c, b, a
            v = abc
        else:
            d, b = b, a
            v = -abc
        a = support(shape2, shape1, v)

    return a, b, c, d, flag


def support(shape1, shape2, v):
    p1 = get_farthest_in_dir(shape1, v)
    p2 = get_farthest_in_dir(shape2, -v)
    return p1 - p2


def get_farthest_in_dir(shape, v):
    X = np.array(shape['XData'])
    Y = np.array(shape['YData'])
    Z = np.array(shape['ZData'])
    dots = X * v[0] + Y * v[1] + Z * v[2]
    idx = np.unravel_index(np.argmax(dots), dots.shape)
    return np.array([X[idx], Y[idx], Z[idx]])
