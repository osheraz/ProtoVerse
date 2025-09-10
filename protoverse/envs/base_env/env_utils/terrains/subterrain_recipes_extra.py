import numpy as np

# All functions expect a SubTerrain with:
#  - height_field_raw (int16 [width, length])
#  - ceiling_field_raw (optional)
#  - horizontal_scale (m / cell), vertical_scale (m / unit)
#  - width, length (in cells)
# They write in-place into height_field_raw.


def pit_subterrain(terrain, depth, platform_size=1.0):
    d = int(depth / terrain.vertical_scale)
    half = int(platform_size / terrain.horizontal_scale / 2)
    cx, cy = terrain.width // 2, terrain.length // 2
    x1, x2 = cx - half, cx + half
    y1, y2 = cy - half, cy + half
    terrain.height_field_raw[x1:x2, y1:y2] = -d


def half_sloped_subterrain(terrain, wall_width=4.0, start2center=0.7, max_height=1.0):
    ww = max(int(wall_width / terrain.horizontal_scale), 1)
    mh = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.width // 2)
    xs = np.arange(slope_start, terrain.width)
    h2w = mh / ww
    heights = (h2w * (xs - slope_start)).clip(max=mh).astype(np.int16)
    terrain.height_field_raw[slope_start : terrain.width, :] = heights[:, None]


def half_platform_subterrain(terrain, start2center=2.0, max_height=1.0):
    mh = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.width // 2)
    terrain.height_field_raw[:, :] = mh
    terrain.height_field_raw[-slope_start:slope_start, -slope_start:slope_start] = 0


def gap_subterrain(terrain, gap_size, platform_size=1.0):
    g = int(gap_size / terrain.horizontal_scale)
    plat = int(platform_size / terrain.horizontal_scale)
    cx, cy = terrain.width // 2, terrain.length // 2
    x1 = (terrain.width - plat) // 2
    x2 = x1 + g
    y1 = (terrain.length - plat) // 2
    y2 = y1 + g
    terrain.height_field_raw[cx - x2 : cx + x2, cy - y2 : cy + y2] = -1000
    terrain.height_field_raw[cx - x1 : cx + x1, cy - y1 : cy + y1] = 0


def gap_parkour_subterrain(terrain, difficulty, platform_size=2.0):
    g = int((0.1 + 0.3 * difficulty) / terrain.horizontal_scale)
    plat = int(platform_size / terrain.horizontal_scale)
    cx, cy = terrain.width // 2, terrain.length // 2
    x1 = (terrain.width - plat) // 2
    x2 = x1 + g
    y1 = (terrain.length - plat) // 2
    y2 = y1 + g
    terrain.height_field_raw[cx - x2 : cx + x2, cy - y2 : cy + y2] = -400
    terrain.height_field_raw[cx - x1 : cx + x1, cy - y1 : cy + y1] = 0


def parkour_subterrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    x_range=(1.8, 1.9),
    y_range=(0.0, 0.1),
    z_range=(-0.2, 0.2),
    stone_len=1.0,
    stone_width=0.6,
    pad_width=0.1,
    pad_height=0.5,
    incline_height=0.1,
    last_incline_height=0.6,
    last_stone_len=1.6,
    pit_depth=(0.5, 1.0),
):
    hf = terrain.height_field_raw
    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    # fill pit base
    hf[:] = -round(np.random.uniform(*pit_depth) / vs)

    mid_y = terrain.length // 2
    stone_len_m = np.random.uniform(stone_len, stone_len)
    stone_len_i = 2 * round(stone_len_m / 2.0, 1)
    stone_len_i = round(stone_len_i / hs)
    dis_x_min = stone_len_i + round(x_range[0] / hs)
    dis_x_max = stone_len_i + round(x_range[1] / hs)
    dis_y_min = round(y_range[0] / hs)
    dis_y_max = round(y_range[1] / hs)

    plat = round(platform_len / hs)
    plat_h = round(platform_height / vs)
    hf[0:plat, :] = plat_h

    stone_w = round(stone_width / hs)
    last_stone_len_i = round(last_stone_len / hs)
    inc_h = round(incline_height / vs)
    last_inc_h = round(last_incline_height / vs)

    dis_x = plat - np.random.randint(dis_x_min, dis_x_max) + stone_len_i // 2
    left_right = np.random.randint(0, 2)
    dis_z = 0

    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos = 1 if left_right else -1
        dis_y = mid_y + pos * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len_i // 4
            heights = (
                np.tile(
                    np.linspace(-last_inc_h, last_inc_h, stone_w), (last_stone_len_i, 1)
                )
                * pos
            )
            hf[
                dis_x - last_stone_len_i // 2 : dis_x + last_stone_len_i // 2,
                dis_y - stone_w // 2 : dis_y + stone_w // 2,
            ] = (
                heights.astype(int) + dis_z
            )
        else:
            heights = (
                np.tile(np.linspace(-inc_h, inc_h, stone_w), (stone_len_i, 1)) * pos
            )
            hf[
                dis_x - stone_len_i // 2 : dis_x + stone_len_i // 2,
                dis_y - stone_w // 2 : dis_y + stone_w // 2,
            ] = (
                heights.astype(int) + dis_z
            )
        left_right = 1 - left_right

    final_platform_start = dis_x + last_stone_len_i // 2 + max(1, round(0.05 / hs))
    hf[final_platform_start:, :] = plat_h


def parkour_gap_subterrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_gaps=8,
    gap_size=0.3,
    x_range=(1.6, 2.4),
    y_range=(-1.2, 1.2),
    half_valid_width=(0.6, 1.2),
    gap_depth=(0.2, 1.0),
    pad_width=0.1,
    pad_height=0.5,
    flat=False,
):
    hf = terrain.height_field_raw
    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    mid_y = terrain.length // 2
    plat = round(platform_len / hs)
    hf[0:plat, :] = round(platform_height / vs)

    gap_i = round(gap_size / hs)
    dis_x_min = round(x_range[0] / hs) + gap_i
    dis_x_max = round(x_range[1] / hs) + gap_i
    dis_y_min, dis_y_max = round(y_range[0] / hs), round(y_range[1] / hs)
    half_w = round(np.random.uniform(*half_valid_width) / hs)
    gdepth = -round(np.random.uniform(*gap_depth) / vs)

    dis_x = plat
    last_dis_x = dis_x
    for _ in range(num_gaps):
        rx = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rx
        ry = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            hf[dis_x - gap_i // 2 : dis_x + gap_i // 2, :] = gdepth
        hf[last_dis_x:dis_x, : mid_y + ry - half_w] = gdepth
        hf[last_dis_x:dis_x, mid_y + ry + half_w :] = gdepth
        last_dis_x = dis_x


def parkour_hurdle_subterrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    stone_len=0.3,
    x_range=(1.5, 2.4),
    y_range=(-0.4, 0.4),
    half_valid_width=(0.4, 0.8),
    hurdle_height_range=(0.2, 0.3),
    pad_width=0.1,
    pad_height=0.5,
    flat=False,
):
    hf = terrain.height_field_raw
    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    mid_y = terrain.length // 2
    plat = round(platform_len / hs)
    hf[0:plat, :] = round(platform_height / vs)
    sl = round(stone_len / hs)
    dis_x_min, dis_x_max = round(x_range[0] / hs), round(x_range[1] / hs)
    dis_y_min, dis_y_max = round(y_range[0] / hs), round(y_range[1] / hs)
    half_w = round(np.random.uniform(*half_valid_width) / hs)
    hmin, hmax = round(hurdle_height_range[0] / vs), round(hurdle_height_range[1] / vs)

    dis_x = plat
    last_dis_x = dis_x
    for _ in range(num_stones):
        rx = np.random.randint(dis_x_min, dis_x_max)
        ry = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rx
        if not flat:
            hf[dis_x - sl // 2 : dis_x + sl // 2,] = np.random.randint(hmin, hmax)
            hf[dis_x - sl // 2 : dis_x + sl // 2, : mid_y + ry - half_w] = 0
            hf[dis_x - sl // 2 : dis_x + sl // 2, mid_y + ry + half_w :] = 0
        last_dis_x = dis_x


def parkour_step_subterrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    x_range=(0.2, 0.4),
    y_range=(-0.15, 0.15),
    half_valid_width=(0.45, 0.5),
    step_height=0.2,
    pad_width=0.1,
    pad_height=0.5,
):
    hf = terrain.height_field_raw
    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    mid_y = terrain.length // 2
    plat = round(platform_len / hs)
    hf[0:plat, :] = round(platform_height / vs)

    dis_x_min = round((x_range[0] + step_height) / hs)
    dis_x_max = round((x_range[1] + step_height) / hs)
    dis_y_min, dis_y_max = round(y_range[0] / hs), round(y_range[1] / hs)
    half_w = round(np.random.uniform(*half_valid_width) / hs)
    step_h = round(step_height / vs)

    dis_x = plat
    last_dis_x = dis_x
    stair_h = 0
    for i in range(num_stones):
        rx = np.random.randint(dis_x_min, dis_x_max)
        ry = np.random.randint(dis_y_min, dis_y_max)
        stair_h += (
            step_h if i < num_stones // 2 else (-step_h if i > num_stones // 2 else 0)
        )
        hf[dis_x : dis_x + rx,] = stair_h
        dis_x += rx
        hf[last_dis_x:dis_x, : mid_y + ry - half_w] = 0
        hf[last_dis_x:dis_x, mid_y + ry + half_w :] = 0
        last_dis_x = dis_x


def demo_subterrain(terrain):
    hs, vs = terrain.horizontal_scale, terrain.vertical_scale
    hf = terrain.height_field_raw
    mid_y = terrain.length // 2

    plat = round(2.0 / hs)
    h_depth = round(np.random.uniform(0.35, 0.4) / hs)
    h_height = round(np.random.uniform(0.3, 0.36) / vs)
    h_width = round(np.random.uniform(1.0, 1.2) / hs)
    hf[
        plat : plat + h_depth, round(mid_y - h_width / 2) : round(mid_y + h_width / 2)
    ] = h_height

    plat += round(np.random.uniform(1.5, 2.5) / hs)
    d1 = round(np.random.uniform(0.45, 0.8) / hs)
    h1 = round(np.random.uniform(0.35, 0.45) / vs)
    w1 = round(np.random.uniform(1.0, 1.2) / hs)
    hf[plat : plat + d1, round(mid_y - w1 / 2) : round(mid_y + w1 / 2)] = h1

    plat += d1 + round(np.random.uniform(0.45, 0.8) / hs)
    hf[plat : plat + d1, round(mid_y - w1 / 2) : round(mid_y + w1 / 2)] = h1

    plat += d1 + round(np.random.uniform(0.5, 0.8) / hs)
    d3 = round(np.random.uniform(0.25, 0.6) / hs)
    hf[plat : plat + d3, round(mid_y - w1 / 2) : round(mid_y + w1 / 2)] = h1
