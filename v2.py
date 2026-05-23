def render_prediction(snapshots, metrics):
    days = sorted(snapshots)
    tile = 170
    pad = 18
    left_label = 92
    top = 82
    chart_h = 250
    width = left_label + len(days) * (tile + pad) + pad
    height = top + 2 * (tile + 46) + chart_h + 70

    image = Image.new("RGB", (width, height), "#f7f3ec")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((left_label, 22), "KPP reaction-diffusion prediction for scabbed wound healing pattern", fill="#211f1c", font=font)
    draw.text((left_label, 44), "u_t = D * Laplacian(u) + r * u * (1-u); scab intensity fades as epithelial coverage rises", fill="#5a5147", font=font)
    draw.text((18, top + tile // 2), "open\nwound", fill="#211f1c", font=font)
    draw.text((18, top + tile + 46 + tile // 2), "scab", fill="#211f1c", font=font)

    for i, day in enumerate(days):
        u, scab = snapshots[day]
        x0 = left_label + i * (tile + pad)
        draw.text((x0 + 56, top - 22), f"Day {day}", fill="#211f1c", font=font)
        image.paste(to_heatmap(1 - u, "magma").resize((tile, tile), Image.Resampling.BICUBIC), (x0, top))
        image.paste(to_heatmap(scab, "scab").resize((tile, tile), Image.Resampling.BICUBIC), (x0, top + tile + 46))
        draw.rectangle((x0, top, x0 + tile, top + tile), outline="#d4c9bb")
        draw.rectangle((x0, top + tile + 46, x0 + tile, top + 2 * tile + 46), outline="#d4c9bb")

    chart_box = (left_label, top + 2 * (tile + 46) + 12, width - 38, height - 48)
    draw_chart(draw, chart_box, metrics, font)

    path = OUT_DIR / "kpp_wound_healing_pattern.png"
    image.save(path)

    csv_path = OUT_DIR / "kpp_wound_healing_metrics.csv"
    np.savetxt(
        csv_path,
        metrics,
        delimiter=",",
        header="day,open_wound_burden,center_closure,mean_wound_coverage",
        comments="",
    )
    return path, csv_path


def render_animation(animation_frames, metrics):
    frames = []
    width, height = 760, 430
    font = ImageFont.load_default()
    for day, u, scab in animation_frames:
        frame = Image.new("RGB", (width, height), "#f7f3ec")
        draw = ImageDraw.Draw(frame)
        draw.text((26, 18), "KPP wound scab healing pattern animation", fill="#211f1c", font=font)
        draw.text((26, 38), f"Day {day:04.1f}", fill="#5a5147", font=font)

        open_tile = to_heatmap(1 - u, "magma").resize((230, 230), Image.Resampling.BICUBIC)
        scab_tile = to_heatmap(scab, "scab").resize((230, 230), Image.Resampling.BICUBIC)
        frame.paste(open_tile, (26, 82))
        frame.paste(scab_tile, (282, 82))
        draw.rectangle((26, 82, 256, 312), outline="#d4c9bb")
        draw.rectangle((282, 82, 512, 312), outline="#d4c9bb")
        draw.text((26, 62), "open wound field", fill="#211f1c", font=font)
        draw.text((282, 62), "scab intensity", fill="#211f1c", font=font)

        day_index = np.searchsorted(metrics[:, 0], day, side="right") - 1
        day_index = max(0, min(day_index, len(metrics) - 1))
        partial_metrics = metrics[: day_index + 1]
        draw_progress_chart(draw, (540, 82, 734, 312), partial_metrics, metrics[-1, 0], font)

        current = metrics[day_index]
        labels = [
            ("center closure", current[2], "#116466"),
            ("mean coverage", current[3], "#d45d3a"),
            ("open burden", current[1], "#6d597a"),
        ]
        y = 337
        for label, value, color in labels:
            draw.line((36, y + 7, 66, y + 7), fill=color, width=4)
            draw.text((76, y), f"{label}: {value * 100:5.1f}%", fill="#211f1c", font=font)
            y += 22

        draw.text((540, 337), "u_t = D * Laplacian(u) + r * u * (1-u)", fill="#5a5147", font=font)
        draw.text((540, 359), "scab fades as epithelial coverage rises", fill="#5a5147", font=font)
        frames.append(frame)

    path = OUT_DIR / "kpp_wound_healing_animation.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    return path


def render_macro_skin_animation(animation_frames):
    skin_base = make_skin_texture(animation_frames[0][1].shape, seed=17)
    frames = []
    font = ImageFont.load_default()

    for day, u, scab in animation_frames:
        frame = render_macro_skin_frame(skin_base, u, scab, day, font)
        frames.append(frame)

    path = OUT_DIR / "kpp_macro_skin_surface_animation.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    return path


def make_skin_texture(shape, seed=0):
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.mgrid[:h, :w]

    base = np.zeros((h, w, 3), dtype=float)
    base[..., 0] = 222
    base[..., 1] = 168
    base[..., 2] = 138

    low_noise = rng.normal(0, 1, (24, 24))
    low_noise = Image.fromarray(((low_noise - low_noise.min()) / (np.ptp(low_noise)) * 255).astype(np.uint8))
    low_noise = low_noise.resize((w, h), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(2))
    low_noise = np.asarray(low_noise, dtype=float) / 255 - 0.5

    fine_noise = rng.normal(0, 1, (h, w))
    fine_noise = fine_noise / max(np.max(np.abs(fine_noise)), 1)
    skin_lines = 0.5 * np.sin(0.22 * x + 0.08 * y) + 0.35 * np.sin(0.06 * x - 0.28 * y + 1.2)
    pores = np.zeros((h, w), dtype=float)
    for _ in range(280):
        px = rng.integers(0, w)
        py = rng.integers(0, h)
        rr = rng.uniform(0.45, 1.1)
        pores += np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * rr * rr))
    pores = np.clip(pores, 0, 1)

    base += low_noise[..., None] * np.array([24, 15, 12])
    base += fine_noise[..., None] * np.array([5, 4, 4])
    base += skin_lines[..., None] * np.array([5, 3, 2])
    base -= pores[..., None] * np.array([42, 34, 28])
    return np.clip(base, 0, 255).astype(np.uint8)


def render_macro_skin_frame(skin_base, u, scab, day, font):
    open_wound = np.clip(1 - u, 0, 1)
    scab = np.clip(scab, 0, 1)
    scab_strength = np.clip(scab * 1.65, 0, 1)

    skin = skin_base.astype(float)
    open_mask = smooth_field(open_wound, 1.4)
    open_bed_mask = np.clip(open_mask * (1 - scab_strength * 0.82), 0, 1)
    scab_mask = np.maximum(scab_strength, smooth_field(scab_strength, 0.55) * 0.9)
    irritated = smooth_field(np.maximum(open_wound, scab_strength), 5.0)
    new_skin = np.clip((u - 0.65) * 2.2, 0, 1) * smooth_field(open_wound + scab, 6.0)

    red_halo_color = np.array([218, 96, 86])
    fresh_skin_color = np.array([238, 145, 132])
    wound_bed_color = np.array([170, 48, 55])
    moist_highlight = np.array([246, 183, 151])
    scab_color = np.array([84, 42, 22])
    scab_high = np.array([162, 92, 38])

    skin = blend(skin, red_halo_color, np.clip(irritated * 0.34, 0, 0.34))
    skin = blend(skin, fresh_skin_color, np.clip(new_skin * 0.42, 0, 0.42))
    skin = blend(skin, wound_bed_color, np.clip(open_bed_mask * 0.86, 0, 0.86))

    y, x = np.mgrid[:u.shape[0], :u.shape[1]]
    wet_texture = 0.5 + 0.5 * np.sin(0.35 * x - 0.18 * y + day * 0.3)
    skin = blend(skin, moist_highlight, np.clip(open_bed_mask * wet_texture * 0.22, 0, 0.22))

    crust_texture = np.clip(
        0.55
        + 0.24 * np.sin(0.31 * x + 0.12 * y)
        + 0.18 * np.cos(0.17 * x - 0.29 * y + 0.4)
        + 0.10 * np.sin(0.52 * x + 1.8),
        0,
        1,
    )
    crack_lines = (
        np.abs(np.sin(0.19 * x + 0.43 * y + 1.3)) < 0.055
    ) | (
        np.abs(np.sin(0.37 * x - 0.16 * y + 0.9)) < 0.04
    )
    crust_rgb = scab_color * (0.62 + crust_texture[..., None] * 0.58)
    crust_rgb = blend(crust_rgb, scab_high, crust_texture[..., None] * 0.22)
    crust_rgb = blend(crust_rgb, np.array([42, 21, 13]), (crack_lines & (scab_mask > 0.18))[..., None] * 0.42)
    skin = blend(skin, crust_rgb, np.clip(scab_mask * 0.96, 0, 0.96))

    height = smooth_field(scab_strength * 1.15 - open_wound * 0.28 + irritated * 0.12, 1.0)
    gy, gx = np.gradient(height)
    shade = np.clip(1 + gx * -1.8 + gy * -1.2, 0.72, 1.24)
    skin *= shade[..., None]

    surface = Image.fromarray(np.clip(skin, 0, 255).astype(np.uint8), "RGB")
    surface = surface.resize((720, 720), Image.Resampling.BICUBIC)
    surface = surface.filter(ImageFilter.UnsharpMask(radius=1.3, percent=115, threshold=3))

    canvas = Image.new("RGB", (760, 820), "#eee6dc")
    canvas.paste(surface, (20, 62))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), "KPP macro skin-surface pattern prediction", fill="#211f1c", font=font)
    draw.text((628, 20), f"Day {day:04.1f}", fill="#5a5147", font=font)
    draw.rectangle((20, 62, 740, 782), outline="#c8b9aa", width=2)
    draw.text((24, 792), "close surface view: pores, fine lines, irritation halo, open bed, drying scab", fill="#5a5147", font=font)
    return canvas


def smooth_field(values, radius):
    img = Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8), "L")
    img = img.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=float) / 255


def blend(base, overlay, alpha):
    if np.ndim(alpha) == 2:
        alpha = alpha[..., None]
    return base * (1 - alpha) + overlay * alpha


def draw_progress_chart(draw, box, partial_metrics, max_day, font):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill="#fffdf8", outline="#d4c9bb")
    px0, py0 = x0 + 32, y0 + 22
    px1, py1 = x1 - 10, y1 - 28
    draw.text((x0 + 10, y0 + 8), "closure curves", fill="#211f1c", font=font)
    for pct in range(0, 101, 25):
        y = py1 - (py1 - py0) * pct / 100
        draw.line((px0, y, px1, y), fill="#ece3d7")
    for day in range(0, int(max_day) + 1, 12):
        x = px0 + (px1 - px0) * day / max_day
        draw.line((x, py0, x, py1), fill="#f1e9df")
    series = [
        (partial_metrics[:, 2] * 100, "#116466"),
        (partial_metrics[:, 3] * 100, "#d45d3a"),
        (partial_metrics[:, 1] * 100, "#6d597a"),
    ]
    for values, color in series:
        points = []
        for day, value in zip(partial_metrics[:, 0], values):
            x = px0 + (px1 - px0) * day / max_day
            y = py1 - (py1 - py0) * value / 100
            points.append((x, y))
