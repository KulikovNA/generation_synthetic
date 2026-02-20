class D435NoiseModel:
    def __init__(
        self,
        fx: float,
        baseline_m: float,
        sigma_d_px: float = 0.5,
        depth_unit_mm: float = 1.0,
        z_min: float = 0.15,
        z_max: float = 3.0,
        p_hole_base: float = 0.002,
        p_hole_dark: float = 0.03,
        p_hole_lowtex: float = 0.03,
        p_flying: float = 0.2,
        dark_th: int = 20,
        lowtex_th: float = 8.0,
        edge_th: float = 0.02,
        seed: int = 0,
    ):
        self.fx = float(fx)
        self.B = float(baseline_m)
        self.sigma_d = float(sigma_d_px)
        self.depth_unit_mm = float(depth_unit_mm)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

        self.p_hole_base = float(p_hole_base)
        self.p_hole_dark = float(p_hole_dark)
        self.p_hole_lowtex = float(p_hole_lowtex)
        self.p_flying = float(p_flying)

        self.dark_th = int(dark_th)
        self.lowtex_th = float(lowtex_th)
        self.edge_th = float(edge_th)

        self.rng = np.random.default_rng(seed)

    def __call__(self, depth_m, rgb=None):
        # depth_m can be list or ndarray
        if isinstance(depth_m, list):
            return [self._one(d, rgb[i] if rgb is not None else None) for i, d in enumerate(depth_m)]
        if depth_m.ndim == 3:  # batch
            out = []
            for i in range(depth_m.shape[0]):
                out.append(self._one(depth_m[i], rgb[i] if rgb is not None else None))
            return np.stack(out, axis=0)
        return self._one(depth_m, rgb)

    def _one(self, depth_m: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
        z = depth_m.astype(np.float32, copy=True)
        z[~np.isfinite(z)] = 0.0

        # clamp range + valid
        valid = (z > self.z_min) & (z < self.z_max)

        # --- 1) disparity model: d = fx*B / z
        # avoid div by zero
        z_safe = np.where(valid, z, 1.0).astype(np.float32)
        disp = (self.fx * self.B) / z_safe  # pixels

        # --- 2) noise + quantization in disparity
        disp_noisy = disp + self.rng.normal(0.0, self.sigma_d, size=disp.shape).astype(np.float32)

        # optional subpixel quantization (coarse but effective)
        # e.g. 1/16 px steps
        q = 16.0
        disp_noisy = np.round(disp_noisy * q) / q

        # back to depth
        z_noisy = (self.fx * self.B) / np.maximum(disp_noisy, 1e-6)
        z_noisy[~valid] = 0.0

        # --- 3) compute "bad measurement" masks (dark/low texture/edges)
        if rgb is not None:
            rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
            if rgb8.ndim == 3 and rgb8.shape[2] == 4:
                rgb8 = rgb8[:, :, :3]
            gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)

            # low texture proxy: gradient magnitude
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad = cv2.magnitude(gx, gy)
            lowtex = grad < self.lowtex_th

            dark = gray < self.dark_th
        else:
            lowtex = np.zeros_like(valid, dtype=bool)
            dark = np.zeros_like(valid, dtype=bool)

        # edges in depth (proxy for occlusion boundaries -> flying pixels + holes)
        dzx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
        dzy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
        edge = (cv2.magnitude(dzx, dzy) > self.edge_th) & valid

        # --- 4) holes / invalids
        p = self.p_hole_base
        p_map = np.full(z.shape, p, dtype=np.float32)
        p_map += self.p_hole_dark * dark.astype(np.float32)
        p_map += self.p_hole_lowtex * lowtex.astype(np.float32)
        p_map += 0.02 * (z_noisy / max(self.z_max, 1e-6)).astype(np.float32)  # more holes far away
        p_map = np.clip(p_map, 0.0, 0.9)

        holes = (self.rng.random(z.shape) < p_map) & valid
        z_noisy[holes] = 0.0

        # --- 5) flying pixels on edges: swap with neighbor depth
        fly = (self.rng.random(z.shape) < self.p_flying) & edge
        if np.any(fly):
            # shift by 1 px in random direction
            dx = self.rng.integers(-1, 2, size=z.shape, endpoint=False)
            dy = self.rng.integers(-1, 2, size=z.shape, endpoint=False)
            yy, xx = np.indices(z.shape)
            xs = np.clip(xx + dx, 0, z.shape[1] - 1)
            ys = np.clip(yy + dy, 0, z.shape[0] - 1)
            z_noisy[fly] = z_noisy[ys[fly], xs[fly]]

        # --- 6) simple post-filtering (optional but looks closer to sensors)
        # median on valid region (keep zeros as zeros)
        z_f = z_noisy.copy()
        z_f_mm = (z_f * 1000.0).astype(np.float32)
        z_f_mm[z_f_mm <= 0] = np.nan
        # median requires dense; do cheap approach: median blur on mm image with zeros
        z_tmp = np.where(np.isfinite(z_f_mm), z_f_mm, 0.0).astype(np.float32)
        z_tmp = cv2.medianBlur(z_tmp, 3)
        z_f = np.where(z_noisy > 0, z_tmp / 1000.0, 0.0).astype(np.float32)

        # --- 7) quantize to depth units (RealSense-like)
        unit_m = self.depth_unit_mm / 1000.0
        if unit_m > 0:
            qz = np.round(z_f / unit_m) * unit_m
            z_f = np.where(z_f > 0, qz, 0.0).astype(np.float32)

        return z_f
