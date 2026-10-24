"""Ego trajectory for MUHOS: dead reckoning anchored to the GPS.

MUHOS has no pose product; the drive is recorded by a consumer GPS (PCAN-GPS) and the
car's CAN bus. Measured on the data (``datasets/muhos/README.md``, *Ego trajectory*):

* the bus logger's clock (CAN and GPS logs alike) is not the sensors' clock: it stamps
  an event 0.3-1.3 s before the lidar and the cameras record it, and the offset drifts
  by 60-680 ppm within a set. It is measured per set by cross-correlating the bus yaw
  rate with the lidar IMU's gyro (sensor clock) in 60 s windows and fitted as offset +
  drift (:func:`estimate_bus_clock`); every bus timestamp is mapped onto the sensor
  clock first;
* the GPS **fixes at 1 Hz** -- position, course and speed change once a second; the
  10 Hz CAN messages repeat the last fix. A fix's **position** reaches the bus logger
  :data:`GPS_POSITION_LAG_S` after the moment it describes (0.20-0.26 s on the five
  sets whose speed varies enough to measure it), while its **speed and course** lag by
  0.6-1.5 s, differently per set, so they are used only where that does not matter;
* the CAN bus carries the wheel speed (``SPEED``, ~38 Hz, km/h, within a few percent of
  the GPS distance, per set) and the yaw rate (``KINEMATICS.YAW_RATE``, 10 Hz, deg/s,
  positive counter-clockwise, quantised to 0.244 deg/s, its scale within 1 % of the GPS
  course change across turns);
* one set (``alakylantie_80kmh``) ships an empty CAN log. There the speed comes from
  the GPS positions (fix-to-fix distance) and the yaw rate from the PCAN-GPS gyro,
  which reads :data:`PCAN_GYRO_SCALE` low against the GPS course and reaches the log
  :data:`PCAN_GYRO_DELAY_S` after the CAN yaw rate.

So the pose is integrated from the speed and yaw rate (smooth and on time) and then
pulled towards the GPS by corrections that change slowly (Gaussian-smoothed over
``CORRECTION_SIGMA_S``): the position towards the lag-corrected fixes, the heading
towards the GPS course on straight stretches, and the wheel speed is scaled to the GPS
distance. Seconds-scale motion -- past/future trajectories -- comes from the vehicle
bus, drift and absolute position from the GPS. Positions are metres east/north in a
local azimuthal-equidistant frame anchored at the set's first fix (true north, so the
GPS course needs no grid convergence; small numbers survive the float32
``Trajectory``); the heading is counter-clockwise from east. The GPS altitude is too
noisy to use: z = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from pyproj import Transformer

from standard_e2e.caching.src_datasets.muhos._muhos_io import read_signal

GPS_POSITION_LAG_S = 0.2
PCAN_GYRO_SCALE = 1.087
CORRECTION_SIGMA_S = 15.0
MIN_SPEED_FOR_COURSE_MPS = 5.0
# A fix's course is used when the yaw rate stayed below this over the preceding
# ``_COURSE_WINDOW_S`` (the course lags by up to 1.5 s).
STRAIGHT_YAW_RATE_DEG_S = 1.0
_COURSE_WINDOW_S = (-2.5, 0.5)
INTEGRATION_STEP_S = 0.01
_KMH_TO_MPS = 1.0 / 3.6
# Bus-clock alignment: windows of the yaw-rate cross-correlation, their acceptance.
CLOCK_WINDOW_S = 60.0
CLOCK_STEP_S = 30.0
CLOCK_MAX_LAG_S = 3.0
CLOCK_MIN_CORR = 0.8
CLOCK_MIN_YAW_STD = np.radians(0.3)
# The PCAN-GPS gyro reaches the bus log this much later than the CAN yaw rate (40-59 ms
# on the six sets with both); a clock fitted on it is shifted by this to match CAN's.
PCAN_GYRO_DELAY_S = 0.047


@dataclass(frozen=True)
class BusClock:
    """Bus-log time = sensor time + lag, with lag = ``lag0 + drift * (t_bus - t0)``."""

    t0: float
    lag0: float
    drift: float
    # Windows the fit used (0: the bus log is taken as on the sensor clock).
    n_windows: int

    def to_sensor(self, t_bus: np.ndarray) -> np.ndarray:
        return t_bus - (self.lag0 + self.drift * (t_bus - self.t0))


def _window_lag(
    t_bus: np.ndarray,
    w_bus: np.ndarray,
    t_imu: np.ndarray,
    w_imu: np.ndarray,
    lo: float,
    hi: float,
) -> tuple[float, float] | None:
    """``(lag, correlation)`` aligning the bus yaw rate on [lo, hi) with the IMU's, or
    None."""
    grid = np.arange(lo, hi, 0.02)
    ref = np.interp(grid, t_bus, w_bus)
    if ref.std() < CLOCK_MIN_YAW_STD:
        return None
    ref = (ref - ref.mean()) / ref.std()

    def corr(lag: float) -> float:
        x = np.interp(grid - lag, t_imu, w_imu)
        sd = x.std()
        return float(np.mean(ref * (x - x.mean())) / sd) if sd > 0 else -1.0

    coarse = np.arange(-CLOCK_MAX_LAG_S, CLOCK_MAX_LAG_S + 1e-9, 0.02)
    k = int(np.argmax([corr(lag) for lag in coarse]))
    fine = np.arange(coarse[k] - 0.03, coarse[k] + 0.03 + 1e-9, 0.002)
    c = np.array([corr(lag) for lag in fine])
    j = int(np.argmax(c))
    return (float(fine[j]), float(c[j])) if c[j] >= CLOCK_MIN_CORR else None


def estimate_bus_clock(
    t_bus: np.ndarray, w_bus: np.ndarray, t_imu: np.ndarray, w_imu: np.ndarray
) -> BusClock:
    """Fit the bus-log clock to the sensor clock: per window, the lag that best aligns
    the bus yaw rate with the IMU gyro's (both counter-clockwise; scale does not
    matter); a robust line through the window lags gives offset and drift."""
    lo = max(t_bus[0], t_imu[0] + CLOCK_MAX_LAG_S)
    hi = min(t_bus[-1], t_imu[-1] - CLOCK_MAX_LAG_S)
    centres, lags = [], []
    for start in np.arange(lo, hi - CLOCK_WINDOW_S + 1e-9, CLOCK_STEP_S):
        found = _window_lag(t_bus, w_bus, t_imu, w_imu, start, start + CLOCK_WINDOW_S)
        if found is not None:
            centres.append(start + CLOCK_WINDOW_S / 2)
            lags.append(found[0])
    if not lags:
        raise ValueError(
            "MUHOS: could not align the bus log to the sensor clock (no window where "
            "the yaw rates correlate)."
        )
    t0 = 0.5 * (lo + hi)
    c, lag = np.array(centres) - t0, np.array(lags)
    if len(lag) == 1:
        return BusClock(t0=t0, lag0=float(lag[0]), drift=0.0, n_windows=1)
    keep = np.ones(len(lag), bool)
    for _ in range(3):  # drop windows more than 3 MAD (at least 20 ms) off the line
        drift, lag0 = np.polyfit(c[keep], lag[keep], 1)
        res = np.abs(lag - (lag0 + drift * c))
        keep = res <= max(0.02, 3.0 * np.median(res[keep]))
        if keep.sum() < 2:
            keep[np.argsort(res)[:2]] = True
    drift, lag0 = np.polyfit(c[keep], lag[keep], 1)
    return BusClock(
        t0=t0, lag0=float(lag0), drift=float(drift), n_windows=int(keep.sum())
    )


@dataclass(frozen=True)
class GpsFixes:
    """GPS fixes at their (lag-uncorrected) reception times; one row per fix."""

    t: np.ndarray  # reception time, s
    lat: np.ndarray
    lon: np.ndarray
    speed: np.ndarray  # m/s
    course_deg: np.ndarray  # clockwise from true north


def read_gps_fixes(gps_rows: list[dict]) -> GpsFixes:
    """One row per fix. The PCAN-GPS sends course/speed, longitude, latitude and then
    the fix's UTC time every 100 ms, repeating the last fix; a new fix changes all of
    them in the same cycle, so a change of the UTC second marks a fix and the other
    values are the latest messages up to it."""
    second = read_signal(gps_rows, "GPS_DateTime", "UTC_Second")
    t = second[1:, 0][np.diff(second[:, 1]) != 0]

    def at_fix(message: str, signal: str) -> np.ndarray:
        series = read_signal(gps_rows, message, signal)
        idx = np.searchsorted(series[:, 0], t, side="right") - 1
        values: np.ndarray = series[np.clip(idx, 0, len(series) - 1), 1]
        return values

    lat = (
        at_fix("GPS_PositionLatitude", "GPS_Latitude_Degree")
        + at_fix("GPS_PositionLatitude", "GPS_Latitude_Minutes") / 60.0
    )
    lon = (
        at_fix("GPS_PositionLongitude", "GPS_Longitude_Degree")
        + at_fix("GPS_PositionLongitude", "GPS_Longitude_Minutes") / 60.0
    )
    lat *= np.where(
        at_fix("GPS_PositionLatitude", "GPS_IndicatorNS") == ord("S"), -1.0, 1.0
    )
    lon *= np.where(
        at_fix("GPS_PositionLongitude", "GPS_IndicatorEW") == ord("W"), -1.0, 1.0
    )
    return GpsFixes(
        t=t,
        lat=lat,
        lon=lon,
        speed=at_fix("GPS_CourseSpeed", "GPS_Speed") * _KMH_TO_MPS,
        course_deg=at_fix("GPS_CourseSpeed", "GPS_Course"),
    )


def _gaussian_smooth(
    t_samples: np.ndarray, values: np.ndarray, t_query: np.ndarray, sigma: float
) -> np.ndarray:
    """Kernel-weighted average of irregular samples, evaluated at ``t_query``."""
    out = np.empty((len(t_query),) + values.shape[1:])
    for start in range(0, len(t_query), 2048):
        tq = t_query[start : start + 2048]
        w = np.exp(-0.5 * ((tq[:, None] - t_samples[None, :]) / sigma) ** 2)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1.0
        out[start : start + 2048] = (
            w @ values.reshape(len(t_samples), -1) / w_sum
        ).reshape((len(tq),) + values.shape[1:])
    return out


@dataclass(frozen=True)
class EgoTrajectory:
    """Fused ego trajectory of one set, sampled every :data:`INTEGRATION_STEP_S`."""

    t: np.ndarray
    x: np.ndarray  # metres east of the anchor
    y: np.ndarray  # metres north of the anchor
    heading: np.ndarray  # rad, counter-clockwise from east
    speed: np.ndarray  # m/s
    anchor_lat: float
    anchor_lon: float
    source: str  # "can" or "gps+pcan_gyro" (empty CAN log)
    clock: BusClock  # how the bus log was mapped onto the sensor clock
    speed_scale: float
    yaw_rate_bias: float  # rad/s, removed
    gps_residual_m: np.ndarray  # |fused - GPS| at each lag-corrected fix

    def at(
        self, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(x, y, heading, speed)`` at ``times`` (linear; heading unwrapped)."""
        return (
            np.interp(times, self.t, self.x),
            np.interp(times, self.t, self.y),
            np.interp(times, self.t, self.heading),
            np.interp(times, self.t, self.speed),
        )


def _bus_yaw_rate(can_rows: list[dict], gps_rows: list[dict]) -> tuple[np.ndarray, str]:
    """``(yaw_rate (M, 2) [bus time, rad/s], source)``: the CAN yaw rate, or the
    PCAN-GPS gyro when the CAN log is empty."""
    yaw_rate = read_signal(can_rows, "KINEMATICS", "YAW_RATE")
    if len(yaw_rate) > 1 and len(read_signal(can_rows, "SPEED", "SPEED")) > 1:
        yaw_rate[:, 1] = np.radians(yaw_rate[:, 1])
        return yaw_rate, "can"
    yaw_rate = read_signal(gps_rows, "L3GD20_Rotation_B", "Rotation_Z")
    yaw_rate[:, 1] = np.radians(yaw_rate[:, 1] * PCAN_GYRO_SCALE)
    return yaw_rate, "gps+pcan_gyro"


def _motion_inputs(
    can_rows: list[dict], source: str, fix_t: np.ndarray, gx: np.ndarray, gy: np.ndarray
) -> np.ndarray:
    """Speed ``(N, 2) [time, m/s]`` on the same clock as ``fix_t``."""
    if source == "can":
        speed = read_signal(can_rows, "SPEED", "SPEED")
        speed[:, 1] *= _KMH_TO_MPS
        return speed
    # No vehicle bus: fix-to-fix GPS speed, at the middle of each interval.
    step = np.hypot(np.diff(gx), np.diff(gy)) / np.diff(fix_t)
    return np.stack([0.5 * (fix_t[1:] + fix_t[:-1]), step], axis=1)


def build_ego_trajectory(
    can_rows: list[dict],
    gps_rows: list[dict],
    imu: tuple[np.ndarray, np.ndarray] | None = None,
) -> EgoTrajectory:
    """The fused trajectory on the sensor clock. ``imu`` is the lidar IMU ``(t,
    angular velocity)`` (:func:`._muhos_io.read_imu`) the bus log is aligned to;
    without it the bus log is taken as already on the sensor clock."""
    gps = read_gps_fixes(gps_rows)
    bus_yaw, source = _bus_yaw_rate(can_rows, gps_rows)
    clock = (
        estimate_bus_clock(bus_yaw[:, 0], bus_yaw[:, 1], imu[0], imu[1][:, 2])
        if imu is not None
        else BusClock(t0=0.0, lag0=0.0, drift=0.0, n_windows=0)
    )
    if source != "can" and clock.n_windows:
        clock = replace(clock, lag0=clock.lag0 - PCAN_GYRO_DELAY_S)
    gps = replace(gps, t=clock.to_sensor(gps.t))
    bus_yaw[:, 0] = clock.to_sensor(bus_yaw[:, 0])
    anchor_lat, anchor_lon = float(gps.lat[0]), float(gps.lon[0])
    to_local = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={anchor_lat} +lon_0={anchor_lon} +datum=WGS84 +units=m",
        always_xy=True,
    )
    gx, gy = (np.asarray(c) for c in to_local.transform(gps.lon, gps.lat))
    fix_t = gps.t - GPS_POSITION_LAG_S
    speed_in = _motion_inputs(can_rows, source, fix_t, gx, gy)
    if source == "can":
        speed_in[:, 0] = clock.to_sensor(speed_in[:, 0])
    yaw_in = bus_yaw

    t = np.arange(
        max(speed_in[0, 0], yaw_in[0, 0]),
        min(speed_in[-1, 0], yaw_in[-1, 0]),
        INTEGRATION_STEP_S,
    )
    v = np.interp(t, speed_in[:, 0], speed_in[:, 1])
    omega = np.interp(t, yaw_in[:, 0], yaw_in[:, 1])
    inside = (fix_t >= t[0]) & (fix_t <= t[-1])
    fix_t, gx, gy = fix_t[inside], gx[inside], gy[inside]
    course_deg = gps.course_deg[inside]

    # Yaw-rate bias from standstill (a parked car does not turn).
    still = v < 0.05
    bias = (
        float(np.median(omega[still]))
        if still.sum() * INTEGRATION_STEP_S > 5.0
        else 0.0
    )
    omega = omega - bias
    # Straight driving before each fix: its course is valid despite the course lag.
    abs_omega_integral = np.concatenate(
        [[0.0], np.cumsum(np.abs(omega)) * INTEGRATION_STEP_S]
    )
    lo, hi = (np.searchsorted(t, fix_t + w) for w in _COURSE_WINDOW_S)
    mean_abs_omega = (
        (abs_omega_integral[hi] - abs_omega_integral[lo])
        / np.maximum(hi - lo, 1)
        / INTEGRATION_STEP_S
    )
    v_at_fix = np.interp(fix_t, t, v)
    straight = (mean_abs_omega < np.radians(STRAIGHT_YAW_RATE_DEG_S)) & (
        v_at_fix > MIN_SPEED_FOR_COURSE_MPS
    )

    # Speed scale: GPS distance over dead-reckoned distance between fixes, on straight
    # stretches while moving.
    travelled = np.concatenate([[0.0], np.cumsum(v[1:] * np.diff(t))])
    d_dr = np.diff(np.interp(fix_t, t, travelled))
    d_gps = np.hypot(np.diff(gx), np.diff(gy))
    use = straight[1:] & straight[:-1]
    scale = float(d_gps[use].sum() / d_dr[use].sum()) if use.sum() > 20 else 1.0
    speed = v * scale

    # Heading: integrated yaw rate, offset to the GPS course on straight stretches.
    heading = np.cumsum(omega) * INTEGRATION_STEP_S
    if straight.any():
        course_heading = np.radians(90.0 - course_deg[straight])
        offset = np.unwrap(
            np.angle(
                np.exp(1j * (course_heading - np.interp(fix_t[straight], t, heading)))
            )
        )
        heading = heading + _gaussian_smooth(
            fix_t[straight], offset, t, CORRECTION_SIGMA_S
        )
    x_dr = np.cumsum(speed * np.cos(heading)) * INTEGRATION_STEP_S
    y_dr = np.cumsum(speed * np.sin(heading)) * INTEGRATION_STEP_S

    # Position: dead reckoning plus the smoothed offset to the lag-corrected fixes.
    err = np.stack(
        [gx - np.interp(fix_t, t, x_dr), gy - np.interp(fix_t, t, y_dr)], axis=1
    )
    corr_xy = _gaussian_smooth(fix_t, err, t, CORRECTION_SIGMA_S)
    x, y = x_dr + corr_xy[:, 0], y_dr + corr_xy[:, 1]
    residual = np.hypot(gx - np.interp(fix_t, t, x), gy - np.interp(fix_t, t, y))
    return EgoTrajectory(
        t=t,
        x=x,
        y=y,
        heading=heading,
        speed=speed,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        source=source,
        clock=clock,
        speed_scale=scale,
        yaw_rate_bias=bias,
        gps_residual_m=residual,
    )
