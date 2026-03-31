use bytemuck::cast_slice;
use rayon::prelude::*;
use serde::Deserialize;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const R_D: f64 = 287.04;
const CP_D: f64 = 1004.5;
const KAPPA: f64 = R_D / CP_D;
const P0: f64 = 100000.0;

const PROJ_MAGIC: &[u8; 8] = b"GWMPRJ1\0";
const PRES_MAGIC: &[u8; 8] = b"GWMPRES1";
const TERRAIN_MAGIC: &[u8; 8] = b"GWMTERR1";
const INIT_MODE_MAGIC: &[u8; 8] = b"GWMINIT1";
const TIME_MAGIC: &[u8; 8] = b"GWMTIME1";

#[derive(Debug, Deserialize)]
struct Manifest {
    nx: usize,
    ny: usize,
    nz: usize,
    n_plev: usize,
    dx: f64,
    dy: f64,
    ztop: f64,
    truelat1: f64,
    truelat2: f64,
    stand_lon: f64,
    ref_lat: f64,
    ref_lon: f64,
    terrain_following_init: bool,
    validity_unix: i64,
    reference_unix: i64,
    forecast_hour: i32,
    output: PathBuf,
    scratch_dir: PathBuf,
    z_levels: PathBuf,
    p_levels_pa: PathBuf,
    u_plev: PathBuf,
    v_plev: PathBuf,
    t_plev: PathBuf,
    q_plev: PathBuf,
    gh_plev: PathBuf,
    orog_model: PathBuf,
}

#[derive(Default, Clone, Copy)]
struct Stats {
    min: f64,
    max: f64,
    sum: f64,
    count: usize,
}

impl Stats {
    fn update_from_slice(&mut self, values: &[f64]) {
        if values.is_empty() {
            return;
        }
        let local = values.iter().fold(
            Stats {
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
                sum: 0.0,
                count: 0,
            },
            |mut acc, &v| {
                acc.min = acc.min.min(v);
                acc.max = acc.max.max(v);
                acc.sum += v;
                acc.count += 1;
                acc
            },
        );
        if self.count == 0 {
            *self = local;
        } else {
            self.min = self.min.min(local.min);
            self.max = self.max.max(local.max);
            self.sum += local.sum;
            self.count += local.count;
        }
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

struct Inputs {
    z_levels: Vec<f64>,
    p_levels_pa: Vec<f64>,
    log_p_levels: Vec<f64>,
    theta_factor: Vec<f64>,
    u_plev: Vec<f64>,
    v_plev: Vec<f64>,
    t_plev: Vec<f64>,
    q_plev: Vec<f64>,
    gh_plev: Vec<f64>,
    orog_model: Vec<f64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_path = parse_manifest_arg()?;
    let started = Instant::now();
    let manifest: Manifest = serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;

    let plane_len = manifest.nx * manifest.ny;
    let n3d_plev = manifest.n_plev * plane_len;

    println!("GPU-WM Rust init writer");
    println!(
        "  grid: {} x {} x {} ({} pressure levels)",
        manifest.nx, manifest.ny, manifest.nz, manifest.n_plev
    );
    println!(
        "  interpolation: {}",
        if manifest.terrain_following_init {
            "terrain-following"
        } else {
            "flat-height"
        }
    );

    let inputs = Inputs {
        z_levels: read_f64_vec(&manifest.z_levels, manifest.nz)?,
        p_levels_pa: read_f64_vec(&manifest.p_levels_pa, manifest.n_plev)?,
        log_p_levels: Vec::new(),
        theta_factor: Vec::new(),
        u_plev: read_f64_vec(&manifest.u_plev, n3d_plev)?,
        v_plev: read_f64_vec(&manifest.v_plev, n3d_plev)?,
        t_plev: read_f64_vec(&manifest.t_plev, n3d_plev)?,
        q_plev: read_f64_vec(&manifest.q_plev, n3d_plev)?,
        gh_plev: read_f64_vec(&manifest.gh_plev, n3d_plev)?,
        orog_model: read_f64_vec(&manifest.orog_model, plane_len)?,
    };
    let mut inputs = inputs;
    inputs.log_p_levels = inputs.p_levels_pa.iter().map(|v| v.ln()).collect();
    inputs.theta_factor = inputs
        .p_levels_pa
        .iter()
        .map(|p| (P0 / *p).powf(KAPPA))
        .collect();

    let scratch = &manifest.scratch_dir;
    fs::create_dir_all(scratch)?;
    let u_tmp = scratch.join("u_model.tmp");
    let v_tmp = scratch.join("v_model.tmp");
    let theta_tmp = scratch.join("theta_model.tmp");
    let qv_tmp = scratch.join("qv_model.tmp");
    let p_tmp = scratch.join("p_model.tmp");

    let mut u_writer = BufWriter::new(File::create(&u_tmp)?);
    let mut v_writer = BufWriter::new(File::create(&v_tmp)?);
    let mut theta_writer = BufWriter::new(File::create(&theta_tmp)?);
    let mut qv_writer = BufWriter::new(File::create(&qv_tmp)?);
    let mut p_writer = BufWriter::new(File::create(&p_tmp)?);

    let mut u_stats = Stats::default();
    let mut v_stats = Stats::default();
    let mut theta_stats = Stats::default();
    let mut qv_stats = Stats::default();
    let mut p_stats = Stats::default();

    for k in 0..manifest.nz {
        if k == 0 || (k + 1) % 10 == 0 || k + 1 == manifest.nz {
            println!("  level {}/{}", k + 1, manifest.nz);
        }

        let mut u_plane = vec![0.0_f64; plane_len];
        let mut v_plane = vec![0.0_f64; plane_len];
        let mut theta_plane = vec![0.0_f64; plane_len];
        let mut qv_plane = vec![0.0_f64; plane_len];
        let mut p_plane = vec![0.0_f64; plane_len];

        u_plane
            .par_iter_mut()
            .zip(v_plane.par_iter_mut())
            .zip(theta_plane.par_iter_mut())
            .zip(qv_plane.par_iter_mut())
            .zip(p_plane.par_iter_mut())
            .enumerate()
            .for_each(|(idx, ((((u_out, v_out), theta_out), qv_out), p_out))| {
                let (_, value) = compute_column_values(&manifest, &inputs, idx, k);
                *u_out = value.u;
                *v_out = value.v;
                *theta_out = value.theta;
                *qv_out = value.qv;
                *p_out = value.pressure;
            });

        write_f64_slice(&mut u_writer, &u_plane)?;
        write_f64_slice(&mut v_writer, &v_plane)?;
        write_f64_slice(&mut theta_writer, &theta_plane)?;
        write_f64_slice(&mut qv_writer, &qv_plane)?;
        write_f64_slice(&mut p_writer, &p_plane)?;

        u_stats.update_from_slice(&u_plane);
        v_stats.update_from_slice(&v_plane);
        theta_stats.update_from_slice(&theta_plane);
        qv_stats.update_from_slice(&qv_plane);
        p_stats.update_from_slice(&p_plane);
    }

    u_writer.flush()?;
    v_writer.flush()?;
    theta_writer.flush()?;
    qv_writer.flush()?;
    p_writer.flush()?;

    let output_parent = manifest
        .output
        .parent()
        .ok_or_else(|| "output path has no parent directory".to_string())?;
    fs::create_dir_all(output_parent)?;
    let mut out = BufWriter::new(File::create(&manifest.output)?);

    write_i32(&mut out, manifest.nx as i32)?;
    write_i32(&mut out, manifest.ny as i32)?;
    write_i32(&mut out, manifest.nz as i32)?;
    write_f64(&mut out, manifest.dx)?;
    write_f64(&mut out, manifest.dy)?;
    write_f64(&mut out, manifest.ztop)?;
    write_f64_slice(&mut out, &inputs.z_levels)?;

    copy_file_contents(&u_tmp, &mut out)?;
    copy_file_contents(&v_tmp, &mut out)?;
    write_zero_field(&mut out, plane_len, manifest.nz)?;
    copy_file_contents(&theta_tmp, &mut out)?;
    copy_file_contents(&qv_tmp, &mut out)?;
    write_zero_field(&mut out, plane_len, manifest.nz)?;
    write_zero_field(&mut out, plane_len, manifest.nz)?;

    out.write_all(PROJ_MAGIC)?;
    write_f64(&mut out, manifest.truelat1)?;
    write_f64(&mut out, manifest.truelat2)?;
    write_f64(&mut out, manifest.stand_lon)?;
    write_f64(&mut out, manifest.ref_lat)?;
    write_f64(&mut out, manifest.ref_lon)?;

    out.write_all(PRES_MAGIC)?;
    copy_file_contents(&p_tmp, &mut out)?;

    out.write_all(TERRAIN_MAGIC)?;
    write_f64_slice(&mut out, &inputs.orog_model)?;

    out.write_all(INIT_MODE_MAGIC)?;
    write_i32(&mut out, if manifest.terrain_following_init { 1 } else { 0 })?;
    write_i32(&mut out, 0)?;

    out.write_all(TIME_MAGIC)?;
    write_i64(&mut out, manifest.validity_unix)?;
    write_i64(&mut out, manifest.reference_unix)?;
    write_i32(&mut out, manifest.forecast_hour)?;
    write_i32(&mut out, 0)?;
    out.flush()?;

    println!("Field statistics after interpolation:");
    println!(
        "  u     : min={:.2}  max={:.2}  mean={:.2} m/s",
        u_stats.min,
        u_stats.max,
        u_stats.mean()
    );
    println!(
        "  v     : min={:.2}  max={:.2}  mean={:.2} m/s",
        v_stats.min,
        v_stats.max,
        v_stats.mean()
    );
    println!(
        "  theta : min={:.2}  max={:.2}  mean={:.2} K",
        theta_stats.min,
        theta_stats.max,
        theta_stats.mean()
    );
    println!(
        "  qv    : min={:.6}  max={:.6}  mean={:.6} kg/kg",
        qv_stats.min,
        qv_stats.max,
        qv_stats.mean()
    );
    println!(
        "  p     : min={:.0}  max={:.0}  mean={:.0} Pa",
        p_stats.min,
        p_stats.max,
        p_stats.mean()
    );
    let terrain_stats = slice_stats(&inputs.orog_model);
    println!(
        "  orog  : min={:.0}  max={:.0} m",
        terrain_stats.min, terrain_stats.max
    );
    println!("  Wrote: {}", manifest.output.display());
    println!("  Elapsed: {:.2} s", started.elapsed().as_secs_f64());

    let _ = fs::remove_file(u_tmp);
    let _ = fs::remove_file(v_tmp);
    let _ = fs::remove_file(theta_tmp);
    let _ = fs::remove_file(qv_tmp);
    let _ = fs::remove_file(p_tmp);

    Ok(())
}

struct ColumnValues {
    u: f64,
    v: f64,
    theta: f64,
    qv: f64,
    pressure: f64,
}

fn compute_column_values(
    manifest: &Manifest,
    inputs: &Inputs,
    cell_idx: usize,
    k: usize,
) -> (usize, ColumnValues) {
    let terrain = if manifest.terrain_following_init {
        inputs.orog_model[cell_idx].min(manifest.ztop - 1.0)
    } else {
        0.0
    };
    let z_target = if manifest.terrain_following_init {
        let column_depth = (manifest.ztop - terrain).max(1.0);
        terrain + (inputs.z_levels[k] / manifest.ztop) * column_depth
    } else {
        inputs.z_levels[k]
    };

    let (idx_lo, idx_hi) = find_bracketing_levels(inputs, manifest, cell_idx, z_target);
    let h_lo = at3d(&inputs.gh_plev, manifest, idx_lo, cell_idx);
    let h_hi = at3d(&inputs.gh_plev, manifest, idx_hi, cell_idx);
    let w_hi = if (h_hi - h_lo).abs() < 1.0e-3 {
        0.5
    } else {
        ((z_target - h_lo) / (h_hi - h_lo)).clamp(0.0, 1.0)
    };
    let w_lo = 1.0 - w_hi;

    let pressure = (w_lo * inputs.log_p_levels[idx_lo] + w_hi * inputs.log_p_levels[idx_hi]).exp();
    let u = interp3d(&inputs.u_plev, manifest, cell_idx, idx_lo, idx_hi, w_lo, w_hi);
    let v = interp3d(&inputs.v_plev, manifest, cell_idx, idx_lo, idx_hi, w_lo, w_hi);
    let t_lo = at3d(&inputs.t_plev, manifest, idx_lo, cell_idx);
    let t_hi = at3d(&inputs.t_plev, manifest, idx_hi, cell_idx);
    let theta = w_lo * t_lo * inputs.theta_factor[idx_lo] + w_hi * t_hi * inputs.theta_factor[idx_hi];

    let q_lo = at3d(&inputs.q_plev, manifest, idx_lo, cell_idx);
    let q_hi = at3d(&inputs.q_plev, manifest, idx_hi, cell_idx);
    let qv_lo = (q_lo / (1.0 - q_lo).max(1.0e-12)).max(0.0);
    let qv_hi = (q_hi / (1.0 - q_hi).max(1.0e-12)).max(0.0);
    let qv = (w_lo * qv_lo + w_hi * qv_hi).max(0.0);

    (
        cell_idx,
        ColumnValues {
            u,
            v,
            theta,
            qv,
            pressure,
        },
    )
}

fn find_bracketing_levels(
    inputs: &Inputs,
    manifest: &Manifest,
    cell_idx: usize,
    z_target: f64,
) -> (usize, usize) {
    let n_plev = manifest.n_plev;
    let lowest = at3d(&inputs.gh_plev, manifest, 0, cell_idx);
    if z_target <= lowest {
        return (0, 1);
    }

    let highest = at3d(&inputs.gh_plev, manifest, n_plev - 1, cell_idx);
    if z_target >= highest {
        return (n_plev - 2, n_plev - 1);
    }

    for ip in 0..(n_plev - 1) {
        let h_lo = at3d(&inputs.gh_plev, manifest, ip, cell_idx);
        let h_hi = at3d(&inputs.gh_plev, manifest, ip + 1, cell_idx);
        if h_lo <= z_target && z_target <= h_hi {
            return (ip, ip + 1);
        }
    }

    (n_plev - 2, n_plev - 1)
}

fn at3d(values: &[f64], manifest: &Manifest, level: usize, cell_idx: usize) -> f64 {
    values[level * manifest.nx * manifest.ny + cell_idx]
}

fn interp3d(
    values: &[f64],
    manifest: &Manifest,
    cell_idx: usize,
    idx_lo: usize,
    idx_hi: usize,
    w_lo: f64,
    w_hi: f64,
) -> f64 {
    w_lo * at3d(values, manifest, idx_lo, cell_idx) + w_hi * at3d(values, manifest, idx_hi, cell_idx)
}

fn slice_stats(values: &[f64]) -> Stats {
    let mut stats = Stats::default();
    stats.update_from_slice(values);
    stats
}

fn parse_manifest_arg() -> Result<PathBuf, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--manifest" {
            let path = args
                .next()
                .ok_or_else(|| "--manifest requires a path".to_string())?;
            return Ok(PathBuf::from(path));
        }
    }
    Err("usage: gpuwm-init-writer --manifest <path>".into())
}

fn read_f64_vec(path: &Path, expected_len: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let expected_bytes = expected_len * std::mem::size_of::<f64>();
    if bytes.len() != expected_bytes {
        return Err(format!(
            "unexpected byte length for {}: got {}, expected {}",
            path.display(),
            bytes.len(),
            expected_bytes
        )
        .into());
    }

    let mut values = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(8) {
        values.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(values)
}

fn write_f64_slice(writer: &mut dyn Write, values: &[f64]) -> Result<(), Box<dyn Error>> {
    writer.write_all(cast_slice(values))?;
    Ok(())
}

fn write_f64(writer: &mut dyn Write, value: f64) -> Result<(), Box<dyn Error>> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_i32(writer: &mut dyn Write, value: i32) -> Result<(), Box<dyn Error>> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_i64(writer: &mut dyn Write, value: i64) -> Result<(), Box<dyn Error>> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_zero_field(
    writer: &mut dyn Write,
    plane_len: usize,
    nz: usize,
) -> Result<(), Box<dyn Error>> {
    let zero_plane = vec![0.0_f64; plane_len];
    for _ in 0..nz {
        write_f64_slice(writer, &zero_plane)?;
    }
    Ok(())
}

fn copy_file_contents(path: &Path, writer: &mut dyn Write) -> Result<(), Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        writer.write_all(&buffer[..read])?;
    }
    Ok(())
}
