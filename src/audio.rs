use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, Sample, SizedSample, StreamConfig};
use rustfft::num_traits::ToPrimitive;

use crate::buffer::SharedBuf;
use std::sync::{Arc, Mutex};

pub fn pick_input_device() -> Result<Device> {
    let host = cpal::default_host();

    if let Ok(devices) = host.input_devices() {
        for dev in devices {
            if let Ok(name) = dev.name() {
                if name.to_lowercase().contains("monitor") {
                    return Ok(dev);
                }
            }
        }
    }

    host.default_input_device()
        .context("No default input device")
}

pub fn best_config_for(device: &Device) -> Result<StreamConfig> {
    let mut cfg = device.default_input_config()?.config();
    cfg.sample_rate.0 = cfg.sample_rate.0.clamp(44_100, 48_000);
    Ok(cfg)
}

pub fn build_stream<T>(
    device: Device,
    cfg: StreamConfig,
    shared: Arc<Mutex<SharedBuf>>,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample + ToPrimitive,
{
    let ch = cfg.channels as usize;
    let err_fn = |e| eprintln!("Stream error: {}", e);

    let stream = device.build_input_stream(
        &cfg,
        move |data: &[T], _| {
            // Try to acquire lock, skip if busy
            if let Ok(mut buf) = shared.try_lock() {
                let frames = data.chunks_exact(ch);
                for frame in frames {
                    let mut acc = 0.0f32;
                    for &s in frame {
                        acc += s.to_f32().unwrap_or(0.0);
                    }
                    buf.push(acc / ch as f32);
                }
            }
        },
        err_fn,
        None,
    )?;

    Ok(stream)
}
