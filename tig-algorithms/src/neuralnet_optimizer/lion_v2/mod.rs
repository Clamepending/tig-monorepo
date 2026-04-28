// Lion v2 for c006 — depth-aware LR scheduling.
//
// v1 used a single LR (1e-4) and worked on n_hidden=4 (quality 703k > threshold
// 400k) but failed on deeper tracks: n_hidden=7 → 368k (sub-threshold),
// n_hidden=10 → -284k (diverged). Gradient magnitudes grow with depth so
// Lion's sign(momentum) update needs smaller LR for deeper nets.
//
// v2 picks LR by depth empirically:
//   n_hidden=4   → lr 1e-4   (matches v1 result)
//   n_hidden=7   → lr 5e-5
//   n_hidden=10  → lr 3e-5
//   n_hidden=14  → lr 2e-5
//   n_hidden=18  → lr 1e-5
// And longer warmup for deeper nets to let early-batch grad noise settle.

use anyhow::Result;
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tig_challenges::neuralnet_optimizer::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn help() {
    println!("Lion v2 — depth-aware LR (1e-4 for n_hidden=4 → 1e-5 for n_hidden=18)");
}

const THREADS_PER_BLOCK: u32 = 256;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.99;
const WEIGHT_DECAY: f32 = 0.01;

// Stash num_hidden across the init/step boundary because training_loop only
// passes (seed, param_sizes) to optimizer_init. We read challenge.num_hidden_layers
// in solve_challenge BEFORE training_loop, then optimizer_init reads it back.
static NUM_HIDDEN: AtomicUsize = AtomicUsize::new(0);

fn pick_lr(num_hidden: usize) -> (f32, usize) {
    // (lr, warmup_steps)
    match num_hidden {
        0..=4 => (1.0e-4, 32),
        5..=7 => (5.0e-5, 48),
        8..=10 => (3.0e-5, 64),
        11..=14 => (2.0e-5, 96),
        _ => (1.0e-5, 128),
    }
}

#[derive(Clone)]
struct LionState {
    m: Vec<CudaSlice<f32>>,
    cfgs: Vec<LaunchConfig>,
    lr: f32,
    warmup_steps: usize,
    step_count: usize,
}

impl OptimizerStateTrait for LionState {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    NUM_HIDDEN.store(challenge.num_hidden_layers as usize, Ordering::SeqCst);
    training_loop(
        challenge,
        save_solution,
        module,
        stream,
        prop,
        optimizer_init,
        optimizer_query_at_params,
        optimizer_step,
    )?;
    Ok(())
}

fn optimizer_init(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let nh = NUM_HIDDEN.load(Ordering::SeqCst);
    let (lr, warmup_steps) = pick_lr(nh);
    let mut m = Vec::with_capacity(param_sizes.len());
    let mut cfgs = Vec::with_capacity(param_sizes.len());
    for &sz in param_sizes {
        m.push(stream.alloc_zeros::<f32>(sz)?);
        let n_blocks = ((sz as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cfgs.push(LaunchConfig {
            grid_dim: (n_blocks.max(1), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        });
    }
    Ok(Box::new(LionState {
        m,
        cfgs,
        lr,
        warmup_steps,
        step_count: 0,
    }))
}

fn optimizer_query_at_params(
    _state: &dyn OptimizerStateTrait,
    _params: &[CudaSlice<f32>],
    _epoch: usize,
    _train_loss: Option<f32>,
    _val_loss: Option<f32>,
    _stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>> {
    Ok(None)
}

fn optimizer_step(
    state: &mut dyn OptimizerStateTrait,
    model_params: &[CudaSlice<f32>],
    gradients: &[CudaSlice<f32>],
    _epoch: usize,
    _train_loss: Option<f32>,
    _val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let st = state
        .as_any_mut()
        .downcast_mut::<LionState>()
        .ok_or_else(|| anyhow::anyhow!("downcast failed"))?;
    st.step_count += 1;

    let lr_now = if st.step_count < st.warmup_steps {
        st.lr * (st.step_count as f32 / st.warmup_steps as f32)
    } else {
        st.lr
    };

    let kernel = module.load_function("lion_step")?;
    let mut updates = Vec::with_capacity(model_params.len());

    for i in 0..model_params.len() {
        let n = model_params[i].len() as i32;
        let mut update = stream.alloc_zeros::<f32>(n as usize)?;
        let cfg = st.cfgs[i];
        let lr = lr_now;
        let beta1 = BETA1;
        let beta2 = BETA2;
        let wd = WEIGHT_DECAY;
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&gradients[i])
                .arg(&model_params[i])
                .arg(&mut st.m[i])
                .arg(&mut update)
                .arg(&n)
                .arg(&lr)
                .arg(&beta1)
                .arg(&beta2)
                .arg(&wd)
                .launch(cfg)?;
        }
        updates.push(update);
    }

    Ok(updates)
}
