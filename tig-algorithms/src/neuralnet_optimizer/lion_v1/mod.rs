// Lion optimizer for c006 neuralnet_optimizer.
//
// Single-state momentum optimizer (Chen et al. 2023).
//
// Compared to neural_advanced_v4 (Adam-with-many-tricks-per-track):
//  * One state buffer (m) instead of multiple (m, v, prev_g, prev_u, slow_u, ...)
//  * No bias-correction, no spectral boost, no per-layer LR — keep it simple
//  * Same hyperparameters across all 5 tracks (n_hidden ∈ {4,7,10,14,18})
//
// Defaults: lr=3e-4 (Lion uses ~3-10x smaller LR than Adam), β1=0.9, β2=0.99,
// weight_decay=0.01.

use anyhow::Result;
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub lr: Option<f32>,
    pub beta1: Option<f32>,
    pub beta2: Option<f32>,
    pub weight_decay: Option<f32>,
    pub warmup_steps: Option<usize>,
    pub total_steps: Option<usize>,
}

pub fn help() {
    println!("Lion v1 — single-state momentum optimizer (Chen et al. 2023).");
    println!("  hyperparameters: lr (3e-4), beta1 (0.9), beta2 (0.99), weight_decay (0.01),");
    println!("                   warmup_steps (32), total_steps (1024)");
}

const THREADS_PER_BLOCK: u32 = 256;

#[derive(Clone)]
struct LionState {
    m: Vec<CudaSlice<f32>>,
    cfgs: Vec<LaunchConfig>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    warmup_steps: usize,
    total_steps: usize,
    step_count: usize,
}

impl OptimizerStateTrait for LionState {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
}

// Hardcoded defaults (the training_loop signature requires fn pointers, not
// closures, so we can't pass hyperparameters through. To tune, edit these
// constants and rebuild.)
const LR: f32 = 1e-4;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.99;
const WEIGHT_DECAY: f32 = 0.01;
const WARMUP_STEPS: usize = 64;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
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
    let lr = LR;
    let beta1 = BETA1;
    let beta2 = BETA2;
    let weight_decay = WEIGHT_DECAY;
    let warmup_steps = WARMUP_STEPS;
    let total_steps = 1024;
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
        beta1,
        beta2,
        weight_decay,
        warmup_steps,
        total_steps,
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
    // No lookahead.
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

    // Linear warmup → constant LR. (No cosine decay yet; Lion is robust enough.)
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
        let beta1 = st.beta1;
        let beta2 = st.beta2;
        let wd = st.weight_decay;
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
