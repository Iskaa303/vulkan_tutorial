use std::sync::Arc;

use anyhow::{Context, Ok, Result};
use log::info;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano::{sync, VulkanLibrary};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

fn main() -> Result<()>
{
    env_logger::init();

    info!("Start of the program");

    let library = VulkanLibrary::new().context("no local Vulkan library/DLL")?;
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .context("failed to create instance")?;

    let physical_device = instance
        .enumerate_physical_devices()
        .context("could not enumerate physical devices")?
        .next()
        .context("no devices available")?;

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        })
        .context("couldn't find a graphical queue family")? as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .context("failed to create device")?;

    let queue = queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
        device.clone()
    ));

    // Compute pipeline

    let data_iter = 0..65536u32;
    let data_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data_iter,
    )
    .context("failed to create a buffer from an iterator")?;

    mod compute_shader {
        vulkano_shaders::shader!{
            ty: "compute",
            path: "src/shaders/compute.comp"
        }
    }

    let shader = compute_shader::load(device.clone())
        .context("failed to load a compute shader")?;

    let compute_shader = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(compute_shader);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .context("failed to create PipelineLayoutCreateInfo")?
    )
    .context("failed to create a new PipelineLayout")?;

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout)
    )
    .context("failed to create a new ComputePipeline")?;

    let descriptor_set_allocator = Arc::new(
        StandardDescriptorSetAllocator::new(device.clone(), Default::default())
    );
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .context("failed to get descriptor_set_layout")?;
    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
        [],
    )
    .context("failed to create a new DescriptorSet")?;

    // Command buffer

    let command_buffer_allocator = Arc::new(
        StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .context("failed to create an AutoCommandBufferBuilder")?;

    let work_group_counts = [1024, 1, 1];

    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .context("failed to bind a compute pipeline to a command buffer")?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )
        .context("failed to bind descriptor sets to a command buffer")?;

    unsafe {
        command_buffer_builder
            .dispatch(work_group_counts)
            .context("failed to dispatch work_group_counts")?;
    }
    
    let command_buffer = command_buffer_builder
        .build()
        .context("failed to build a PrimaryAutoCommandBuffer")?;

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer.clone())
        .context("failed to execute a command buffer after this future")?
        .then_signal_fence_and_flush()
        .context("failed to signal a fence after this future and flush")?;

    future.wait(None).context("failed to block current thread")?;

    // Check if the pipeline has been correctly executed
    let content = data_buffer.read().context("failed to read data_buffer")?;
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    info!("Everything succeeded!");

    Ok(())
}