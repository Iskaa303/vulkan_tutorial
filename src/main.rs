use std::sync::Arc;

use anyhow::{Context, Ok, Result};
use log::info;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::GpuFuture;
use vulkano::{sync, VulkanLibrary};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

fn main() -> Result<()>
{
    env_logger::init();

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

    let command_buffer_allocator = Arc::new(
        StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .context("failed to create an AutoCommandBufferBuilder")?;

    // Example buffer operation

    let source_content: Vec<i32> = (0..64).collect();
    let source = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        source_content,
    )
    .context("failed to create a source buffer from an iterator")?;

    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let destination = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        destination_content,
    )
    .context("failed to create a destination buffer from an iterator")?;
    
    command_buffer_builder
        .copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))
        .context("failed to copy source and destination buffers into an AutoCommandBufferBuilder")?;

    // Example buffer operation end

    let command_buffer = command_buffer_builder
        .build()
        .context("failed to build a PrimaryAutoCommandBuffer")?;

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer.clone())
        .context("failed to execute a command buffer after this future")?
        .then_signal_fence_and_flush()
        .context("failed to signal a fence after this future and flush")?;

    future.wait(None).context("failed to block current thread")?;

    // Example buffer operation continuation

    let src_content = source
        .read()
        .context("failed to read the source content from the buffer")?;

    let dest_content = destination
        .read()
        .context("failed to read the destination content from the buffer")?;

    assert_eq!(&*src_content, &*dest_content);

    info!("everything succeeded!");

    // Example buffer operation end

    Ok(())
}