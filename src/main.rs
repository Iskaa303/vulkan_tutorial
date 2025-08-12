use std::sync::Arc;

use anyhow::{Context, Ok, Result};
use image::{ImageBuffer, Rgba};
use log::info;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyImageToBufferInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::format::{ClearColorValue, Format};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
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

    mod compute_shader {
        vulkano_shaders::shader!{
            ty: "compute",
            path: "src/shaders/compute.comp"
        }
    }

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .context("failed to create an image")?;

    let buf = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .context("failed to create a buffer from an iterator")?;

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

    command_buffer_builder
        .clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float([0.0, 0.0, 1.0, 1.0]),
            ..ClearColorImageInfo::image(image.clone())
        })
        .context("failed to clear color image")?
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))
        .context("failed to copy from an image to a buffer")?;
    
    let command_buffer = command_buffer_builder
        .build()
        .context("failed to build a PrimaryAutoCommandBuffer")?;

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer.clone())
        .context("failed to execute a command buffer after this future")?
        .then_signal_fence_and_flush()
        .context("failed to signal a fence after this future and flush")?;

    future.wait(None).context("failed to block current thread")?;

    let buffer_content = buf.read().context("failed to read buffer")?;
    let image = 
        ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..])
    .context("failed to construct an ImageBuffer")?;

    image.save("image.png").context("failed to save an image")?;

    info!("Everything succeeded!");

    Ok(())
}